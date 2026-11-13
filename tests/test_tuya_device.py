"""Test KKTKolbeTuyaDevice protocol detection."""

from __future__ import annotations

import pytest

from custom_components.kkt_kolbe.tuya_device import KKTKolbeTuyaDevice


@pytest.mark.asyncio
async def test_auto_detect_protocol_includes_3_5():
    """Auto-detection must try Tuya protocol 3.5.

    Regression: Issue #8, dernick79 reported Error 914 on all detected
    versions. tinytuya scan of his Ploom hood revealed it speaks 3.5,
    but our auto-detection loop only tested [3.3, 3.4, 3.1, 3.2].
    Devices on 3.5 firmware (newer Tuya devices since ~2024) cannot
    connect locally without 3.5 in the test list.
    """
    import inspect

    source = inspect.getsource(KKTKolbeTuyaDevice._perform_connection)
    # The version test list must include 3.5 alongside the older versions.
    assert "3.5" in source, (
        "Protocol 3.5 missing from auto-detect list — devices on 3.5 firmware "
        "(e.g. Ploom hoods, some 2024+ models) cannot connect"
    )


# --- Local connection keep-alive -------------------------------------------

import asyncio  # noqa: E402
import socket as socket_module  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from custom_components.kkt_kolbe import tuya_device as tuya_device_module  # noqa: E402
from custom_components.kkt_kolbe.exceptions import KKTConnectionError  # noqa: E402


def _connected_device() -> tuple[KKTKolbeTuyaDevice, MagicMock]:
    device = KKTKolbeTuyaDevice("bf735dfe2ad64fba7cpyhn", "192.168.1.100", "1234567890abcdef", version="3.3")
    raw = MagicMock()
    raw.socket = None
    raw.status.return_value = {"dps": {"1": True}}
    device._device = raw
    device._connected = True
    return device, raw


@pytest.mark.asyncio
async def test_drop_device_closes_socket() -> None:
    """Dropping a connection must close tinytuya's socket (no half-open leftovers)."""
    device, raw = _connected_device()
    device._drop_device()
    raw.close.assert_called_once()
    assert device.is_connected is False


@pytest.mark.asyncio
async def test_status_failure_closes_socket() -> None:
    """An exception during status closes the socket before reconnecting."""
    device, raw = _connected_device()
    raw.status.side_effect = OSError("reset by peer")
    with pytest.raises(KKTConnectionError):
        await device.async_get_status()
    raw.close.assert_called_once()
    assert device.is_connected is False


@pytest.mark.asyncio
async def test_device_calls_are_serialized() -> None:
    """Status and writes never run on the socket at the same time."""
    device, raw = _connected_device()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow(*_args):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"dps": {"1": True}}

    raw.status.side_effect = slow
    raw.set_value.side_effect = slow
    await asyncio.gather(device.async_get_status(), device.async_set_dp(1, True), device.async_get_status())
    assert max_active == 1


@pytest.mark.asyncio
async def test_keepalive_configured_on_tinytuya_socket() -> None:
    """TCP keep-alive is set on ``device.socket`` (tinytuya's attribute), once per socket."""
    device, raw = _connected_device()
    sock = MagicMock()
    raw.socket = sock
    await device.async_get_status()
    await device.async_get_status()
    sock.setsockopt.assert_any_call(socket_module.SOL_SOCKET, socket_module.SO_KEEPALIVE, 1)
    keepalive_calls = [c for c in sock.setsockopt.call_args_list if c.args[1] == socket_module.SO_KEEPALIVE]
    assert len(keepalive_calls) == 1

    raw.socket = MagicMock()  # tinytuya reconnected -> new socket gets configured too
    await device.async_get_status()
    raw.socket.setsockopt.assert_any_call(socket_module.SOL_SOCKET, socket_module.SO_KEEPALIVE, 1)


@pytest.mark.asyncio
async def test_heartbeat_sent_when_idle_and_stopped_on_disconnect(monkeypatch) -> None:
    """An idle connection gets HEART_BEATs; disconnect stops the loop."""
    monkeypatch.setattr(tuya_device_module, "HEARTBEAT_INTERVAL", 0.01)
    device, raw = _connected_device()
    device._last_io = 0.0
    device._start_heartbeat()
    for _ in range(50):
        if raw.heartbeat.called:
            break
        await asyncio.sleep(0.01)
    raw.heartbeat.assert_called_with(True)

    await device.async_disconnect()
    assert device._heartbeat_task is None
    raw.close.assert_called()


@pytest.mark.asyncio
async def test_failed_heartbeat_drops_connection(monkeypatch) -> None:
    """A failing heartbeat marks the device disconnected so the next poll reconnects."""
    monkeypatch.setattr(tuya_device_module, "HEARTBEAT_INTERVAL", 0.01)
    device, raw = _connected_device()
    raw.heartbeat.side_effect = OSError("broken pipe")
    device._last_io = 0.0
    device._start_heartbeat()
    for _ in range(50):
        if not device.is_connected:
            break
        await asyncio.sleep(0.01)
    assert device.is_connected is False
    raw.close.assert_called_once()
    device._stop_heartbeat()


# --- Write verification (device ignores a command) -------------------------


@pytest.mark.asyncio
async def test_set_dp_acknowledged_by_echo(monkeypatch) -> None:
    """A response that echoes the written value counts as success without a status read."""
    monkeypatch.setattr(tuya_device_module, "SET_DP_VERIFY_DELAY", 0)
    device, raw = _connected_device()
    raw.set_value.return_value = {"dps": {"1": False}}
    raw.status.reset_mock()
    assert await device.async_set_dp(1, False) is True
    raw.set_value.assert_called_once_with(1, False)
    raw.status.assert_not_called()


@pytest.mark.asyncio
async def test_set_dp_without_ack_verified_by_status(monkeypatch) -> None:
    """No acknowledgement, but the device did apply the value: success after one status read."""
    monkeypatch.setattr(tuya_device_module, "SET_DP_VERIFY_DELAY", 0)
    device, raw = _connected_device()
    raw.set_value.return_value = None
    raw.status.return_value = {"dps": {"1": False}}
    assert await device.async_set_dp(1, False) is True
    assert raw.set_value.call_count == 1
    assert device.get_dp_value(1) is False


@pytest.mark.asyncio
async def test_set_dp_ignored_by_device_returns_false(monkeypatch) -> None:
    """The device keeps its old value: retry once, then report failure instead of success."""
    monkeypatch.setattr(tuya_device_module, "SET_DP_VERIFY_DELAY", 0)
    device, raw = _connected_device()
    raw.set_value.return_value = None
    raw.status.return_value = {"dps": {"1": True}}
    assert await device.async_set_dp(1, False) is False
    assert raw.set_value.call_count == tuya_device_module.SET_DP_ATTEMPTS
    assert device.get_dp_value(1) is True


@pytest.mark.asyncio
async def test_set_dp_echo_with_old_value_triggers_verification(monkeypatch) -> None:
    """A response reporting the old value is not an acknowledgement."""
    monkeypatch.setattr(tuya_device_module, "SET_DP_VERIFY_DELAY", 0)
    device, raw = _connected_device()
    raw.set_value.side_effect = [{"dps": {"10": "three"}}, {"dps": {"10": "off"}}]
    raw.status.return_value = {"dps": {"10": "three"}}
    assert await device.async_set_dp(10, "off") is True
    assert raw.set_value.call_count == 2


@pytest.mark.asyncio
async def test_set_dp_error_dict_and_unverifiable_dp(monkeypatch) -> None:
    """An error response is verified; if the device does not report the DP, keep the old behaviour."""
    monkeypatch.setattr(tuya_device_module, "SET_DP_VERIFY_DELAY", 0)
    device, raw = _connected_device()
    raw.set_value.return_value = {"Error": "Network Error: Device Unreachable", "Err": "905"}
    raw.status.return_value = {"dps": {"4": False}}
    assert await device.async_set_dp(1, False) is True
    assert raw.set_value.call_count == 1


def test_dp_values_equal_tolerates_types() -> None:
    """Bool/str/int representations of the same value compare equal."""
    eq = tuya_device_module._dp_values_equal
    assert eq(True, "true") and eq("1", True) and eq(False, "false") and eq(0, False)
    assert not eq(True, False) and not eq("one", "two")
    assert eq("Three", "three") and eq(5, "5")


# --- Stale messages on the persistent socket -------------------------------


class _FakeTuyaMessage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload


class _FramedDevice:
    """Minimal stand-in for tinytuya: one message per line, like ``_receive()``.

    ``status()`` behaves like tinytuya: it sends a query and returns the next
    message with a payload - whichever is first on the socket.
    """

    def __init__(self, sock: socket_module.socket) -> None:
        self.socket: socket_module.socket | None = sock
        self.force_closed = False

    def _receive(self) -> _FakeTuyaMessage:
        assert self.socket is not None
        data = b""
        while not data.endswith(b"\n"):
            chunk = self.socket.recv(1)
            if not chunk:
                raise ConnectionError("connection closed")
            data += chunk
        return _FakeTuyaMessage(data.strip())

    def _check_socket_close(self, force: bool = False) -> None:
        if force and self.socket is not None:
            self.socket.close()
            self.socket = None
            self.force_closed = True

    def status(self) -> dict[str, str]:
        assert self.socket is not None
        self.socket.sendall(b"query\n")
        while True:
            msg = self._receive()
            if msg.payload:
                return {"answer": msg.payload.decode()}

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None


def _framed_device() -> tuple[KKTKolbeTuyaDevice, _FramedDevice, socket_module.socket]:
    ours, theirs = socket_module.socketpair()
    ours.settimeout(5)
    device = KKTKolbeTuyaDevice("bf735dfe2ad64fba7cpyhn", "192.168.1.100", "1234567890abcdef", version="3.5")
    raw = _FramedDevice(ours)
    device._device = raw
    device._connected = True
    return device, raw, theirs


def test_drain_discards_pending_messages() -> None:
    """Acks and status reports nobody asked for are removed before a request."""
    device, raw, peer = _framed_device()
    try:
        # two heartbeat acks (empty) and two unsolicited status reports
        peer.sendall(b"\n\nreport-1\nreport-2\n")
        time.sleep(0.05)

        assert device._drain_pending_messages(raw) == 4
        stats = device.connection_stats
        assert stats["stale_acks_dropped"] == 2
        assert stats["stale_status_dropped"] == 2
        assert stats["last_stale_status_time"] is not None
        assert stats["stale_resets"] == 0
        # nothing left, socket untouched and its timeout restored
        assert device._drain_pending_messages(raw) == 0
        assert raw.socket is not None
        assert raw.socket.gettimeout() == 5
    finally:
        peer.close()
        raw.close()


def test_drain_ignores_devices_without_real_socket() -> None:
    """Mocks and devices that are not connected are left alone."""
    device, raw = _connected_device()
    assert device._drain_pending_messages(raw) == 0
    raw.socket = MagicMock()
    assert device._drain_pending_messages(raw) == 0
    raw._receive.assert_not_called()


def test_drain_resets_connection_closed_by_device() -> None:
    """A connection the device closed is dropped so the request reconnects."""
    device, raw, peer = _framed_device()
    peer.close()
    time.sleep(0.05)

    assert device._drain_pending_messages(raw) == 0
    assert raw.force_closed is True
    assert raw.socket is None
    assert device.connection_stats["stale_resets"] == 1


def test_drain_resets_connection_on_message_flood(monkeypatch) -> None:
    """More pending messages than the limit: start over instead of reading forever."""
    monkeypatch.setattr(tuya_device_module, "DRAIN_MAX_MESSAGES", 3)
    device, raw, peer = _framed_device()
    try:
        peer.sendall(b"a\nb\nc\nd\ne\n")
        time.sleep(0.05)

        assert device._drain_pending_messages(raw) == 3
        assert raw.force_closed is True
        assert device.connection_stats["stale_resets"] == 1
    finally:
        peer.close()


@pytest.mark.asyncio
async def test_request_gets_its_own_answer_despite_old_reports() -> None:
    """Regression: an old report must not be returned as the answer to a new request.

    Without clearing the socket, ``status()`` returned ``old-report`` and the
    real answer stayed behind for the next poll - every later poll was one
    message late.
    """
    device, raw, peer = _framed_device()

    def _hood() -> None:
        # the device answers each query with the state at that moment
        for answer in (b"fresh-1\n", b"fresh-2\n"):
            data = b""
            while not data.endswith(b"query\n"):
                data += peer.recv(16)
            peer.sendall(answer)

    try:
        peer.sendall(b"\nold-report\n")  # heartbeat ack + report after a change at the panel
        time.sleep(0.05)
        thread = threading.Thread(target=_hood, daemon=True)
        thread.start()

        first = await device._async_device_call(raw.status, timeout=5.0)
        peer.sendall(b"another-old-report\n")
        time.sleep(0.05)
        second = await device._async_device_call(raw.status, timeout=5.0)
        thread.join(timeout=2)

        assert first == {"answer": "fresh-1"}
        assert second == {"answer": "fresh-2"}
        assert device.connection_stats["stale_status_dropped"] == 2
    finally:
        peer.close()
        raw.close()


@pytest.mark.asyncio
async def test_status_keeps_data_points_missing_from_a_report() -> None:
    """A report with only the changed DP must not wipe the other known DPs."""
    device, raw = _connected_device()
    raw.status.return_value = {"dps": {"1": True, "4": False, "10": "off"}}
    await device.async_get_status()
    raw.status.return_value = {"dps": {"10": "three"}}

    assert await device.async_get_status() == {"10": "three"}
    assert device.get_dp_value(1) is True
    assert device.get_dp_value(10) == "three"
