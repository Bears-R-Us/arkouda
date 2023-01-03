# This test verifies that Arkouda's async send infrastructure is actually exercised
# when the client is in asynchronous mode.
#
# Background:
# - In ASYNC mode (`RequestMode.ASYNCHRONOUS`) most client commands
#   are sent using an async event loop instead of blocking sends.
# - In this mode, the client logs periodic "request sent..." progress messages
#   from `_async_send_string_message` while waiting for a server reply.
# - Certain commands (`delete`, `connect`, `getconfig`) are excluded and remain synchronous.
#
# What this test does:
# 1. Forces the global `requestMode` in `arkouda.client` to `ASYNCHRONOUS`
#    so that the async path will be used for eligible commands.
# 2. Monkeypatches `client.asyncio.sleep` to return immediately, so the 5-second
#    logging interval in the async loop is reduced to effectively zero — this
#    makes the test run in milliseconds instead of timing out.
# 3. Subclasses `ZmqChannel` into `AsyncProbeZmqChannel` that replaces the
#    underlying ZeroMQ socket with a fake socket (`FakeSocket`) that:
#       - Returns a minimal valid server reply containing `msgType`, `msg`, and `user`.
#       - Introduces a short delay in `send_string` to give the async loop a
#         chance to iterate and emit progress logs.
# 4. Attaches a temporary in-memory `logging.Handler` directly to
#    `client.clientLogger` (the logger used for async progress output),
#    ensuring we capture exactly the log messages of interest regardless of
#    pytest’s `caplog` configuration.
# 5. Calls `send_string_message("noop")`, which is not in the exclusion list,
#    so it should be routed through the async send path.
# 6. Asserts that at least one of the captured log messages contains the
#    "request sent..." text, proving that:
#       - The async code path ran.
#       - The async loop executed far enough to produce progress output.
#
# This test is deliberately isolated from any real Arkouda server or ZMQ
# dependencies. By faking the channel and the socket, we can test the async
# machinery in a deterministic, fast, and side-effect-free manner.

import asyncio as py_asyncio
import json
import logging
import time

import pytest

from arkouda import client
from arkouda.client import RequestMode, ZmqChannel


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(self.format(record))


class FakeSocket:
    def __init__(self):
        self._reply = json.dumps({"msgType": "NORMAL", "msg": "ok", "user": "u"}).encode()

    def send_string(self, s):
        # Small delay to allow the (patched) async loop to iterate
        time.sleep(0.03)

    def recv_string(self):
        return self._reply.decode()


class AsyncProbeZmqChannel(ZmqChannel):
    def connect(self, timeout=0):
        self.socket = FakeSocket()
        self.heartbeatSocket = None
        self.heartbeatInterval = -1

    def wait_with_heartbeat(self):
        return


@pytest.fixture(autouse=True)
def force_async_mode(monkeypatch):
    # ensure async mode even if client was already imported
    monkeypatch.setenv("ARKOUDA_REQUEST_MODE", "ASYNCHRONOUS")
    monkeypatch.setattr(client, "requestMode", RequestMode.ASYNCHRONOUS, raising=True)


@pytest.fixture(autouse=True)
def fast_async_sleep(monkeypatch):
    # capture the REAL asyncio.sleep before patching
    real_sleep = py_asyncio.sleep

    async def _tiny_sleep(_seconds):
        # yield control to let the executor progress, but stay fast
        await real_sleep(0)

    # patch only the sleep used by the client module
    monkeypatch.setattr(client.asyncio, "sleep", _tiny_sleep, raising=True)


@pytest.fixture
def user_logger_handler():
    h = ListHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    client.clientLogger.addHandler(h)
    client.clientLogger.setLevel(logging.INFO)
    try:
        yield h
    finally:
        client.clientLogger.removeHandler(h)


def test_async_send_progress_logs_with_handler(user_logger_handler):
    ch = AsyncProbeZmqChannel(user="u", server="s", port=1)
    ch.connect()

    ch.send_string_message("noop")

    assert any("request sent..." in m for m in user_logger_handler.messages)
