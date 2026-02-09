import threading
import time

import pytest

import arkouda as ak

from server_util.test.server_test_util import get_server_info


class TestHeartbeat:
    class_server_args = []  # have 'class_server_args' => start server per class

    def test_heartbeat(self):
        """Ensure the client recognizes server disconnect reasonably soon."""
        hb_timeout = 1
        kill_delay = 0.5
        # reconnect with the specified timeout
        ak.connect(server=pytest.server, port=pytest.port, timeout=hb_timeout)

        def kill_server():
            time.sleep(kill_delay)  # wait for the main thread to issue server_sleep
            get_server_info().process.kill()
            ak.client.connected = False
            pytest.server_already_stopped = True

        kill_thread = threading.Thread(target=kill_server)
        kill_thread.start()
        try:
            # 'keepalive' options in the heartbeat implementation
            # allow 4 times the timeout. Add a wiggle room of 2*timeout.
            ak.client.server_sleep(hb_timeout * 6)
            raise AssertionError("Expected RuntimeError was not raised")
        except Exception as e:
            assert "connection to the server is closed or disconnected" in str(e)
