import pytest

import arkouda as ak
from server_util.test.server_test_util import TestRunningMode, start_arkouda_server


@pytest.mark.skipif(pytest.host == "horizon", reason="nightly test failures due to machine busyness")
class TestClient:
    def test_client_docstrings(self):
        import doctest

        from arkouda import client

        result = doctest.testmod(client, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_client_connected(self):
        """
        Tests the following methods:
        ak.client.connected()
        ak.client.disconnect()
        ak.client.connect()

        :return: None
        :raise: AssertionError if an assert* method returns incorrect value or
                if there is a error in connecting or disconnecting from  the
                Arkouda server
        """
        assert ak.client.connected
        try:
            ak.disconnect()
        except Exception as e:
            raise AssertionError(e)

        assert not ak.client.connected
        try:
            ak.connect(server=pytest.server, port=pytest.port)
        except Exception as e:
            raise AssertionError(e)

        assert ak.client.connected

    def test_disconnect_on_disconnected_client(self):
        """
        Tests the ak.disconnect() method invoked on a client that is already
        disconnect to ensure there is no error
        """
        ak.disconnect()
        assert not ak.client.connected
        ak.disconnect()
        ak.connect(server=pytest.server, port=pytest.port)

    @pytest.mark.skipif(
        pytest.test_running_mode == TestRunningMode.CLIENT,
        reason="start_arkouda_server won't restart if running mode is client",
    )
    def test_shutdown(self):
        """
        Tests the ak.shutdown() method
        """
        ak.shutdown()
        pytest.server, _, _ = start_arkouda_server(numlocales=pytest.nl)
        # reconnect to server so subsequent tests will pass
        ak.connect(server=pytest.server, port=pytest.port, timeout=pytest.timeout)

    def test_client_get_config(self):
        """
        Tests the ak.client.get_config() method

        :return: None
        :raise: AssertionError if one or more Config values are not as expected
                or the call to ak.client.get_config() fails
        """
        try:
            config = ak.client.get_config()
        except Exception as e:
            raise AssertionError(e)
        assert pytest.port == config["ServerPort"]
        assert "arkoudaVersion" in config
        assert "INFO" == config["logLevel"]

        import json

        def get_server_max_array_dims():
            try:
                return max(
                    json.load(open("registration-config.json", "r"))["parameter_classes"]["array"]["nd"]
                )
            except (ValueError, FileNotFoundError, TypeError, KeyError):
                return 1

        assert get_server_max_array_dims() == ak.client.get_max_array_rank()

    def test_get_mem_used(self):
        """
        Tests the ak.get_mem_used and ak.get_mem_avail methods

        :return: None
        :raise: AssertionError if one or more ak.get_mem_used values are not as
                expected or the call to ak.client.get_mem_used() fails
        """
        try:
            config = ak.client.get_config()
            a = ak.ones(1024 * 1024 * config["numLocales"])  # noqa: F841
            mem_used = ak.client.get_mem_used()
        except Exception as e:
            raise AssertionError(e)
        assert mem_used > 0

        # test units
        mem_used = ak.get_mem_used()
        mem_avail = ak.get_mem_avail()
        for u, f in ak.client._memunit2factor.items():
            assert round(mem_used / f) == ak.get_mem_used(u)
            assert round(mem_avail / f) == ak.get_mem_avail(u)

        # test as_percent
        tot_mem = ak.get_mem_used() + ak.get_mem_avail()
        assert ak.get_mem_used(as_percent=True) == round((ak.get_mem_used() / tot_mem) * 100)
        assert ak.get_mem_avail(as_percent=True), round((ak.get_mem_avail() / tot_mem) * 100)

        assert 100 == ak.get_mem_used(as_percent=True) + ak.get_mem_avail(as_percent=True)

    def test_no_op(self):
        """
        Tests the ak.client._no_op method

        :return: None
        :raise: AssertionError if return message is not 'noop'
        """
        assert "noop" == ak.client._no_op()

    def test_ruok(self):
        """
        Tests the ak.client.ruok method

        :return: None
        :raise: AssertionError if return message is not 'imok'
        """
        assert "imok" == ak.client.ruok()

    def test_client_configuration(self):
        """
        Tests the ak.client.set_defaults() method as well as set/get
        parrayIterThresh, maxTransferBytes, and verbose config params.
        """
        ak.client.pdarrayIterThresh = 50
        ak.client.maxTransferBytes = 1048576000
        ak.client.verbose = True
        assert 50 == ak.client.pdarrayIterThresh
        assert 1048576000 == ak.client.maxTransferBytes
        assert ak.client.verbose
        ak.client.set_defaults()
        assert 100 == ak.client.pdarrayIterThresh
        assert 1073741824 == ak.client.maxTransferBytes
        assert not ak.client.verbose

    def test_client_get_server_commands(self):
        """
        Tests the ak.client.get_server_commands() method contains an expected
        sample of commands.
        """
        cmds = ak.client.get_server_commands()
        for cmd in ["connect", "info", "str"]:
            assert cmd in cmds

    def test_get_array_ranks(self):
        availableRanks = ak.client.get_array_ranks()
        assert isinstance(availableRanks, list)
        assert len(availableRanks) >= 1
        assert 1 in availableRanks
        assert ak.client.get_max_array_rank() in availableRanks
        assert ak.client.get_max_array_rank() + 1 not in availableRanks
