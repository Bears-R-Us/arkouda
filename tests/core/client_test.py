import time

import pytest

import arkouda as ak

from server_util.test.server_test_util import TestRunningMode, start_arkouda_server


@pytest.mark.skipif(
    pytest.client_host == "horizon", reason="nightly test failures due to machine busyness"
)
class TestClient:
    def test_client_docstrings(self):
        import doctest

        from arkouda.core import client

        result = doctest.testmod(client, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_client_connected(self):
        """
        Tests the following methods:
        ak.core.client.connected()
        ak.core.client.disconnect()
        ak.core.client.connect()

        :return: None
        :raise: AssertionError if an assert* method returns incorrect value or
                if there is a error in connecting or disconnecting from  the
                Arkouda server
        """
        assert ak.core.client.connected
        try:
            ak.disconnect()
        except Exception as e:
            raise AssertionError(e)

        assert not ak.core.client.connected
        try:
            ak.connect(server=pytest.server, port=pytest.port)
        except Exception as e:
            raise AssertionError(e)

        assert ak.core.client.connected

    def test_locale_info(self):
        config = ak.core.client.get_config()
        assert config["numLocales"] > 0
        assert config["numNodes"] > 0
        assert config["numPUs"] > 0
        assert config["maxTaskPar"] > 0
        assert config["physicalMemory"] > 0
        assert config["numNodes"] <= config["numLocales"]

    def test_disconnect_on_disconnected_client(self):
        """
        Tests the ak.disconnect() method invoked on a client that is already
        disconnect to ensure there is no error
        """
        ak.disconnect()
        assert not ak.core.client.connected
        ak.disconnect()
        ak.connect(server=pytest.server, port=pytest.port)

    @pytest.mark.skipif(
        pytest.test_running_mode == TestRunningMode.CLIENT,
        reason="should not stop/start the server in the CLIENT mode",
    )
    def test_shutdown(self):
        """Tests the ak.shutdown() method."""
        ak.shutdown()
        pytest.server, _, _ = start_arkouda_server(numlocales=pytest.nl, port=pytest.port)
        # reconnect to server so subsequent tests will pass
        ak.connect(server=pytest.server, port=pytest.port, timeout=pytest.client_timeout)

    def test_server_sleep(self):
        """Tests ak.core.client.server_sleep()."""
        seconds = 1
        start = time.time()
        ak.core.client.server_sleep(seconds)
        end = time.time()
        assert (end - start) >= seconds

    def test_server_note(self):
        """
        Tests ak.core.client.note_for_server_log().

        We could "tail" the server log, however this may give false negatives
        or crashes due to file system delays, server options, etc.
        So simply check that the method is invocable.
        """
        ak.core.client.note_for_server_log("from test_server_note")

    def test_client_get_config(self):
        """
        Tests the ak.core.client.get_config() method

        :return: None
        :raise: AssertionError if one or more Config values are not as expected
                or the call to ak.core.client.get_config() fails
        """
        try:
            config = ak.core.client.get_config()
        except Exception as e:
            raise AssertionError(e)
        assert pytest.port == config["ServerPort"]
        assert "arkoudaVersion" in config
        assert "INFO" == config["logLevel"]

        def get_server_max_array_dims():
            """
            Ask the running server for its registration config and use that
            to determine the maximum array rank it supports.

            This keeps the test in sync with dynamic multi-dim builds where
            the server may extend the rank list beyond what's in the static
            registration-config.json file.
            """
            try:
                reg = ak.core.client.get_registration_config()
                nd_list = reg["parameter_classes"]["array"]["nd"]
                return max(nd_list)
            except Exception:
                # Minimal safe fallback (legacy servers / odd configs)
                return 1

        assert get_server_max_array_dims() == ak.core.client.get_max_array_rank()

    def test_client_get_registration_config(self):
        """
        Tests that we can call ak.core.client.get_registration_config()
        and that the returned result matches the contents of "registration-config.json".

        In multi-dim builds, the server may extend the 'array.nd' list beyond what
        is recorded in the static JSON. We treat the JSON as a baseline and allow
        the server to advertise additional ranks.
        """
        import copy
        import json

        from_server = ak.core.client.get_registration_config()

        try:
            with open("registration-config.json", "r") as f:
                from_file = json.load(f)
        except FileNotFoundError:
            # In installed / packaged environments this file may not be present.
            # This test is specifically about equality with the file, so if the
            # file isn't available, we skip instead of failing.
            pytest.skip("registration-config.json not found; cannot compare against file baseline")

        # Work on copies so we can adjust without mutating originals
        server = copy.deepcopy(from_server)
        file_cfg = copy.deepcopy(from_file)

        # Pop out the dynamically-extended 'nd' list for array parameters
        server_array = server["parameter_classes"]["array"]
        file_array = file_cfg["parameter_classes"]["array"]

        server_nd = server_array.pop("nd", None)
        file_nd = file_array.pop("nd", None)

        # Everything except the dynamic 'nd' list should match exactly
        assert server == file_cfg

        # If both provide an 'nd' list, require that the file's nd values
        # are a subset of the server's values. This allows multi-dim builds
        # (server nd = [1,2,3]) while keeping the file as a valid baseline.
        if server_nd is not None and file_nd is not None:
            assert set(file_nd).issubset(set(server_nd))

    def test_get_mem_used(self):
        """
        Tests the ak.get_mem_used and ak.get_mem_avail methods

        :return: None
        :raise: AssertionError if one or more ak.get_mem_used values are not as
                expected or the call to ak.core.client.get_mem_used() fails
        """
        try:
            config = ak.core.client.get_config()
            a = ak.ones(1024 * 1024 * config["numNodes"])  # noqa: F841
            mem_used = ak.core.client.get_mem_used()
        except Exception as e:
            raise AssertionError(e)
        assert mem_used > 0

        # test units
        mem_used = ak.get_mem_used()
        mem_avail = ak.get_mem_avail()
        for u, f in ak.core.client._memunit2factor.items():
            assert round(mem_used / f) == ak.get_mem_used(u)
            assert round(mem_avail / f) == ak.get_mem_avail(u)

        # test as_percent
        tot_mem = ak.get_mem_used() + ak.get_mem_avail()
        assert ak.get_mem_used(as_percent=True) == round((ak.get_mem_used() / tot_mem) * 100)
        assert ak.get_mem_avail(as_percent=True), round((ak.get_mem_avail() / tot_mem) * 100)

        assert 100 == ak.get_mem_used(as_percent=True) + ak.get_mem_avail(as_percent=True)

    def test_client_configuration(self):
        """
        Tests the ak.core.client.set_defaults() method as well as set/get
        parrayIterThresh, maxTransferBytes, and verbose config params.
        """
        ak.core.client.pdarrayIterThresh = 50
        ak.core.client.maxTransferBytes = 1048576000
        ak.core.client.verbose = True
        assert 50 == ak.core.client.pdarrayIterThresh
        assert 1048576000 == ak.core.client.maxTransferBytes
        assert ak.core.client.verbose
        ak.core.client.set_defaults()
        assert 100 == ak.core.client.pdarrayIterThresh
        assert 1073741824 == ak.core.client.maxTransferBytes
        assert not ak.core.client.verbose

    def test_client_get_server_commands(self):
        """
        Tests the ak.core.client.get_server_commands() method contains an expected
        sample of commands.
        """
        cmds = ak.core.client.get_server_commands()
        for cmd in ["connect", "info", "str"]:
            assert cmd in cmds

    def test_get_array_ranks(self):
        availableRanks = ak.core.client.get_array_ranks()
        assert isinstance(availableRanks, list)
        assert len(availableRanks) >= 1
        assert 1 in availableRanks
        assert ak.core.client.get_max_array_rank() == max(availableRanks)

    def test_no_op(self):
        """
        Tests the ak.core.client._no_op method

        :return: None
        :raise: AssertionError if return message is not 'noop'
        """
        assert "noop" == ak.core.client._no_op()

    def test_ruok(self):
        """
        Tests the ak.core.client.ruok method

        :return: None
        :raise: AssertionError if return message is not 'imok'
        """
        assert "imok" == ak.core.client.ruok()
