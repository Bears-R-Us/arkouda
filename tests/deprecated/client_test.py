from base_test import ArkoudaTest
from context import arkouda as ak
from arkouda.client import generic_msg

"""
Tests basic Arkouda client functionality
"""
from server_util.test.server_test_util import start_arkouda_server


class ClientTest(ArkoudaTest):
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
        self.assertTrue(ak.client.connected)
        try:
            ak.disconnect()
        except Exception as e:
            raise AssertionError(e)

        self.assertFalse(ak.client.connected)
        try:
            ak.connect(server=ArkoudaTest.server, port=ArkoudaTest.port)
        except Exception as e:
            raise AssertionError(e)
        self.assertTrue(ak.client.connected)

    def test_disconnect_on_disconnected_client(self):
        """
        Tests the ak.disconnect() method invoked on a client that is already
        disconnect to ensure there is no error
        """
        ak.disconnect()
        self.assertFalse(ak.client.connected)
        ak.disconnect()
        ak.connect(server=ArkoudaTest.server, port=ArkoudaTest.port)

    def test_shutdown(self):
        """
        Tests the ak.shutdown() method
        """
        ak.shutdown()
        start_arkouda_server(numlocales=1)

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
        self.assertEqual(ArkoudaTest.port, config["ServerPort"])
        self.assertTrue("arkoudaVersion" in config)
        self.assertTrue("INFO", config["logLevel"])

    def test_get_mem_used(self):
        """
        Tests the ak.get_mem_used and ak.get_mem_avail methods

        :return: None
        :raise: AssertionError if one or more ak.get_mem_used values are not as
                expected or the call to ak.client.get_mem_used() fails
        """
        try:
            config = ak.client.get_config()
            a = ak.ones(1024 * 1024 * config["numLocales"])
            mem_used = ak.client.get_mem_used()
        except Exception as e:
            raise AssertionError(e)
        self.assertTrue(mem_used > 0)

        # test units
        mem_used = ak.get_mem_used()
        mem_avail = ak.get_mem_avail()
        for u, f in ak.client._memunit2factor.items():
            self.assertEqual(round(mem_used / f), ak.get_mem_used(u))
            self.assertEqual(round(mem_avail / f), ak.get_mem_avail(u))

        # test as_percent
        tot_mem = ak.get_mem_used() + ak.get_mem_avail()
        self.assertEqual(ak.get_mem_used(as_percent=True), round((ak.get_mem_used() / tot_mem) * 100))
        self.assertEqual(ak.get_mem_avail(as_percent=True), round((ak.get_mem_avail() / tot_mem) * 100))

        self.assertEqual(100, ak.get_mem_used(as_percent=True) + ak.get_mem_avail(as_percent=True))

    def test_no_op(self):
        """
        Tests the ak.client._no_op method

        :return: None
        :raise: AssertionError if return message is not 'noop'
        """
        self.assertEqual("noop", ak.client._no_op())

    def test_ruok(self):
        """
        Tests the ak.client.ruok method

        :return: None
        :raise: AssertionError if return message is not 'imok'
        """
        self.assertEqual("imok", ak.client.ruok())

    def test_client_configuration(self):
        """
        Tests the ak.client.set_defaults() method as well as set/get
        parrayIterThresh, maxTransferBytes, and verbose config params.
        """
        ak.client.pdarrayIterThresh = 50
        ak.client.maxTransferBytes = 1048576000
        ak.client.verbose = True
        self.assertEqual(50, ak.client.pdarrayIterThresh)
        self.assertEqual(1048576000, ak.client.maxTransferBytes)
        self.assertTrue(ak.client.verbose)
        ak.client.set_defaults()
        self.assertEqual(100, ak.client.pdarrayIterThresh)
        self.assertEqual(1073741824, ak.client.maxTransferBytes)
        self.assertFalse(ak.client.verbose)

    def test_client_get_server_commands(self):
        """
        Tests the ak.client.get_server_commands() method contains an expected
        sample of commands.
        """
        cmds = ak.client.get_server_commands()
        for cmd in ["connect", "create1D", "info", "str"]:
            self.assertTrue(cmd in cmds)

    def test_client_array_dim_cmd_error(self):
        """
        Tests that a user will get a helpful error message if they attempt to
        use a multi-dimensional command when the server is not configured to
        support multi-dimensional arrays of the given rank.
        """
        from arkouda.client import generic_msg
        with self.assertRaises(RuntimeError) as cm:
            resp = generic_msg("create10D")

        assert str(cm.exception) == "Error: Command 'create10D' is not supported with the current server configuration as the maximum array dimensionality is 1. Please recompile with support for at least 10D arrays"

    def test_client_nd_unimplemented_error(self):
        """
        Tests that a user will get a helpful error message if they attempt to
        use a multi-dimensional command when only a 1D implementation exists.
        """
        from arkouda.client import generic_msg
        with self.assertRaises(RuntimeError) as cm:
            resp = generic_msg("connect2D")

        assert str(cm.exception) == "Error: Command 'connect' is not supported for multidimensional arrays"
