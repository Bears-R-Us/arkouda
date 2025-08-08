import importlib
import os
import unittest

from context import arkouda as ak

from server_util.test.server_test_util import (TestRunningMode,
                                               get_arkouda_numlocales,
                                               start_arkouda_server,
                                               stop_arkouda_server)

"""
ArkoudaTest defines the base Arkouda test logic for starting up the arkouda_server at the
launch of a unittest TestCase and shutting down the arkouda_server at the completion of
the unittest TestCase.

Note: each Arkouda TestCase class extends ArkoudaTest and encompasses 1..n test methods, the
names of which match the pattern 'test*' (e.g., ArkoudaTest.test_arkouda_server).
"""


class ArkoudaTest(unittest.TestCase):

    verbose = True if os.getenv("ARKOUDA_VERBOSE") == "True" else False
    port = int(os.getenv("ARKOUDA_SERVER_PORT", 5555))
    server = os.getenv("ARKOUDA_SERVER_HOST", "localhost")
    test_running_mode = TestRunningMode(os.getenv("ARKOUDA_RUNNING_MODE", "GLOBAL_SERVER"))
    timeout = int(os.getenv("ARKOUDA_CLIENT_TIMEOUT", 5))

    @classmethod
    def setUpClass(cls):
        """
        If in full stack mode, configures and starts the arkouda_server process and sets the
        ArkoudaTest.server class attribute, noop in client mode.

        :return: None
        :raise: EnvironmentError if 1..n required test libraries are missing, RuntimeError
                if there is an error in configuring or starting arkouda_server
        """
        if not importlib.util.find_spec("pytest") or not importlib.util.find_spec("pytest_env"):
            raise EnvironmentError("pytest and pytest-env must be installed")
        if TestRunningMode.CLASS_SERVER == ArkoudaTest.test_running_mode:
            try:
                nl = get_arkouda_numlocales()
                ArkoudaTest.server, _, _ = start_arkouda_server(numlocales=nl, port=ArkoudaTest.port)
                print(
                    "Started arkouda_server in TEST_CLASS mode with "
                    "host: {} port: {} locales: {}".format(ArkoudaTest.server, ArkoudaTest.port, nl)
                )
            except Exception as e:
                raise RuntimeError(
                    "in configuring or starting the arkouda_server: {}, check "
                    + "environment and/or arkouda_server installation",
                    e,
                )
        else:
            print(
                "in client stack test mode with host: {} port: {}".format(
                    ArkoudaTest.server, ArkoudaTest.port
                )
            )

    def setUp(self):
        """
        Connects an Arkouda client for each test case

        :return: None
        :raise: ConnectionError if exception is raised in connecting to arkouda_server
        """
        try:
            ak.client.connect(
                server=ArkoudaTest.server, port=ArkoudaTest.port, timeout=ArkoudaTest.timeout
            )
        except Exception as e:
            raise ConnectionError(e)

    def tearDown(self):
        """
        Disconnects the client connection for each test case
        :return: None
        :raise: ConnectionError if exception is raised in connecting to arkouda_server
        """
        try:
            ak.client.disconnect()
        except Exception as e:
            raise ConnectionError(e)

    @classmethod
    def tearDownClass(cls):
        """
        Shuts down the arkouda_server started in the setUpClass method if test is run
        in full stack mode, noop if in client stack mode.

        :return: None
        """
        if TestRunningMode.CLASS_SERVER == ArkoudaTest.test_running_mode:
            try:
                stop_arkouda_server()
            except Exception:
                pass
