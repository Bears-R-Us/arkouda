import pytest
import os
import importlib

import arkouda as ak

from server_util.test.server_test_util import (
    TestRunningMode,
    get_arkouda_numlocales,
    start_arkouda_server,
    stop_arkouda_server,
)

default_dtype = ["int64", "uint64", "float64", "bool", "str", "bigint", "mixed"]
default_encoding = ["ascii", "idna"]
default_compression = [None, "snappy", "gzip", "brotli", "zstd", "lz4"]


def pytest_configure(config):
    pytest.prob_size = eval(config.getoption("size"))
    pytest.trials = eval(config.getoption("trials"))
    pytest.seed = None if config.getoption("seed") == "" else eval(config.getoption("seed"))
    dtype_str = config.getoption("dtype")
    pytest.dtype = default_dtype if dtype_str == "" else dtype_str.split(",")
    pytest.max_bits = eval(config.getoption("maxbits"))
    pytest.alpha = eval(config.getoption("alpha"))
    pytest.random = config.getoption("randomize")
    pytest.numpy = config.getoption("numpy")
    encode_str = config.getoption("encoding")
    pytest.encoding = default_encoding if encode_str == "" else encode_str.split(",")
    pytest.idx_size = None if config.getoption("index_size") == "" else eval(config.getoption("index_size"))
    pytest.val_size = None if config.getoption("value_size") == "" else eval(config.getoption("value_size"))

    # IO settings
    comp_str = config.getoption("io_compression")
    pytest.io_compression = default_compression if comp_str == "" else comp_str.split(",")
    pytest.io_delete = config.getoption("io_only_delete")
    pytest.io_files = eval(config.getoption("io_files_per_loc"))
    pytest.io_path = config.getoption("io_path")
    pytest.io_read = config.getoption("io_only_read")
    pytest.io_write = config.getoption("io_only_write")


@pytest.fixture(scope="module", autouse=True)
def startup_teardown():
    port = int(os.getenv("ARKOUDA_SERVER_PORT", 5555))
    server = os.getenv("ARKOUDA_SERVER_HOST", "localhost")
    test_running_mode = TestRunningMode(os.getenv("ARKOUDA_RUNNING_MODE", "GLOBAL_SERVER"))

    if not importlib.util.find_spec("pytest") or not importlib.util.find_spec("pytest_env"):
        raise EnvironmentError("pytest and pytest-env must be installed")
    if TestRunningMode.CLASS_SERVER == test_running_mode:
        try:
            nl = get_arkouda_numlocales()
            server, _, _ = start_arkouda_server(numlocales=nl, port=port)
            print(
                "Started arkouda_server in TEST_CLASS mode with "
                "host: {} port: {} locales: {}".format(server, port, nl)
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
                server, port
            )
        )

    yield

    if TestRunningMode.CLASS_SERVER == test_running_mode:
        try:
            stop_arkouda_server()
        except Exception:
            pass


@pytest.fixture(scope="function", autouse=True)
def manage_connection():
    port = int(os.getenv("ARKOUDA_SERVER_PORT", 5555))
    server = os.getenv("ARKOUDA_SERVER_HOST", "localhost")
    timeout = int(os.getenv("ARKOUDA_CLIENT_TIMEOUT", 5))
    try:
        ak.connect(
            server=server, port=port, timeout=timeout
        )
    except Exception as e:
        raise ConnectionError(e)

    yield

    try:
        ak.disconnect()
    except Exception as e:
        raise ConnectionError(e)