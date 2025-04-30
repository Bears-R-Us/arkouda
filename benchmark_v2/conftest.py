import importlib
import os

import pytest

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


def pytest_addoption(parser):
    default_host = os.getenv("ARKOUDA_SERVER_HOST", "localhost")
    parser.addoption(
        "--host",
        action="store",
        default=default_host,
        help="arkouda server host",
    )

    default_port = int(os.getenv("ARKOUDA_SERVER_PORT", 5555))
    parser.addoption(
        "--port",
        action="store",
        default=default_port,
        type=int,
        help="arkouda server port",
    )

    default_running_mode = os.getenv("ARKOUDA_RUNNING_MODE", "CLASS_SERVER")
    parser.addoption(
        "--running_mode",
        action="store",
        default=default_running_mode,
        help="arkouda running mode",
    )

    default_client_timeout = int(os.getenv("ARKOUDA_CLIENT_TIMEOUT", 0))
    parser.addoption(
        "--timeout",
        action="store",
        type=int,
        default=default_client_timeout,
        help="client timeout",
    )

    default_log_level = os.getenv("ARKOUDA_LOG_LEVEL", "DEBUG")
    parser.addoption(
        "--log_level",
        action="store",
        default=default_log_level,
        help="log level",
    )

    parser.addoption(
        "--optional-parquet",
        action="store_true",
        default=False,
        help="run optional parquet tests",
    )

    parser.addoption(
        "--size",
        action="store",
        default="10**8",
        help="Benchmark only option. Problem size: length of array to use for benchmarks.",
    )
    parser.addoption(
        "--trials",
        action="store",
        default="5",
        help="Benchmark only option. Problem size: length of array to use for benchmarks. "
        "For tests that run as many trials as possible in a given time, "
        "will be treated as number of seconds to run for.",
    )
    parser.addoption(
        "--seed",
        action="store",
        default="",
        help="Benchmark only option. Value to initialize random number generator.",
    )
    parser.addoption(
        "--dtype",
        action="store",
        default="",
        help="Benchmark only option. Dtypes to run benchmarks against. Comma separated list "
        "(NO SPACES) allowing for multiple. "
        "Accepted values: int64, uint64, bigint, float64, bool, str and mixed."
        "Mixed is used to generate sets of multiple types.",
    )
    parser.addoption(
        "--numpy",
        action="store_true",
        default=False,
        help="Benchmark only option. When set, runs numpy comparison benchmarks.",
    )
    parser.addoption(
        "--maxbits",
        action="store",
        default="-1",
        help="Benchmark only option. Only applies to bigint testing."
        "Maximum number of bits, so values > 2**max_bits will wraparound. "
        "-1 is interpreted as no maximum.",
    )
    parser.addoption(
        "--alpha",
        action="store",
        default="1.0",
        help="Benchmark only option. Scalar multiple",
    )
    parser.addoption(
        "--randomize",
        action="store_true",
        default=False,
        help="Benchmark only option. Fill arrays with random values instead of ones",
    )
    parser.addoption(
        "--index_size",
        action="store",
        default="",
        help="Benchmark only option. Length of index array (number of gathers to perform)",
    )
    parser.addoption(
        "--value_size",
        action="store",
        default="",
        help="Benchmark only option.Length of array from which values are gathered",
    )
    parser.addoption(
        "--encoding",
        action="store",
        default="",
        help="Benchmark only option. Only applies to encoding benchmarks."
        "Comma separated list (NO SPACES) allowing for multiple"
        "Encoding to be used. Accepted values: idna, ascii",
    )
    parser.addoption(
        "--io_only_write",
        action="store_true",
        default=False,
        help="Benchmark only option. Only write the files; files will not be removed",
    )
    parser.addoption(
        "--io_only_read",
        action="store_true",
        default=False,
        help="Benchmark only option. Only read the files; files will not be removed",
    )
    parser.addoption(
        "--io_only_delete",
        action="store_true",
        default=False,
        help="Benchmark only option. Only delete files created from writing with this benchmark",
    )
    parser.addoption(
        "--io_files_per_loc",
        action="store",
        default="1",
        help="Benchmark only option. Number of files to create per locale",
    )
    parser.addoption(
        "--io_compression",
        action="store",
        default="",
        help="Benchmark only option. Compression types to run IO benchmarks against. "
        "Comma delimited list (NO SPACES) allowing for multiple. "
        "Accepted values: none, snappy, gzip, brotli, zstd, and lz4",
    )
    parser.addoption(
        "--io_path",
        action="store",
        default=os.path.join(os.getcwd(), "ak_io_benchmark"),
        help="Benchmark only option. Target path for measuring read/write rates",
    )
    parser.addoption(
        "--correctness_only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
    )


def pytest_configure(config):
    pytest.host = config.getoption("host")
    pytest.port = config.getoption("port")
    pytest.running_mode = TestRunningMode(config.getoption("running_mode"))
    pytest.verbose = config.getoption("verbose")
    pytest.timeout = config.getoption("timeout")
    pytest.log_level = config.getoption("log_level")
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
    pytest.idx_size = (
        None if config.getoption("index_size") == "" else eval(config.getoption("index_size"))
    )
    pytest.val_size = (
        None if config.getoption("value_size") == "" else eval(config.getoption("value_size"))
    )

    # IO settings
    comp_str = config.getoption("io_compression")
    pytest.io_compression = default_compression if comp_str == "" else comp_str.split(",")
    pytest.io_delete = config.getoption("io_only_delete")
    pytest.io_files = eval(config.getoption("io_files_per_loc"))
    pytest.io_path = config.getoption("io_path")
    pytest.io_read = config.getoption("io_only_read")
    pytest.io_write = config.getoption("io_only_write")

    pytest.correctness_only = config.getoption("correctness_only")


@pytest.fixture(scope="module", autouse=True)
def startup_teardown():
    if not importlib.util.find_spec("pytest") or not importlib.util.find_spec("pytest_env"):
        raise EnvironmentError("pytest and pytest-env must be installed")
    if TestRunningMode.CLASS_SERVER == pytest.running_mode:
        try:
            nl = get_arkouda_numlocales()
            server_host, server_port, process_handle = start_arkouda_server(
                host=pytest.host, numlocales=nl, port=pytest.port
            )
            print(
                "Started arkouda_server in TEST_CLASS mode with host: {} port: {} locales: {}".format(
                    server_host, server_port, nl
                )
            )
        except Exception as e:
            raise RuntimeError(
                "in configuring or starting the arkouda_server: {}, check "
                + "environment and/or arkouda_server installation",
                e,
            )
    else:
        print("in client stack test mode with host: {} port: {}".format(pytest.host, pytest.port))

    yield

    if TestRunningMode.CLASS_SERVER == pytest.running_mode:
        try:
            stop_arkouda_server()
        except Exception:
            pass


@pytest.fixture(scope="function", autouse=True)
def manage_connection():
    try:
        ak.connect(server=pytest.host, port=pytest.port, timeout=pytest.timeout)
    except Exception as e:
        raise ConnectionError(e)

    yield

    try:
        ak.disconnect()
    except Exception as e:
        raise ConnectionError(e)


@pytest.fixture(autouse=True)
def skip_correctness_only(request):
    if request.node.get_closest_marker("skip_correctness_only"):
        if request.node.get_closest_marker("skip_correctness_only").args[0] == pytest.correctness_only:
            pytest.skip("this test requires --correctness_only != {}".format(pytest.correctness_only))


@pytest.fixture(autouse=True)
def skip_numpy(request):
    if request.node.get_closest_marker("skip_numpy"):
        if request.node.get_closest_marker("skip_numpy").args[0] == pytest.numpy:
            pytest.skip("this test requires --numpy != {}".format(pytest.numpy))
