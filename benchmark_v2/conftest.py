import importlib
import importlib.util
import os
import sys
from typing import Iterable, Iterator

import pytest

import arkouda as ak
from arkouda.client import get_array_ranks
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
        help="arkouda server host for CLIENT running mode",
    )

    default_port = int(os.getenv("ARKOUDA_SERVER_PORT", "5555"))
    parser.addoption(
        "--port",
        action="store",
        default=default_port,
        type=int,
        help="arkouda server port",
    )

    default_nl = get_arkouda_numlocales()
    parser.addoption(
        "--nl",
        action="store",
        default=default_nl,
        help="Number of Locales to run Arkouda with. "
        "Defaults to ARKOUDA_NUMLOCALES, if set, otherwise 2. "
        "If Arkouda is not configured for multi_locale, 1 locale is used.",
    )

    default_running_mode = os.getenv("ARKOUDA_RUNNING_MODE", "CLASS_SERVER")
    parser.addoption(
        "--running_mode",
        action="store",
        default=default_running_mode,
        help="arkouda running mode",
    )

    default_client_timeout = int(os.getenv("ARKOUDA_CLIENT_TIMEOUT", "0"))
    parser.addoption(
        "--client_timeout",
        action="store",
        type=int,
        default=default_client_timeout,
        help="client timeout",
    )

    # NOTE:  Commented out argument that is not yet implemented and therefore has no effect.
    # default_log_level = os.getenv("ARKOUDA_LOG_LEVEL", "DEBUG")
    # parser.addoption(
    #     "--log_level",
    #     action="store",
    #     default=default_log_level,
    #     help="log level",
    # )

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


def pytest_configure(config):
    pytest.host = config.getoption("host")
    pytest.port = config.getoption("port")
    pytest.nl = config.getoption("nl")
    pytest.running_mode = TestRunningMode(config.getoption("running_mode"))
    pytest.verbose = config.getoption("verbose")
    pytest.client_timeout = config.getoption("client_timeout")
    # NOTE:  Commented out argument that is not yet implemented and therefore has no effect.
    # pytest.log_level = config.getoption("log_level")
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


def _ensure_plugins_installed():
    if not importlib.util.find_spec("pytest") or not importlib.util.find_spec("pytest_env"):
        raise EnvironmentError("pytest and pytest-env must be installed")


@pytest.fixture(scope="session", autouse=True)
def _global_server() -> Iterator[None]:
    """
    If we're in GLOBAL_SERVER mode, start a single Arkouda server
    at the beginning of the session and tear it down at the end.

    Yields
    ------
    None
        Fixture point: control will flow back to pytest after server start,
        and then here again once the session is over to stop the server.
    """
    _ensure_plugins_installed()

    if pytest.running_mode == TestRunningMode.GLOBAL_SERVER:
        nl = pytest.nl
        host, port, proc = start_arkouda_server(numlocales=nl, port=pytest.port)
        pytest.host = host
        sys.stdout.write(f"Started arkouda_server in GLOBAL_SERVER mode on {host}:{port} ({nl} locales)")

        try:
            yield
        finally:
            try:
                stop_arkouda_server()
            except Exception:
                pass

    else:
        # no server needed in CLASS_SERVER or CLIENT mode
        yield


@pytest.fixture(scope="module", autouse=True)
def _module_server() -> Iterator[None]:
    """
    If we're in CLASS_SERVER mode, start/stop Arkouda once per module.
    In GLOBAL_SERVER or CLIENT mode this fixture is a no-op.

    Yields
    ------
    None
        Fixture point: control passes to pytest to run the module’s tests,
        then returns here to stop the server after the module is done.
    """
    if pytest.running_mode == TestRunningMode.CLASS_SERVER:
        nl = pytest.nl
        host, port, proc = start_arkouda_server(numlocales=nl, port=pytest.port)
        pytest.host = host
        sys.stdout.write(f"Started arkouda_server in CLASS_SERVER mode on {host}:{port} ({nl} locales)")

        try:
            yield
        finally:
            try:
                stop_arkouda_server()
            except Exception:
                pass
    else:
        # No server needed in GLOBAL_SERVER or CLIENT mode
        yield


@pytest.fixture(autouse=True)
def manage_connection():
    try:
        ak.connect(server=pytest.host, port=pytest.port, timeout=pytest.client_timeout)
        pytest.cfg = ak.get_config()
        pytest.N = pytest.prob_size * pytest.cfg["numLocales"]
    except Exception as e:
        raise ConnectionError(e)

    yield

    try:
        ak.disconnect()
    except Exception as e:
        raise ConnectionError(e)


@pytest.fixture(autouse=True)
def skip_numpy(request):
    marker = request.node.get_closest_marker("skip_numpy")
    if marker and marker.args and marker.args[0] == pytest.numpy:
        pytest.skip(f"{request.node.name} skipped: requires --numpy != {pytest.numpy}")


@pytest.fixture(autouse=True)
def skip_by_rank(request):
    if request.node.get_closest_marker("skip_if_rank_not_compiled"):
        ranks_requested = request.node.get_closest_marker("skip_if_rank_not_compiled").args[0]
        array_ranks = get_array_ranks()
        if isinstance(ranks_requested, int):
            if ranks_requested not in array_ranks:
                pytest.skip("this test requires server compiled with rank {}".format(ranks_requested))
        elif isinstance(ranks_requested, Iterable):
            for i in ranks_requested:
                if isinstance(i, int):
                    if i not in array_ranks:
                        pytest.skip("this test requires server compiled with rank(s) {}".format(i))
                else:
                    raise TypeError("skip_if_rank_not_compiled only accepts type int or list of int.")
        else:
            raise TypeError("skip_if_rank_not_compiled only accepts type int or list of int.")
