import os

import pytest

from server_util.test.server_test_util import (TestRunningMode,
                                               get_arkouda_numlocales,
                                               start_arkouda_server)


def pytest_addoption(parser):
    parser.addoption(
        "--optional-parquet", action="store_true", default=False, help="run optional parquet tests"
    )

    # due to the order pytest looks for conftest in, options for our benchmarks are added here
    # this has no impact on testing, but allows quick and easy access to settings without the need
    # for env vars. Everything below is for workflows in arkouda/benchmark_v2
    parser.addoption(
        "--size", action="store", default="10**8",
        help="Benchmark only option. Problem size: length of array to use for benchmarks."
    )
    parser.addoption(
        "--trials", action="store", default="5",
        help="Benchmark only option. Number of times to run each test before averaging results. For tests that run "
             "as many trials as possible in a given time, will be treated as number of seconds to run for."
    )
    parser.addoption(
        "--seed", action="store", default="",
        help="Benchmark only option. Value to initialize random number generator."
    )
    parser.addoption(
        "--dtype", action="store", default="",
        help="Benchmark only option. Dtypes to run benchmarks against. Comma separated list "
             "(NO SPACES) allowing for multiple. Accepted values: int64, uint64, bigint, float64, bool, str and mixed."
             "Mixed is used to generate sets of multiple types."
    )
    parser.addoption(
        "--numpy", action="store_true", default=False,
        help="Benchmark only option. When set, runs numpy comparison benchmarks."
    )
    parser.addoption(
        "--maxbits", action="store", default="-1",
        help="Benchmark only option. Only applies to bigint testing."
             "Maximum number of bits, so values > 2**max_bits will wraparound. -1 is interpreted as no maximum."
    )
    parser.addoption(
        "--alpha", action="store", default="1.0",
        help="Benchmark only option. Scalar multiple"
    )
    parser.addoption(
        "--randomize", action="store_true", default=False,
        help="Benchmark only option. Fill arrays with random values instead of ones"
    )
    parser.addoption(
        "--index_size", action="store", default="",
        help="Benchmark only option. Length of index array (number of gathers to perform)"
    )
    parser.addoption(
        "--value_size", action="store", default="",
        help="Benchmark only option. Length of array from which values are gathered"
    )
    parser.addoption(
        "--encoding", action="store", default="",
        help="Benchmark only option. Only applies to encoding benchmarks."
             "Comma separated list (NO SPACES) allowing for multiple"
             "Encoding to be used. Accepted values: idna, ascii"
    )
    parser.addoption(
        "--io_only_write", action="store_true", default=False,
        help="Benchmark only option. Only write the files; files will not be removed"
    )
    parser.addoption(
        "--io_only_read", action="store_true", default=False,
        help="Benchmark only option. Only read the files; files will not be removed"
    )
    parser.addoption(
        "--io_only_delete", action="store_true", default=False,
        help="Benchmark only option. Only delete files created from writing with this benchmark"
    )
    parser.addoption(
        "--io_files_per_loc", action="store", default="1",
        help="Benchmark only option. Number of files to create per locale"
    )
    parser.addoption(
        "--io_compression", action="store", default="",
        help="Benchmark only option. Compression types to run IO benchmarks against. Comma delimited list"
             "(NO SPACES) allowing for multiple. Accepted values: none, snappy, gzip, brotli, zstd, and lz4"
    )
    parser.addoption(
        "--io_path", action="store", default=os.path.join(os.getcwd(), "ak_io_benchmark"),
        help="Benchmark only option. Target path for measuring read/write rates",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--optional-parquet"):
        # --optional-parquet given in cli: do not skip optional parquet tests
        return
    skip_parquet = pytest.mark.skip(reason="need --optional-parquet option to run")
    for item in items:
        if "optional_parquet" in item.keywords:
            item.add_marker(skip_parquet)


def pytest_configure(config):
    config.addinivalue_line("markers", "optional_parquet: mark test as slow to run")
    port = int(os.getenv("ARKOUDA_SERVER_PORT", 5555))
    server = os.getenv("ARKOUDA_SERVER_HOST", "localhost")
    test_server_mode = TestRunningMode(os.getenv("ARKOUDA_RUNNING_MODE", "GLOBAL_SERVER"))

    if TestRunningMode.GLOBAL_SERVER == test_server_mode:
        try:
            nl = get_arkouda_numlocales()
            server, _, _ = start_arkouda_server(numlocales=nl, port=port)
            print(
                (
                    "Started arkouda_server in GLOBAL_SERVER running mode host: {} "
                    + "port: {} locales: {}"
                ).format(server, port, nl)
            )
        except Exception as e:
            raise RuntimeError(
                "in configuring or starting the arkouda_server: {}, check "
                + "environment and/or arkouda_server installation",
                e,
            )
    else:
        print("in client stack test mode with host: {} port: {}".format(server, port))
