import os

import pytest

from server_util.test.server_test_util import TestRunningMode, get_arkouda_numlocales, start_arkouda_server


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
        help="Benchmark only option. Problem size: length of array to use for benchmarks."
    )
    parser.addoption(
        "--seed", action="store", default="",
        help="Benchmark only option. Value to initialize random number generator."
    )
    parser.addoption(
        "--dtype", action="store", default="",
        help="Benchmark only option. Dtypes to run benchmarks against. Comma separated list "
             "(NO SPACES) allowing for multiple. Accepted values: int64, uint64, float64, bool, and str."
    )
    parser.addoption(
        "--numpy", action="store_true", default=False,
        help="Benchmark only option. When set, runs numpy comparison benchmarks."
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
