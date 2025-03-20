import importlib
import os
import subprocess
import sys
from typing import Iterable

import pytest
import scipy

from arkouda import get_max_array_rank
from arkouda.client import get_array_ranks
from server_util.test.server_test_util import (
    is_multilocale_arkouda,  # TODO probably not needed
)
from server_util.test.server_test_util import (
    TestRunningMode,
    start_arkouda_server,
    stop_arkouda_server,
    get_default_temp_directory,
)

os.environ["ARKOUDA_CLIENT_MODE"] = "API"


def pytest_addoption(parser):
    parser.addoption(
        "--optional-parquet",
        action="store_true",
        default=False,
        help="run optional parquet tests",
    )
    parser.addoption(
        "--nl",
        action="store",
        default="2",
        help="Number of Locales to run Arkouda with. "
        "Defaults to 2. If Arkouda is not configured for multi_locale, 1 locale is used",
    )
    parser.addoption(
        "--size",
        action="store",
        default="10**2",
        help="Problem size: length of array to use for tests/benchmarks. For some cases, this will "
        "be multiplied by the number of locales.",
    )
    parser.addoption(
        "--seed",
        action="store",
        default="",
        help="Value to initialize random number generator.",
    )
    parser.addoption(
        "--temp-directory",
        action="store",
        default=get_default_temp_directory(),
        help="Directory to store temporary files.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--optional-parquet"):
        # --optional-parquet given in cli: do not skip optional parquet tests
        return
    skip_parquet = pytest.mark.skip(reason="need --optional-parquet option to run")
    for item in items:
        if "optional_parquet" in item.keywords:
            item.add_marker(skip_parquet)


def _get_test_locales(config):
    """
    Set the number of locales to run Arkouda with.
    The default is 2 provided Arkouda is configured for multi-locale.
    Otherwise, 1 locale will be used
    """
    return eval(config.getoption("nl")) if is_multilocale_arkouda() else 1


def pytest_configure(config):
    config.addinivalue_line("markers", "optional_parquet: mark test as slow to run")
    pytest.port = int(os.getenv("ARKOUDA_SERVER_PORT", 5555))
    pytest.server = os.getenv("ARKOUDA_SERVER_HOST", "localhost")
    pytest.timeout = int(os.getenv("ARKOUDA_CLIENT_TIMEOUT", 5))
    pytest.verbose = bool(os.getenv("ARKOUDA_VERBOSE", False))
    pytest.nl = _get_test_locales(config)
    pytest.seed = None if config.getoption("seed") == "" else eval(config.getoption("seed"))
    pytest.prob_size = [eval(x) for x in config.getoption("size").split(",")]
    pytest.test_running_mode = TestRunningMode(os.getenv("ARKOUDA_RUNNING_MODE", "CLASS_SERVER"))
    pytest.host = subprocess.check_output("hostname").decode("utf-8").strip()

    pytest.temp_directory = config.getoption("--temp-directory")


@pytest.fixture(scope="session", autouse=True)
def startup_teardown():
    test_running_mode = pytest.test_running_mode

    if not importlib.util.find_spec("pytest") or not importlib.util.find_spec("pytest_env"):
        raise EnvironmentError("pytest and pytest-env must be installed")
    if TestRunningMode.CLASS_SERVER == test_running_mode:
        try:
            pytest.server, _, _ = start_arkouda_server(numlocales=pytest.nl, port=pytest.port)
            print(
                "Started arkouda_server in TEST_CLASS mode with "
                "host: {} port: {} locales: {}".format(pytest.server, pytest.port, pytest.nl)
            )
        except Exception as e:
            raise RuntimeError(
                f"in configuring or starting the arkouda_server: {e}, check "
                + "environment and/or arkouda_server installation"
            )
    else:
        print("in client stack test mode with host: {} port: {}".format(pytest.server, pytest.port))

    yield

    if TestRunningMode.CLASS_SERVER == test_running_mode:
        try:
            stop_arkouda_server()
        except Exception:
            pass


@pytest.fixture(scope="class", autouse=True)
def manage_connection():
    import arkouda as ak

    try:
        ak.connect(server=pytest.server, port=pytest.port, timeout=pytest.timeout)
        pytest.max_rank = get_max_array_rank()

    except Exception as e:
        raise ConnectionError(e)

    yield

    try:
        ak.disconnect()
    except Exception as e:
        raise ConnectionError(e)


@pytest.fixture(autouse=True)
def skip_by_rank(request):
    if request.node.get_closest_marker("skip_if_max_rank_less_than"):
        rank_requirement = request.node.get_closest_marker("skip_if_max_rank_less_than").args[0]
        if pytest.max_rank < rank_requirement:
            pytest.skip("this test requires server with max_array_dims >= {}".format(rank_requirement))

    if request.node.get_closest_marker("skip_if_max_rank_greater_than"):
        rank_requirement = request.node.get_closest_marker("skip_if_max_rank_greater_than").args[0]
        if pytest.max_rank > rank_requirement:
            pytest.skip("this test requires server with max_array_dims <= {}".format(rank_requirement))

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


@pytest.fixture(autouse=True)
def skip_by_num_locales(request):
    if request.node.get_closest_marker("skip_if_nl_less_than"):
        nl_requirement = request.node.get_closest_marker("skip_if_nl_less_than").args[0]
        if pytest.nl < nl_requirement:
            pytest.skip("this test requires server with nl <= {}".format(nl_requirement))

    if request.node.get_closest_marker("skip_if_nl_greater_than"):
        nl_requirement = request.node.get_closest_marker("skip_if_nl_greater_than").args[0]
        if pytest.nl > nl_requirement:
            pytest.skip("this test requires server with nl <= {}".format(nl_requirement))

    if request.node.get_closest_marker("skip_if_nl_eq"):
        nl_requirement = request.node.get_closest_marker("skip_if_nl_eq").args[0]
        if nl_requirement == pytest.nl:
            pytest.skip("this test requires server with nl != {}".format(nl_requirement))

    if request.node.get_closest_marker("skip_if_nl_neq"):
        nl_requirement = request.node.get_closest_marker("skip_if_nl_neq").args[0]
        if nl_requirement != pytest.nl:
            pytest.skip("this test requires server with nl == {}".format(nl_requirement))


@pytest.fixture(autouse=True)
def skip_by_python_version(request):
    if request.node.get_closest_marker("skip_if_python_version_less_than"):
        python_requirement = request.node.get_closest_marker("skip_if_python_version_less_than").args[0]
        if sys.version_info < python_requirement:
            pytest.skip("this test requires python version >= {}".format(python_requirement))

    if request.node.get_closest_marker("skip_if_python_version_greater_than"):
        python_requirement = request.node.get_closest_marker("skip_if_python_version_greater_than").args[
            0
        ]
        if sys.version_info > python_requirement:
            pytest.skip("this test requires python version =< {}".format(python_requirement))

    if request.node.get_closest_marker("skip_if_python_version_eq"):
        python_requirement = request.node.get_closest_marker("skip_if_python_version_eq").args[0]
        if sys.version_info == python_requirement:
            pytest.skip("this test requires python version != {}".format(python_requirement))

    if request.node.get_closest_marker("skip_if_python_version_neq"):
        python_requirement = request.node.get_closest_marker("skip_if_python_version_neq").args[0]
        if sys.version_info != python_requirement:
            pytest.skip("this test requires python version == {}".format(python_requirement))


@pytest.fixture(autouse=True)
def skip_by_scipy_version(request):
    if request.node.get_closest_marker("skip_if_scipy_version_less_than"):
        scipy_requirement = request.node.get_closest_marker("skip_if_scipy_version_less_than").args[0]
        if scipy.__version__ < scipy_requirement:
            pytest.skip("this test requires scipy version >= {}".format(scipy_requirement))

    if request.node.get_closest_marker("skip_if_scipy_version_greater_than"):
        scipy_requirement = request.node.get_closest_marker("skip_if_scipy_version_greater_than").args[0]
        if scipy.__version__ > scipy_requirement:
            pytest.skip("this test requires scipy version =< {}".format(scipy_requirement))

    if request.node.get_closest_marker("skip_if_scipy_version_eq"):
        scipy_requirement = request.node.get_closest_marker("skip_if_scipy_version_eq").args[0]
        if scipy.__version__ == scipy_requirement:
            pytest.skip("this test requires scipy version != {}".format(scipy_requirement))

    if request.node.get_closest_marker("skip_if_scipy_version_neq"):
        scipy_requirement = request.node.get_closest_marker("skip_if_scipy_version_neq").args[0]
        if scipy.__version__ != scipy_requirement:
            pytest.skip("this test requires scipy version == {}".format(scipy_requirement))
