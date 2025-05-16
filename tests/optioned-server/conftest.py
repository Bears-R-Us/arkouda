#wass update module vs class server_opts
# Tests in this directory use the server started with added custom options,
# which are given by the global variable 'server_opts' in each test file.
# //'server_opts' is one way to customize with pytest.
#
# <parent dir>/conftest.py is still used by pytest - except for the fixtures
# overriden by this conftest.
#
# ARKOUDA_HOME/pytest.opts.ini configures pytest for this directory.

# IGNORES test_running_mode / ARKOUDA_RUNNING_MODE

# wass remove some imports?
import importlib
import importlib.util
import os
from typing import Iterator

import pytest

import arkouda as ak
from server_util.test.server_test_util import (
    TestRunningMode,
    get_arkouda_numlocales,
    start_arkouda_server,
    stop_arkouda_server,
)

def _ensure_plugins_installed():
    if not importlib.util.find_spec("pytest") or not importlib.util.find_spec("pytest_env"):
        raise EnvironmentError("pytest and pytest-env must be installed")

@pytest.fixture(scope="session", autouse=True)
def startup_teardown():
    _ensure_plugins_installed()
    yield

# Each module must define either module_server_args and class_server_args.
# The server is launched once per module or once per class, respectively.

# convenience shortcuts

def _module_args(req):
    return req.module.__name__ + ".module_server_args "

def _class_args(req):
    return req.module.__name__ + "." + req.cls.__name__ + ".class_server_args "

def _module_error_msg(req, args):
    return _module_args(req) + "is not a list: " + str(args)

def _class_error_msg(req, args):
    return _class_args(req) + "is not a list: " + str(args)

#wass
#def _module_class_conflict_msg(req):
#    return "both " + _module_args(req) + "and " + _class_args + "are defined"

# returns ServerInfo(host, port, process)
def _my_start_server(server_args):
    print("wass <<<_my_start_server>>>", server_args);
    return start_arkouda_server(numlocales=pytest.nl, port=pytest.port,
                                server_args=server_args)

@pytest.fixture(scope="module", autouse=True)
def _module_server(request) -> Iterator[None]:
    module_server_args = getattr(request.module, "module_server_args", None)
    print("\nwass <<<_module_server>>>", module_server_args)
    if module_server_args is not None:
        if type(module_server_args) is not list:
            raise TypeError(_module_error_msg(request, module_server_args))
        _my_start_server(module_server_args)
        pytest.module_server_launched = True
        yield
        stop_arkouda_server()
        pytest.module_server_launched = False
    else:
        # arkouda server will start for each class
        pytest.module_server_launched = False
        yield

@pytest.fixture(scope="class", autouse=True)
def _class_server(request) -> Iterator[None]:
    r = request
    class_server_args = getattr(r.cls, "class_server_args", None)
    print("\nwass <<<_class_server>>>", class_server_args)
    if class_server_args is not None:
        if pytest.module_server_launched:
            raise RuntimeError("both " + _module_args(r) + "and " + _class_args(r) + "are given")
        if type(class_server_args) is not list:
            raise TypeError(_class_error_msg(r, class_server_args))
        _my_start_server(class_server_args)
        yield
        stop_arkouda_server()
    else:
        if not pytest.module_server_launched:
            raise RuntimeError("neither " + _module_args(r) + "nor " + _class_args(r) + "is given")
        yield

# this fixture in the parent conftest.py runs ak.connect() and ak.disconnect():
#@pytest.fixture(scope="class", autouse=True)
#def manage_connection(_class_server):
