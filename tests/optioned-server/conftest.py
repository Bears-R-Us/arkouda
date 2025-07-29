# Tests in this directory use the server started with added custom options.
#
# server launched PER MODULE:
#   define the global `module_server_args`
#
# server launched PER CLASS:
#   define the class variable `class_server_args` in each class
#
# Each module must do one or the other, not both.
# These "server_args" variables, when defined, must be lists of strings.
# `test_running_mode` and `ARKOUDA_RUNNING_MODE` are ignored.
#
# `<parent dir>/conftest.py` still applies - except for the fixtures
# overriden by this conftest.
#
# `ARKOUDA_HOME/pytest.opts.ini` configures pytest for this directory.

import importlib
from typing import Iterator

import pytest

import arkouda as ak
from server_util.test.server_test_util import start_arkouda_server, stop_arkouda_server


def _ensure_plugins_installed():
    if not importlib.util.find_spec("pytest") or not importlib.util.find_spec("pytest_env"):
        raise EnvironmentError("pytest and pytest-env must be installed")


@pytest.fixture(scope="session", autouse=True)
def _global_server() -> Iterator[None]:
    _ensure_plugins_installed()
    yield


# convenience shortcuts #


def _module_args(req):
    return req.module.__name__ + ".module_server_args "


def _class_args(req):
    return req.module.__name__ + "." + req.cls.__name__ + ".class_server_args "


def _module_error_msg(req, args):
    return _module_args(req) + "is not a list: " + str(args)


def _class_error_msg(req, args):
    return _class_args(req) + "is not a list: " + str(args)


def _my_start_server(server_args, note):
    host, port, proc = start_arkouda_server(
        numlocales=pytest.nl, port=pytest.port, server_args=server_args
    )
    pytest.server = host
    print(f"Started arkouda_server {note} -nl {pytest.nl} on {host}:{port}")


def _my_stop_server():
    if hasattr(pytest, "server_already_stopped"):
        del pytest.server_already_stopped
        # Without clearing 'connected', the parent fixture manage_connection()
        # invokes ak.disconnect(), which throws a ZMQ error, which breaks testing.
        ak.client.connected = False
    else:
        stop_arkouda_server()


@pytest.fixture(scope="module", autouse=True)
def _module_server(request) -> Iterator[None]:
    module_server_args = getattr(request.module, "module_server_args", None)
    if module_server_args is not None:
        if type(module_server_args) is not list:
            raise TypeError(_module_error_msg(request, module_server_args))
        _my_start_server(module_server_args, "with module_server_args")
        pytest.module_server_launched = True

        yield

        _my_stop_server()
        pytest.module_server_launched = False
    else:
        # arkouda server will start for each class
        pytest.module_server_launched = False

        yield


@pytest.fixture(scope="class", autouse=True)
def _class_server(request) -> Iterator[None]:
    r = request
    class_server_args = getattr(r.cls, "class_server_args", None)
    if class_server_args is not None:
        if pytest.module_server_launched:
            raise RuntimeError("both " + _module_args(r) + "and " + _class_args(r) + "are given")
        if type(class_server_args) is not list:
            raise TypeError(_class_error_msg(r, class_server_args))
        _my_start_server(class_server_args, "with class_server_args")

        yield

        _my_stop_server()
    else:
        if not pytest.module_server_launched:
            raise RuntimeError("neither " + _module_args(r) + "nor " + _class_args(r) + "is given")

        yield


# NB ak.connect() and ak.disconnect() are invoked in the parent conftest.py, see:
#  @pytest.fixture(scope="class", autouse=True)
#  def manage_connection(_class_server):
