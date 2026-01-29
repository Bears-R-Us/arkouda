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
import re
import subprocess

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
        ak.core.client.connected = False
    else:
        stop_arkouda_server()


def _server_help_text() -> str:
    for cmd in (["./arkouda_server", "--help"], ["arkouda_server", "--help"]):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        except Exception:
            pass
    return ""


_HELP = _server_help_text()


def _has_flag(flag: str) -> bool:
    return bool(_HELP) and re.search(rf"--{re.escape(flag)}(?:[=\s]|$)", _HELP) is not None


def _maybe_skip_by_rank_mark(request) -> None:
    mark_gt = request.node.get_closest_marker("skip_if_max_rank_greater_than")
    if mark_gt:
        req = mark_gt.args[0]
        max_rank = getattr(pytest, "max_rank", None)
        if max_rank is not None and max_rank > req:
            pytest.skip(f"max_rank {max_rank} > {req}")

    mark_lt = request.node.get_closest_marker("skip_if_max_rank_less_than")
    if mark_lt:
        req = mark_lt.args[0]
        max_rank = getattr(pytest, "max_rank", None)
        if max_rank is not None and max_rank < req:
            pytest.skip(f"max_rank {max_rank} < {req}")


def _maybe_skip_if_flags_missing(request, server_args: list[str]) -> None:
    missing = []
    for arg in server_args:
        if arg.startswith("--"):
            flag = arg[2:].split("=", 1)[0]
            if not _has_flag(flag):
                missing.append(flag)
    if missing:
        pytest.skip(f"arkouda_server does not support flags: {', '.join(sorted(set(missing)))}")


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

        # NEW: pre-class skip checks
        _maybe_skip_by_rank_mark(request)
        _maybe_skip_if_flags_missing(request, class_server_args)

        _my_start_server(class_server_args, "with class_server_args")

        yield

        _my_stop_server()
    else:
        if not pytest.module_server_launched:
            raise RuntimeError("neither " + _module_args(r) + "nor " + _class_args(r) + "is given")
        yield
