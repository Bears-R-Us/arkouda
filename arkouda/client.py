import asyncio
import json
import os
import warnings
from asyncio.exceptions import CancelledError
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple, Union, cast

from arkouda import __version__, security
from arkouda.logger import LogLevel, getArkoudaLogger
from arkouda.message import (
    MessageFormat,
    MessageType,
    ParameterObject,
    ReplyMessage,
    RequestMessage,
)

__all__ = [
    "connect",
    "disconnect",
    "shutdown",
    "get_config",
    "get_max_array_rank",
    "get_mem_used",
    "get_mem_avail",
    "get_mem_status",
    "wait_for_async_activity",
    "get_server_commands",
    "print_server_commands",
    "generate_history",
    "ruok",
    "exit",
    "async_connect",
]

username = security.get_username()
connected = False
serverConfig = None
registrationConfig = None
verboseDefVal = False
verbose = verboseDefVal
pdarrayIterThreshDefVal = 100
pdarrayIterThresh = pdarrayIterThreshDefVal
sparrayIterThreshDefVal = 20
sparrayIterThresh = sparrayIterThreshDefVal
maxTransferBytesDefVal = 2**30
maxTransferBytes = maxTransferBytesDefVal
regexMaxCaptures: int = -1

logger = getArkoudaLogger(name="Arkouda Client", logLevel=LogLevel.INFO)
clientLogger = getArkoudaLogger(name="Arkouda User Logger", logFormat="%(message)s")


class RequestMode(Enum):
    ASYNC = "ASYNC"
    SYNC = "SYNC"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class ClientMode(Enum):
    UI = "UI"
    API = "API"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class ShellMode(Enum):
    IPYTHON_NOTEBOOK = "TerminalInteractiveShell"
    JUPYTER_NOTEBOOK = "ZMQInteractiveShell"
    REPL_SHELL = "REPL_SHELL"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class RequestStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


def get_shell_mode():
    try:
        return ShellMode(get_ipython().__class__.__name__)
    except NameError:
        return ShellMode.REPL_SHELL


requestMode = RequestMode(os.getenv("ARKOUDA_REQUEST_MODE", "SYNC").upper())
mode = ClientMode(os.getenv("ARKOUDA_CLIENT_MODE", "UI").upper())

if mode == ClientMode.UI:
    try:
        import pyfiglet  # type: ignore

        print("{}".format(pyfiglet.figlet_format("Arkouda")))
        print(f"Client Version: {__version__}")
    except ImportError:
        pass

_memunit2normunit = {
    "bytes": "b",
    "kilobytes": "kb",
    "megabytes": "mb",
    "gigabytes": "gb",
    "terabytes": "tb",
    "petabytes": "pb",
}
_memunit2factor = {"b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12, "pb": 10**15}


def _mem_get_factor(unit: str) -> int:
    unit = unit.lower()
    if unit in _memunit2factor:
        return _memunit2factor[unit]
    for key, normunit in _memunit2normunit.items():
        if key.startswith(unit):
            return _memunit2factor[normunit]
    raise ValueError(
        f"Argument must be one of {set(_memunit2factor.keys()) | set(_memunit2normunit.keys())}"
    )


class ChannelType(Enum):
    ZMQ = "ZMQ"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class Channel:
    def connect(self, timeout: int = 0):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def send_string_message(
        self, cmd: str, recv_binary: bool = False, args: Optional[str] = None, size: int = -1
    ) -> Union[str, memoryview]:
        raise NotImplementedError

    def send_binary_message(
        self,
        cmd: str,
        payload: memoryview,
        recv_binary: bool = False,
        args: Optional[str] = None,
        size: int = -1,
    ) -> Union[str, memoryview]:
        raise NotImplementedError


class ZmqChannel(Channel):
    def __init__(
        self, server: str, port: int, user: str, token: Optional[str], connect_url: Optional[str]
    ):
        import zmq

        self.user = user
        self.token = token
        self.url = connect_url if connect_url else f"tcp://{server}:{port}"
        self.context = zmq.Context()
        self.socket = None

    def connect(self, timeout: int = 0):
        import zmq

        self.socket = self.context.socket(zmq.REQ)
        if timeout > 0:
            self.socket.setsockopt(zmq.SNDTIMEO, timeout * 1000)
            self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
        self.socket.connect(self.url)

    def disconnect(self):
        self.socket.disconnect(self.url)

    def send_string_message(
        self, cmd: str, recv_binary: bool = False, args: Optional[str] = None, size: int = -1
    ):
        msg = RequestMessage(
            user=username, token=self.token, cmd=cmd, format=MessageFormat.STRING, args=args, size=size
        )
        self.socket.send_string(json.dumps(msg.asdict()))
        if recv_binary:
            frame = self.socket.recv(copy=False)
            view = frame.buffer
            if bytes(view[0 : len(b"Error:")]) == b"Error:":
                raise RuntimeError(frame.bytes.decode())
            return view
        raw = self.socket.recv_string()
        reply = ReplyMessage.fromdict(json.loads(raw))
        if reply.msgType == MessageType.ERROR:
            raise RuntimeError(reply.msg)
        if reply.msgType == MessageType.WARNING:
            warnings.warn(reply.msg)
        return reply.msg

    def send_binary_message(
        self,
        cmd: str,
        payload: memoryview,
        recv_binary: bool = False,
        args: Optional[str] = None,
        size: int = -1,
    ):
        import zmq

        msg = RequestMessage(
            user=username, token=self.token, cmd=cmd, format=MessageFormat.BINARY, args=args, size=size
        )
        self.socket.send(f"{json.dumps(msg.asdict())}BINARY_PAYLOAD".encode(), flags=zmq.SNDMORE)
        self.socket.send(payload, copy=False)
        if recv_binary:
            frame = self.socket.recv(copy=False)
            view = frame.buffer
            if bytes(view[0 : len(b"Error:")]) == b"Error:":
                raise RuntimeError(frame.bytes.decode())
            return view
        raw = self.socket.recv_string()
        reply = ReplyMessage.fromdict(json.loads(raw))
        if reply.msgType == MessageType.ERROR:
            raise RuntimeError(reply.msg)
        if reply.msgType == MessageType.WARNING:
            warnings.warn(reply.msg)
        return reply.msg


channelType = ChannelType.ZMQ
channel: Optional[Channel] = None


def get_channel(
    server: str = "localhost",
    port: int = 5555,
    token: Optional[str] = None,
    connect_url: Optional[str] = None,
) -> "Channel":
    if channelType == ChannelType.ZMQ:
        return ZmqChannel(server=server, port=port, user=username, token=token, connect_url=connect_url)
    raise EnvironmentError(f"Invalid channelType {channelType}")


def get_event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def async_connect(
    server: str = "localhost",
    port: int = 5555,
    timeout: int = 0,
    access_token: Optional[str] = None,
    connect_url: Optional[str] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> asyncio.Future:
    if not loop:
        loop = get_event_loop()
    return loop.run_in_executor(None, connect, server, port, timeout, access_token, connect_url)


async def _run_async_connect(
    loop: asyncio.AbstractEventLoop,
    server: str = "localhost",
    port: int = 5555,
    timeout: int = 0,
    access_token: Optional[str] = None,
    connect_url: Optional[str] = None,
) -> asyncio.Future:
    try:
        if not loop:
            loop = get_event_loop()
        future = loop.run_in_executor(None, connect, server, port, timeout, access_token, connect_url)
        while not future.done() and not connected:
            clientLogger.info("connecting...")
            await asyncio.sleep(5)
    except CancelledError:
        future.cancel()
    except KeyboardInterrupt:
        future.cancel()
    finally:
        return future


def connect(
    server: str = "localhost",
    port: int = 5555,
    timeout: int = 0,
    access_token: Optional[str] = None,
    connect_url: Optional[str] = None,
    access_channel: Optional["Channel"] = None,
) -> None:
    global connected, serverConfig, regexMaxCaptures, channel, registrationConfig
    if requestMode == RequestMode.ASYNC:
        try:
            loop = get_event_loop()
            task = loop.create_task(
                _run_async_connect(loop, server, port, timeout, access_token, connect_url)
            )
            loop.run_until_complete(task)
        except KeyboardInterrupt:
            task.cancel()
            loop.run_until_complete(loop.create_task(cancel_task()))
    else:
        cmd = "connect"
        logger.debug(f"[Python] Sending request: {cmd}")
        if access_channel:
            channel = access_channel
        else:
            channel = get_channel(server=server, port=port, token=access_token, connect_url=connect_url)
        channel.connect(timeout)
        return_message = channel.send_string_message(cmd=cmd)
        connected = True
        serverConfig = _get_config_msg()
        if serverConfig["arkoudaVersion"] != __version__:
            warnings.warn(
                f"Version mismatch between client ({__version__}) and server "
                f"({serverConfig['arkoudaVersion']})",
                RuntimeWarning,
            )
        regexMaxCaptures = serverConfig["regexMaxCaptures"]
        registrationConfig = _get_registration_config_msg()
        clientLogger.info(return_message)


async def cancel_task() -> None:
    await asyncio.sleep(0)
    logger.debug("task cancelled")


def _json_args_to_str(json_obj: Optional[Dict] = None) -> Tuple[int, str]:
    j: List[str] = []
    if json_obj is None:
        return 0, json.dumps(j)
    for key, val in json_obj.items():
        if not isinstance(key, str):
            raise TypeError(f"Argument keys are required to be str. Found {type(key)}")
        param = ParameterObject.factory(key, val)
        j.append(json.dumps(param.dict))
    return len(j), json.dumps(j)


def generic_msg(
    cmd: str,
    args: Optional[Dict] = None,
    payload: Optional[memoryview] = None,
    send_binary: bool = False,
    recv_binary: bool = False,
) -> Union[str, memoryview]:
    if not connected:
        raise RuntimeError("client is not connected to a server")
    size, msg_args = _json_args_to_str(args)
    from typing import cast as type_cast

    try:
        if send_binary:
            assert payload is not None
            return type_cast("Channel", channel).send_binary_message(
                cmd=cmd, payload=payload, recv_binary=recv_binary, args=msg_args, size=size
            )
        assert payload is None
        return type_cast("Channel", channel).send_string_message(
            cmd=cmd, args=msg_args, size=size, recv_binary=recv_binary
        )
    except KeyboardInterrupt as e:
        type_cast("Channel", channel).connect(timeout=0)
        raise e


def _get_config_msg() -> Mapping[str, Union[str, int, float]]:
    raw_message = cast(str, generic_msg(cmd="getconfig"))
    return json.loads(raw_message)


def _get_registration_config_msg() -> dict:
    raw_message = cast(str, generic_msg(cmd="getRegistrationConfig"))
    return json.loads(raw_message)


def get_config() -> Mapping[str, Union[str, int, float]]:
    if serverConfig is None:
        raise RuntimeError("client is not connected to a server")
    return serverConfig


def get_max_array_rank() -> int:
    if serverConfig is None:
        raise RuntimeError("client is not connected to a server")
    return max(get_array_ranks())


def get_array_ranks() -> list[int]:
    if registrationConfig is None:
        raise RuntimeError("Registration config missing; connect to server first.")
    return registrationConfig["parameter_classes"]["array"]["nd"]


def wait_for_async_activity() -> None:
    generic_msg("wait_for_async_activity")


def get_mem_used(unit: str = "b", as_percent: bool = False) -> int:
    msg = generic_msg(cmd="getmemused", args={"factor": _mem_get_factor(unit), "as_percent": as_percent})
    return int(cast(str, msg))


def get_mem_avail(unit: str = "b", as_percent: bool = False) -> int:
    msg = generic_msg(
        cmd="getavailmem", args={"factor": _mem_get_factor(unit), "as_percent": as_percent}
    )
    return int(cast(str, msg))


def get_mem_status() -> List[Mapping[str, Union[str, int, float]]]:
    raw = cast(str, generic_msg(cmd="getmemstatus"))
    return json.loads(raw)


def get_server_commands() -> Mapping[str, str]:
    raw = cast(str, generic_msg(cmd="getCmdMap"))
    return json.loads(raw)


def print_server_commands():
    cmds = sorted(get_server_commands().keys())
    print(f"Total available server commands: {len(cmds)}")
    for cmd in cmds:
        print(f"\t{cmd}")


def generate_history(
    num_commands: Optional[int] = None, command_filter: Optional[str] = None
) -> List[str]:
    if get_shell_mode() == ShellMode.REPL_SHELL:
        from arkouda.history import ShellHistoryRetriever

        return ShellHistoryRetriever().retrieve(command_filter, num_commands)
    from arkouda.history import NotebookHistoryRetriever

    return NotebookHistoryRetriever().retrieve(command_filter, num_commands)


def ruok() -> str:
    try:
        res = cast(str, generic_msg(cmd="ruok"))
        return "imok" if res == "imok" else f"imnotok because: {res}"
    except Exception as e:
        return f"ruok did not return response: {str(e)}"


def disconnect() -> None:
    global connected, serverConfig
    if connected:
        msg = "disconnect"
        logger.debug(f"[Python] Sending request: {msg}")
        resp = cast(str, cast("Channel", channel).send_string_message(msg))
        logger.debug(f"[Python] Received response: {resp}")
        channel.disconnect()
        connected = False
        serverConfig = None
        clientLogger.info(resp)
    else:
        clientLogger.info("not connected; cannot disconnect")


def shutdown() -> None:
    global connected, serverConfig
    if not connected:
        raise RuntimeError("not connected, cannot shutdown server")
    msg = "shutdown"
    logger.debug(f"[Python] Sending request: {msg}")
    resp = cast(str, cast("Channel", channel).send_string_message(msg))
    logger.debug(f"[Python] Received response: {resp}")
    channel.disconnect()
    connected = False
    serverConfig = None
    clientLogger.info(resp)


def _no_op() -> str:
    """
    Send a no-op message just to gather round trip time.

    Returns
    -------
    str
        The noop command result

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in executing noop request

    """
    return cast(str, generic_msg(cmd="noop"))


def set_defaults() -> None:
    """Reset client configuration parameters to their default values."""
    global verbose, maxTransferBytes, pdarrayIterThresh
    verbose = verboseDefVal
    pdarrayIterThresh = pdarrayIterThreshDefVal
    maxTransferBytes = maxTransferBytesDefVal
