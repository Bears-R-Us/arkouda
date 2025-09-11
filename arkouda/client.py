"""
Client interface for connecting to and communicating with the Arkouda server.

The `arkouda.client` module provides the core logic for managing a client-server
session in Arkouda. It includes methods to connect, disconnect, send commands,
check system status, and retrieve configuration details from the server. The client
communicates with the server via ZMQ by default and handles both string and binary
message formats.

Key Responsibilities
--------------------
- Establish and manage server connections via different `Channel` types (e.g., ZMQ)
- Format and send commands using Arkouda’s request-response protocol
- Retrieve memory and configuration metrics from the server
- Provide a health check (`ruok`) and shutdown mechanism
- Maintain client-side logging, verbosity, and session parameters
- (Optional) Provide async-style request *sending* and an `async_connect()` helper

Classes
-------
Channel
    Abstract base class for communication between the client and the server.
ZmqChannel
    Default implementation of `Channel` using ZeroMQ request-reply pattern.
ClientMode, ShellMode, RequestMode, ChannelType
    Enum classes defining modes of client interaction, shell type detection, and channel selection.

Functions
---------
connect(...)
    Establish a connection to an Arkouda server.
disconnect()
    Cleanly disconnect from the server and reset session state.
shutdown()
    Shut down the server, delete its symbol table, and disconnect the client.
get_config()
    Return server runtime configuration and environment settings.
get_mem_used(), get_mem_avail(), get_mem_status()
    Retrieve memory usage and availability statistics from the server.
get_server_commands()
    Get a mapping of available server commands and their functions.
print_server_commands()
    Print a list of all supported server-side commands.
generate_history(...)
    Retrieve interactive shell or notebook command history.
ruok()
    Send a health check to the server ("ruok") and receive status.

Notes
-----
- This module is foundational to all Arkouda workflows.
- The `generic_msg()` function handles message transmission.
- Clients must call `connect()` (or `async_connect()`) before performing any operations.
- The async additions are fully backward-compatible and purely opt‑in via env var.

Examples
--------
>>> import arkouda as ak
>>> ak.connect()  # doctest: +SKIP
>>> ak.get_config()  # doctest: +SKIP
{'serverHostname': 'localhost', 'numLocales': 4, ...}
>>> ak.disconnect()  # doctest: +SKIP

"""

from enum import Enum
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Tuple, Union, cast
import warnings

import zmq  # for typechecking

from arkouda import __version__, security
from arkouda.logger import ArkoudaLogger, LogLevel, getArkoudaLogger
from arkouda.message import (
    MessageFormat,
    MessageType,
    ParameterObject,
    ReplyMessage,
    RequestMessage,
)
from arkouda.pandas import io_util

__all__ = [
    "connect",
    "disconnect",
    "shutdown",
    "get_config",
    "get_registration_config",
    "get_max_array_rank",
    "get_mem_used",
    "get_mem_avail",
    "get_mem_status",
    "get_server_commands",
    "print_server_commands",
    "generate_history",
    "ruok",
]

username = security.get_username()
connected = False
serverConfig = None
registrationConfig = None
# verbose flag for arkouda module
verboseDefVal = False
verbose = verboseDefVal
# threshold for __iter__() to limit comms to arkouda_server
pdarrayIterThreshDefVal = 100
pdarrayIterThresh = pdarrayIterThreshDefVal
sparrayIterThreshDefVal = 20  # sps print format is longer; we have small thresh
sparrayIterThresh = sparrayIterThreshDefVal

maxTransferBytesDefVal = 2**30
maxTransferBytes = maxTransferBytesDefVal
# maximum number of capture group for regex
regexMaxCaptures: int = -1
# unit conversion for get_mem_used
_memunit2normunit = {
    "bytes": "b",
    "kilobytes": "kb",
    "megabytes": "mb",
    "gigabytes": "gb",
    "terabytes": "tb",
    "petabytes": "pb",
}
_memunit2factor = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "pb": 10**15,
}


def _mem_get_factor(unit: str) -> int:
    """
    Normalize a memory unit to a multiplier.

    Parameters
    ----------
    unit : str
        Unit string. Accepts canonical short forms (``'b','kb','mb','gb','tb','pb'``)
        or long forms (``'bytes','kilobytes',...``). Long forms may be abbreviated
        so long as the beginning matches (e.g., ``'kilo'`` → ``'kb'``).

    Returns
    -------
    int
        Multiplicative factor for converting the unit to bytes.

    Raises
    ------
    ValueError
        If the unit is not recognized.

    """
    unit = unit.lower()

    if unit in _memunit2factor:
        return _memunit2factor[unit]
    else:
        for key, normunit in _memunit2normunit.items():
            if key.startswith(unit):
                return _memunit2factor[normunit]
        raise ValueError(
            f"Argument must be one of {set(_memunit2factor.keys()) | set(_memunit2normunit.keys())}"
        )


logger = getArkoudaLogger(name="Arkouda Client", logLevel=LogLevel.INFO)
clientLogger = getArkoudaLogger(name="Arkouda User Logger", logFormat="%(message)s")


class ClientMode(Enum):
    """
    Provide controlled vocabulary indicating whether the Arkouda client is in UI mode or API mode.

    If in API mode, it is assumed the Arkouda client is being used via an API call
    instead of a Python shell or notebook.
    """

    UI = "UI"
    API = "API"

    def __str__(self) -> str:
        """Return the enum value."""
        return self.value

    def __repr__(self) -> str:
        """Return the enum value."""
        return self.value


class ShellMode(Enum):
    """Indicate whether the Python shell corresponds to a Jupyter notebook, or REPL or IPython shell."""

    IPYTHON_NOTEBOOK = "TerminalInteractiveShell"
    JUPYTER_NOTEBOOK = "ZMQInteractiveShell"
    REPL_SHELL = "REPL_SHELL"

    def __str__(self) -> str:
        """Return the enum value."""
        return self.value

    def __repr__(self) -> str:
        """Return the enum value."""
        return self.value


class RequestMode(Enum):
    """
    The Arkouda client-server communication pattern (synchronous or asynchronous).

    Values
    ------
    SYNCHRONOUS
        Standard blocking send/receive (default).
    ASYNCHRONOUS
        Non-blocking *send* using a background executor with periodic progress
        logs while awaiting the server's response. Receive remains blocking.

    """

    SYNCHRONOUS = "SYNCHRONOUS"
    ASYNCHRONOUS = "ASYNCHRONOUS"

    def __str__(self) -> str:
        """Return the enum value."""
        return self.value

    def __repr__(self) -> str:
        """Return the enum value."""
        return self.value


def get_shell_mode():
    """
    Determine the Python shell type and return the `ShellMode` enum.

    Returns
    -------
    ShellMode
        The shell mode corresponding to a Python shell, Jupyter notebook,
        or IPython notebook.

    """
    shell_mode = None
    try:
        shell_mode = ShellMode(get_ipython().__class__.__name__)
    except NameError:
        shell_mode = ShellMode.REPL_SHELL
    finally:
        return shell_mode


# Get ClientMode, defaulting to UI
mode = ClientMode(os.getenv("ARKOUDA_CLIENT_MODE", "UI").upper())

# Print splash message if in UI mode
if mode == ClientMode.UI:
    import pyfiglet  # type: ignore

    sys.stdout.write("{}".format(pyfiglet.figlet_format("Arkouda")))
    sys.stdout.write(f"Client Version: {__version__}")  # type: ignore


def set_defaults() -> None:
    """Set client variables including verbose, maxTransferBytes, pdarrayIterThresh to default values."""
    global verbose, maxTransferBytes, pdarrayIterThresh
    verbose = verboseDefVal
    pdarrayIterThresh = pdarrayIterThreshDefVal
    maxTransferBytes = maxTransferBytesDefVal


class ChannelType(Enum):
    """Specify which Channel implementation is to be used for an Arkouda client deployment."""

    ZMQ = "ZMQ"
    GRPC = "GRPC"
    ASYNC_GRPC = "ASYNC_GRPC"
    STREAMING_GRPC = "STREAMING_GRPC"

    def __str__(self) -> str:
        """Return the enum value."""
        return self.value

    def __repr__(self) -> str:
        """Return the enum value."""
        return self.value


class Channel:
    """
    Define methods for connecting to and communicating with the Arkouda server.

    Attributes
    ----------
    url : str
        Channel url used to connect to the Arkouda server which is either set
        to the connect_url or generated from supplied server and port values.
    user : str
        Arkouda user who will use the Channel to connect to the arkouda_server.
    token : Union[str, None]
        Token used to connect to the arkouda_server if authentication is enabled.
    logger : ArkoudaLogger
        ArkoudaLogger used for logging.

    """

    url: str
    user: str
    token: Union[str, None]
    logger: ArkoudaLogger

    __slots__ = ("url", "user", "token", "logger")

    def __init__(
        self,
        user: str,
        server: str = "localhost",
        port: int = 5555,
        token: Optional[str] = None,
        connect_url: Optional[str] = None,
    ) -> None:
        """
        Initialize a channel for connecting to and communicating with the Arkouda server.

        Parameters
        ----------
        user : str
            Arkouda user who will use the Channel to connect to the arkouda_server.
        server : str, default 'localhost'
            The hostname of the server (must be visible to the current machine).
        port : int, default 5555
            The port of the server.
        token : str, optional
            Token used to connect to the arkouda_server if authentication is enabled.
        connect_url : str, optional
            Complete URL in the form ``tcp://server:port?token=<token_value>``
            where the token is optional.

        """
        self._set_url(server, port, connect_url)
        self.user = user
        self._set_access_token(server, port, token)
        self.logger = getArkoudaLogger(name="Arkouda Client")

    def _set_url(self, server: str, port: int, connect_url: Optional[str] = None) -> None:
        """
        Generate and set the channel URL.

        If `connect_url` is None, generate the channel URL per protocol/host/port;
        otherwise, use the supplied `connect_url`.

        Parameters
        ----------
        server : str
            Arkouda server hostname, ip address, or service name
        port : int
            Arkouda server host port
        connect_url : str, optional
            The complete url in the format of tcp://server:port?token=<token_value>
            where the token is optional

        """
        self.url = connect_url if connect_url else f"tcp://{server}:{port}"

    def _set_access_token(self, server: str, port: int, token: Optional[str]) -> None:
        """
        Set the access token used by the channel.

        Behavior:
          1) Try to read the token for ``{server}:{port}`` from ``~/.arkouda/tokens.txt``.
          2) If `token` is None, keep the retrieved token.
          3) If `token` is not None, write/update the token value for the URL.

        Parameters
        ----------
        server : str
            The hostname of the server (must be visible to the current machine)
        port : int
            The port of the server
        token : str, optional
            The token supplied by the user, which is required if authentication
            is enabled, defaults to None

        Raises
        ------
        IOError
            If there is an error reading/writing the tokens file.

        """
        path = f"{security.get_arkouda_client_directory()}/tokens.txt"
        url = f"{server}:{port}"

        try:
            tokens = io_util.delimited_file_to_dict(path)
        except Exception as e:
            raise IOError(e)

        if cast(str, token) and cast(str, token) not in {"", "None"}:
            saved_token = tokens.get(url)
            if saved_token is None or saved_token != token:
                tokens[url] = cast(str, token)
                try:
                    io_util.dict_to_delimited_file(values=tokens, path=path, delimiter=",")
                except Exception as e:
                    raise IOError(e)
            self.token = token
        else:
            try:
                tokens = io_util.delimited_file_to_dict(path)
            except Exception as e:
                raise IOError(e)
            self.token = tokens.get(url)

    def send_string_message(
        self,
        cmd: str,
        recv_binary: bool = False,
        args: Optional[str] = None,
        size: int = -1,
        request_id: Optional[str] = None,
    ):
        """
        Generate and send a string `RequestMessage` to the Arkouda server.

        Parameters
        ----------
        cmd : str
            Server command name to execute.
        recv_binary : bool, default False
            If True, expect a binary reply (memoryview).
        args : str, optional
            JSON array string of serialized `ParameterObject` entries.
        size : int, default -1
            Number of parameters contained in `args`. Provided for future use.
        request_id : str, optional
            Optional request identifier (no-op for now).

        Raises
        ------
        RuntimeError
            On server-side error reply.
        ValueError
            If the reply is malformed/invalid JSON or missing fields.

        """
        raise NotImplementedError("send_string_message must be implemented in derived class")

    def send_binary_message(
        self,
        cmd: str,
        payload: memoryview,
        recv_binary: bool = False,
        args: Optional[str] = None,
        size: int = -1,
        request_id: Optional[str] = None,
    ):
        """
        Generate and send a binary `RequestMessage` to the Arkouda server.

        Parameters
        ----------
        cmd : str
            Server command name to execute.
        payload : memoryview
            Binary payload to send.
        recv_binary : bool, default False
            If True, expect a binary reply (memoryview).
        args : str, optional
            JSON array string of serialized `ParameterObject` entries.
        size : int, default -1
            Number of parameters contained in `args`. Provided for future use.
        request_id : str, optional
            Optional request identifier (no-op for now).

        Raises
        ------
        RuntimeError
            On server-side error reply.
        ValueError
            If the reply is malformed/invalid JSON or missing fields.

        """
        raise NotImplementedError("send_binary_message must be implemented in derived class")

    def connect(self, timeout: int = 0) -> None:
        """
        Establish a connection to the Arkouda server.

        Parameters
        ----------
        timeout : int, default 0
            Timeout in seconds. If positive, also enables heartbeat (ZMQ).

        Raises
        ------
        RuntimeError
            On server-side error.

        """
        raise NotImplementedError("connect must be implemented in derived class")

    def disconnect(self) -> None:
        """
        Disconnect from the Arkouda server.

        Raises
        ------
        RuntimeError
            On error during disconnect.

        """
        raise NotImplementedError("connect must be implemented in derived class")


class ZmqChannel(Channel):
    """
    Implement the `Channel` methods for ZMQ request/reply communication patterns.

    ZMQ is the default Arkouda transport.
    """

    socket: zmq.Socket
    heartbeatSocket: Optional[zmq.Socket]
    heartbeatInterval: int

    __slots__ = ("socket", "heartbeatSocket", "heartbeatInterval")

    # --------- Core send/recv ---------

    def send_string_message(
        self,
        cmd: str,
        recv_binary: bool = False,
        args: Optional[str] = None,
        size: int = -1,
        request_id: Optional[str] = None,
    ) -> Union[str, memoryview]:
        """
        Generate and send a string `RequestMessage` to the Arkouda server.

        See `Channel.send_string_message` for full parameter and return docs.
        """
        message = RequestMessage(
            user=username, token=self.token, cmd=cmd, format=MessageFormat.STRING, args=args, size=size
        )
        # Note - Size is a placeholder here because Binary msg not yet support json args and
        # request_id is a noop for now
        logger.debug(f"sending message {json.dumps(message.asdict())}")

        self.socket.send_string(json.dumps(message.asdict()))

        self.wait_with_heartbeat()

        if recv_binary:
            frame = self.socket.recv(copy=False)
            view = frame.buffer
            # raise errors sent back from the server
            if bytes(view[0 : len(b"Error:")]) == b"Error:":
                raise RuntimeError(frame.bytes.decode())
            return view
        else:
            raw_message = self.socket.recv_string()
            try:
                return_message = ReplyMessage.fromdict(json.loads(raw_message))
                # raise errors or warnings sent back from the server
                if return_message.msgType == MessageType.ERROR:
                    raise RuntimeError(return_message.msg)
                elif return_message.msgType == MessageType.WARNING:
                    warnings.warn(return_message.msg)
                return return_message.msg
            except KeyError as ke:
                raise ValueError(f"Return message is missing the {ke} field")
            except json.decoder.JSONDecodeError:
                raise ValueError(f"Return message is not valid JSON: {raw_message}")

    def send_binary_message(
        self,
        cmd: str,
        payload: memoryview,
        recv_binary: bool = False,
        args: Optional[str] = None,
        size: int = -1,
        request_id: Optional[str] = None,
    ) -> Union[str, memoryview]:
        """
        Generate and send a binary `RequestMessage` to the Arkouda server.

        See `Channel.send_binary_message` for full parameter and return docs.
        """
        message = RequestMessage(
            user=username, token=self.token, cmd=cmd, format=MessageFormat.BINARY, args=args, size=size
        )
        import zmq

        self.logger.debug(f"sending message {message}")

        self.socket.send(f"{json.dumps(message.asdict())}BINARY_PAYLOAD".encode(), flags=zmq.SNDMORE)
        self.socket.send(payload, copy=False)
        self.wait_with_heartbeat()

        if recv_binary:
            frame = self.socket.recv(copy=False)
            view = frame.buffer
            # raise errors sent back from the server
            if bytes(view[0 : len(b"Error:")]) == b"Error:":
                raise RuntimeError(frame.bytes.decode())
            return view
        else:
            raw_message = self.socket.recv_string()
            try:
                return_message = ReplyMessage.fromdict(json.loads(raw_message))
                # raise errors or warnings sent back from the server
                if return_message.msgType == MessageType.ERROR:
                    raise RuntimeError(return_message.msg)
                elif return_message.msgType == MessageType.WARNING:
                    warnings.warn(return_message.msg)
                return return_message.msg
            except KeyError as ke:
                raise ValueError(f"Return message is missing the {ke} field")
            except json.decoder.JSONDecodeError:
                raise ValueError(f"{raw_message} is not valid JSON, may be server-side error")

    def connect(self, timeout: int = 0) -> None:
        """
        Establish a connection to the Arkouda server using ZMQ.

        Parameters
        ----------
        timeout : int, default 0
            If > 0, sets send/recv timeouts and enables heartbeat monitoring.

        """
        import zmq

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.setup_heartbeat(timeout)  # enables heartbeat if timeout > 0

        logger.debug(f"ZMQ version: {zmq.zmq_version()}")

        # if timeout is specified, set send and receive timeout params
        if timeout > 0:
            self.socket.setsockopt(zmq.SNDTIMEO, timeout * 1000)
            self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)

        try:
            self.socket.connect(self.url)
            # connect() may return right away even if a connection was not
            # established. This case will be caught by the heartbeat.
        except Exception as e:
            raise ConnectionError(e)

    def disconnect(self) -> None:
        """Disconnect the ZMQ socket and tear down heartbeat monitoring."""
        try:
            self.socket.disconnect(self.url)
            self.setup_heartbeat(0)  # close the socket
        except Exception as e:
            raise RuntimeError(e)

    def setup_heartbeat(self, timeout) -> None:
        """
        Turn on/off a server heartbeat with the given timeout (in seconds).

        The timeout is used both for waiting for a response and as the
        heartbeat frequency. If timeout <= 0, heartbeat is disabled.

        Parameters
        ----------
        timeout : int
            Timeout in seconds. Also determines the heartbeat frequency.
            If timeout <= 0, heartbeat monitoring is disabled.

        Notes
        -----
        Must be invoked *before* `self.socket.connect()` for the socket
        options to take effect.

        """
        import zmq

        if timeout <= 0:
            if hasattr(self, "heartbeatSocket") and self.heartbeatSocket is not None:
                self.heartbeatSocket.close()
            self.heartbeatSocket = None
            self.heartbeatInterval = -1
            return

        self.heartbeatInterval = timeout
        timeout = max(int(timeout), 1)  # whole seconds

        # Given the settings below, an exception will be raised within
        # (4 * timeout) seconds. We can tune these.
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)  # turn on "keepalive"
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, timeout)  # seconds of idle before probing
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, timeout)  # interval between probes
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)  # probes before dropping

        # Attach a monitor socket.
        self.heartbeatSocket = self.socket.get_monitor_socket(zmq.EVENT_CLOSED | zmq.EVENT_DISCONNECTED)

        if not self.heartbeat_is_enabled():
            raise RuntimeError("heartbeat_is_enabled() returned false unexpectedly")

    def heartbeat_is_enabled(self) -> bool:
        """
        Check if the heartbeat is enabled.

        Returns
        -------
        bool
            True if enabled, False otherwise.

        """
        return self.heartbeatInterval > 0 and self.heartbeatSocket is not None

    def wait_with_heartbeat(self) -> None:
        """
        Wait for a server response while checking if the server is alive.

        Returns immediately if heartbeat is not active.

        Raises
        ------
        ConnectionError
            If the monitor socket reports a disconnect/close event.

        """
        if not self.heartbeat_is_enabled():
            return

        import zmq

        while True:
            if self.socket.poll(self.heartbeatInterval * 1000, zmq.POLLIN):
                return  # got a response

            # If no response, check monitor socket.
            if cast(zmq.Socket, self.heartbeatSocket).poll(timeout=0):  # non-blocking
                raise ConnectionError("connection to the server is closed or disconnected")
            # Otherwise keep waiting.


# Global Channel object reference
channel: Optional[Channel] = None


# Get ChannelType, defaulting to ZMQ
channelType = ChannelType(os.getenv("ARKOUDA_CHANNEL_TYPE", "ZMQ").upper())


def get_channel(
    server: str = "localhost",
    port: int = 5555,
    token: Optional[str] = None,
    connect_url: Optional[str] = None,
) -> Channel:
    """
    Return the configured `Channel` implementation.

    Parameters
    ----------
    server : str, default "localhost"
        Hostname visible to the current machine.
    port : int, default 5555
        Server port.
    token : str, optional
        Access token if authentication is enabled.
    connect_url : str, optional
        Complete URL in the form ``tcp://server:port?token=<token_value>``.

    Returns
    -------
    Channel
        The channel implementation as determined by `ARKOUDA_CHANNEL_TYPE`.

    Raises
    ------
    EnvironmentError
        If the environment variable references an invalid channel type.

    """
    if channelType == ChannelType.ZMQ:
        return ZmqChannel(server=server, port=port, user=username, token=token, connect_url=connect_url)
    else:
        raise EnvironmentError(f"Invalid channelType {channelType}")


def connect(
    server: str = "localhost",
    port: int = 5555,
    timeout: int = 0,
    access_token: Optional[str] = None,
    connect_url: Optional[str] = None,
    access_channel: Optional[Channel] = None,
) -> None:
    """
    Connect to a running Arkouda server.

    Parameters
    ----------
    server : str, default "localhost"
        Hostname visible to the current machine.
    port : int, default 5555
        Server port.
    timeout : int, default 0
        Timeout in seconds for send/receive. If positive, also activates
        heartbeat monitoring (ZMQ).
    access_token : str, optional
        Access token for authenticated servers.
    connect_url : str, optional
        Complete URL in the form ``tcp://server:port?token=<token_value>``.
    access_channel : Channel, optional
        A pre-constructed channel instance to use instead of the default.

    Raises
    ------
    ConnectionError
        If there is an error connecting to the server.
    ValueError
        If `connect_url` cannot be parsed.
    RuntimeError
        If a server-side error occurs during connect.

    Notes
    -----
    On success, prints the connected address (as seen by the server). If called
    with an existing connection, the socket will be re-initialized.

    """
    global channel
    global connected, serverConfig, regexMaxCaptures, registrationConfig

    cmd = "connect"
    logger.debug(f"[Python] Sending request: {cmd}")

    if access_channel:
        channel = access_channel
    else:
        channel = get_channel(server=server, port=port, token=access_token, connect_url=connect_url)

    channel.connect(timeout)

    return_message = channel.send_string_message(cmd=cmd)
    logger.debug(f"[Python] Received response: {str(return_message)}")
    connected = True

    serverConfig = _get_config_msg()
    if serverConfig["arkoudaVersion"] != __version__:
        warnings.warn(
            (
                "Version mismatch between client ({}) and server ({}); "
                + "this may cause some commands to fail or behave "
                + "incorrectly! Updating arkouda is strongly recommended."
            ).format(__version__, serverConfig["arkoudaVersion"]),
            RuntimeWarning,
        )
    regexMaxCaptures = serverConfig["regexMaxCaptures"]  # type: ignore
    registrationConfig = _get_registration_config_msg()  # type: ignore
    clientLogger.info(return_message)


def _parse_url(url: str) -> Tuple[str, int, Optional[str]]:
    """
    Parse a ``tcp://<host>:<port>`` or ``tcp://<host>:<port>?token=<token>`` URL.

    Parameters
    ----------
    url : str
        The URL string.

    Returns
    -------
    tuple[str, int, Optional[str]]
        Host, port, and optional token.

    Raises
    ------
    ValueError
        If the URL does not match an expected form, or the port is invalid.

    """
    try:
        no_protocol_stub = url.split("tcp://")
        if len(no_protocol_stub) < 2:
            raise ValueError(
                "url must be in form tcp://<hostname/url>:<port> or "
                "tcp://<hostname/url>:<port>?token=<token>"
            )

        host_stub = no_protocol_stub[1].split(":")
        if len(host_stub) < 2:
            raise ValueError(
                "url must be in form tcp://<hostname/url>:<port> or "
                "tcp://<hostname/url>:<port>?token=<token>"
            )
        host = host_stub[0]
        port_stub = host_stub[1]

        if "?token=" in port_stub:
            port_token_stub = port_stub.split("?token=")
            return (host, int(port_token_stub[0]), port_token_stub[1])
        else:
            return (host, int(port_stub), None)
    except Exception as e:
        raise ValueError(e)


def _start_tunnel(addr: str, tunnel_server: str) -> Tuple[str, object]:
    """
    Start an SSH tunnel and return the tunneled address and tunnel object.

    Parameters
    ----------
    addr : str
        ``host:port`` address to connect the tunnel to.
    tunnel_server : str
        SSH server spec.

    Returns
    -------
    tuple[str, object]
        (new_tunneled_addr, tunnel_object).

    Raises
    ------
    ConnectionError
        If the tunnel cannot be established.

    """
    from zmq import ssh

    kwargs = {"addr": addr, "server": tunnel_server}
    keyfile = os.getenv("ARKOUDA_KEY_FILE")
    password = os.getenv("ARKOUDA_PASSWORD")

    if keyfile:
        kwargs["keyfile"] = keyfile
    if password:
        kwargs["password"] = password

    try:
        return ssh.tunnel.open_tunnel(**kwargs)
    except Exception as e:
        raise ConnectionError(e)


def disconnect() -> None:
    """
    Disconnect the client from the Arkouda server.

    Raises
    ------
    ConnectionError
        If there is an error during disconnect.

    """
    global connected, serverConfig, regexMaxCaptures, registrationConfig

    if connected:
        message = "disconnect"
        logger.debug(f"[Python] Sending request: {message}")
        return_message = cast(str, cast(Channel, channel).send_string_message(message))
        logger.debug(f"[Python] Received response: {return_message}")
        try:
            cast(Channel, channel).disconnect()
        except Exception as e:
            raise ConnectionError(e)
        connected = False
        serverConfig = None
        regexMaxCaptures = -1
        registrationConfig = None
        clientLogger.info(return_message)
    else:
        clientLogger.info("not connected; cannot disconnect")


def shutdown() -> None:
    """
    Send a shutdown message and disconnect.

    Performs:
      1. Delete all objects in the server SymTable.
      2. Shut down the server.
      3. Disconnect the client.

    Raises
    ------
    RuntimeError
        Raised if the client is not connected to the Arkouda server or
        there is an error in disconnecting from the server.

    """
    global connected, serverConfig, regexMaxCaptures, registrationConfig

    if not connected:
        raise RuntimeError("not connected, cannot shutdown server")
    message = "shutdown"

    logger.debug(f"[Python] Sending request: {message}")
    return_message = cast(str, cast(Channel, channel).send_string_message(message))
    logger.debug(f"[Python] Received response: {return_message}")

    try:
        cast(Channel, channel).disconnect()
    except Exception as e:
        raise RuntimeError(e)
    connected = False
    serverConfig = None
    regexMaxCaptures = -1
    registrationConfig = None


def _json_args_to_str(json_obj: Optional[Dict] = None) -> Tuple[int, str]:
    """
    Convert a Python dictionary into a JSON-formatted string of parameters.

    Parameters
    ----------
    json_obj : dict, optional
        Mapping of command argument keys to values.

    Returns
    -------
    tuple[int, str]
        (number_of_parameters, json_string)

    Raises
    ------
    TypeError
        If keys are non-strings or values are not supported.

    Notes
    -----
    - Nested dictionaries are not yet supported.
    - Lists must be homogeneous.

    """
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
    """
    Send a command (string or binary) to the server and return its response.

    Parameters
    ----------
    cmd : str
        Server-side command name.
    args : dict, optional
        Mapping of argument keys to values.
    payload : memoryview, optional
        Binary payload for binary requests.
    send_binary : bool, default False
        If True, send as binary request.
    recv_binary : bool, default False
        If True, expect a binary reply (memoryview).

    Returns
    -------
    Union[str, memoryview]
        Server reply as string or memoryview.

    Raises
    ------
    KeyboardInterrupt
        If the user interrupts during command execution (socket is reset).
    RuntimeError
        If the client is not connected or the server reports an error.

    Notes
    -----
    If the server response is a string, the string corresponds to a success
    confirmation, warn message, or error message. A memoryview response
    corresponds to an Arkouda array output as a numpy array.

    """
    if not connected:
        raise RuntimeError("client is not connected to a server")

    size, msg_args = _json_args_to_str(args)

    try:
        if send_binary:
            assert payload is not None
            return cast(Channel, channel).send_binary_message(
                cmd=cmd, payload=payload, recv_binary=recv_binary, args=msg_args, size=size
            )
        else:
            assert payload is None
            return cast(Channel, channel).send_string_message(
                cmd=cmd, args=msg_args, size=size, recv_binary=recv_binary
            )
    except KeyboardInterrupt as e:
        # Reset the socket before re-raising to keep the REQ/REP stream in sync.
        cast(Channel, channel).connect(timeout=0)
        raise e


def wait_for_async_activity() -> None:
    """
    Wait for completion of asynchronous activities on the server.

    Intended for testing (e.g., automatic checkpointing). The server still
    considers itself "idle" while serving this message.
    """
    generic_msg("wait_for_async_activity")


def server_sleep(seconds) -> None:
    """
    Instruct the server to sleep for a given number of seconds (testing).

    Parameters
    ----------
    seconds : int or float
        Number of seconds for the server to sleep.

    """
    seconds = float(seconds)
    generic_msg("sleep", {"seconds": seconds})


def note_for_server_log(message) -> None:
    """
    Add an INFO entry to the server log (testing).

    Parameters
    ----------
    message : str
        Message to write to the server log.

    """
    generic_msg("note", {"message": str(message)})


def get_config() -> Mapping[str, Union[str, int, float]]:
    """
    Get runtime information about the server.

    Returns
    -------
    Mapping[str, Union[str, int, float]]
        Mapping containing selected server configuration and environment settings.
        Common keys include:

        - ``serverHostname`` : str
            Hostname of the Arkouda server.
        - ``serverPort`` : int
            Port number the server is listening on.
        - ``numLocales`` : int
            Number of Chapel locales (nodes) in the server deployment.
        - ``numPUs`` : int
            Number of processor units (hardware threads) per locale.
        - ``maxTaskPar`` : int
            Maximum number of tasks per locale.
        - ``physicalMemory`` : int
            Total physical memory on each locale (bytes).

    Raises
    ------
    RuntimeError
        If the client is not connected to a server.

    """
    if serverConfig is None:
        raise RuntimeError("client is not connected to a server, no 'serverConfig'")
    return serverConfig


def get_registration_config():
    """
    Get the registration settings the server was built with.

    Returns
    -------
    dict
        Snapshot of `registration-config.json` taken at server build time.

    Raises
    ------
    RuntimeError
        Raised if the client is not connected to a server

    """
    if registrationConfig is None:
        raise RuntimeError("client is not connected to a server, no 'registrationConfig'")
    return registrationConfig


def get_max_array_rank() -> int:
    """
    Get the maximum pdarray rank the server was compiled to support.

    Returns
    -------
    int
        Maximum pdarray rank.

    """
    if serverConfig is None:
        raise RuntimeError("client is not connected to a server, no 'serverConfig'")
    return max(get_array_ranks())


def get_array_ranks() -> list[int]:
    """
    Get the list of pdarray ranks the server was compiled to support.

    Returns
    -------
    list[int]
        Supported ranks.

    """
    if registrationConfig is None:
        raise RuntimeError("client is not connected to a server, no 'registrationConfig'")
    return registrationConfig["parameter_classes"]["array"]["nd"]


def _get_config_msg() -> Mapping[str, Union[str, int, float]]:
    """
    Fetch and parse the server configuration via ``getconfig``.

    Raises
    ------
    RuntimeError
        On server-side error.
    ValueError
        If the reply is not valid JSON.

    """
    try:
        raw_message = cast(str, generic_msg(cmd="getconfig"))
        return json.loads(raw_message)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"Returned config is not valid JSON: {raw_message}")
    except Exception as e:
        raise RuntimeError(f"{e} in retrieving Arkouda server config")


def _get_registration_config_msg() -> dict:
    """
    Fetch and parse the command registration configuration.

    Raises
    ------
    RuntimeError
        On server-side error.
    ValueError
        If the reply is not valid JSON.

    """
    try:
        raw_message = cast(str, generic_msg(cmd="getRegistrationConfig"))
        return json.loads(raw_message)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"Returned config is not valid JSON: {raw_message}")
    except Exception as e:
        raise RuntimeError(f"{e} in retrieving Arkouda server config")


def get_mem_used(unit: str = "b", as_percent: bool = False) -> int:
    """
    Compute the amount of memory used by objects in the server's symbol table.

    Parameters
    ----------
    unit : {'b','kb','mb','gb','tb','pb'}
        Unit of the return value (default 'b').
    as_percent : bool, default False
        If True, return the percentage of available memory that is used.

    Returns
    -------
    int
        Amount of memory used (scaled per `unit`) or percentage if `as_percent`.

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in getting memory used.
    ValueError
        Raised if the returned value is not an int-formatted string.

    """
    mem_used_message = cast(
        str,
        generic_msg(cmd="getmemused", args={"factor": _mem_get_factor(unit), "as_percent": as_percent}),
    )
    return int(mem_used_message)


def get_mem_avail(unit: str = "b", as_percent: bool = False) -> int:
    """
    Compute the amount of memory available to be used.

    Parameters
    ----------
    unit : {'b','kb','mb','gb','tb','pb'}
        Unit of the return value (default 'b').
    as_percent : bool, default False
        If True, return the percentage of memory that is available.

    Returns
    -------
    int
        Amount of memory available (scaled per `unit`) or percentage if `as_percent`.

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in getting memory used.
    ValueError
        Raised if the returned value is not an int-formatted string.

    """
    mem_avail_message = cast(
        str,
        generic_msg(cmd="getavailmem", args={"factor": _mem_get_factor(unit), "as_percent": as_percent}),
    )
    return int(mem_avail_message)


def get_mem_status() -> List[Mapping[str, Union[str, int, float]]]:
    """
    Retrieve the memory status for each locale.

    Returns
    -------
    list of mapping
        Each mapping contains:

        - ``total_mem`` : int
            Total physical memory on the locale host (bytes).
        - ``avail_mem`` : int
            Current available memory on the locale host (bytes).
        - ``arkouda_mem_alloc`` : int
            Memory allocated to the Arkouda Chapel process on the locale host (bytes).
        - ``pct_avail_mem`` : float
            Percentage of physical memory currently available on the locale host.
        - ``locale_id`` : int
            Locale identifier (between 0 and numLocales - 1).
        - ``locale_hostname`` : str
            Hostname of the locale.

    Raises
    ------
    RuntimeError
        If there is a server-side error in retrieving memory status.
    ValueError
        If the returned data is not valid JSON.

    """
    try:
        raw_message = cast(str, generic_msg(cmd="getmemstatus"))
        return json.loads(raw_message)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"Returned memory status is not valid JSON: {raw_message}")
    except Exception as e:
        raise RuntimeError(f"{e} in retrieving Arkouda server config")


def get_server_commands() -> Mapping[str, str]:
    """
    Return a dictionary of available server commands and their functions.

    Returns
    -------
    dict
        Mapping of command name → function (stringified).

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in retrieving and formatting the CommandMap
    ValueError
        Raised if there's an error in parsing the JSON-formatted server string

    """
    try:
        raw_message = cast(str, generic_msg(cmd="getCmdMap"))
        return json.loads(raw_message)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"Returned config is not valid JSON: {raw_message}")
    except Exception as e:
        raise RuntimeError(f"{e} in retrieving Arkouda server config")


def print_server_commands():
    """Print the list of available server commands."""
    cmdMap = get_server_commands()
    cmds = [k for k in sorted(cmdMap.keys())]
    sys.stdout.write(f"Total available server commands: {len(cmds)}")
    for cmd in cmds:
        sys.stdout.write(f"\t{cmd}")


def _no_op() -> str:
    """
    Send a no-op message to measure round-trip time.

    Returns
    -------
    str
        The server reply to ``noop``.

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in executing noop request

    """
    return cast(str, generic_msg(cmd="noop"))


def ruok() -> str:
    """
    Quick health check that does not require error handling.

    Returns
    -------
    str
        "imok" if the server is operating normally, otherwise an error string.

    """
    try:
        res = cast(str, generic_msg(cmd="ruok"))
        if res == "imok":
            return "imok"
        else:
            return f"imnotok because: {res}"
    except Exception as e:
        return f"ruok did not return response: {str(e)}"


def generate_history(
    num_commands: Optional[int] = None, command_filter: Optional[str] = None
) -> List[str]:
    """
    Generate a list of commands executed in the shell/notebook.

    Parameters
    ----------
    num_commands : int, optional
        Number of commands to retrieve from history.
    command_filter : str, optional
        Filter string to select a subset of commands.

    Returns
    -------
    list[str]
        List of command strings.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.get_config()    # doctest: +SKIP
    {'arkoudaVersion': 'v2025.01.13+165.g23ccdfd6c', 'chplVersion': '2.4.0',
        ... 'pythonVersion': '3.13', 'ZMQVersion': '4.3.5', 'HDF5Version': '1.14.4',
        ... 'serverHostname': 'pop-os', 'ServerPort': 5555, 'numLocales': 1,
        ... 'numPUs': 8, 'maxTaskPar': 8, 'physicalMemory': 67258408960,
        ... 'distributionType': 'domain(1,int(64),one)',
        ... 'LocaleConfigs': [{'id': 0, 'name': 'pop-os', 'numPUs': 8,
        ... 'maxTaskPar': 8, 'physicalMemory': 67258408960}], 'authenticate': False,
        ... 'logLevel': 'INFO', 'logChannel': 'FILE', 'regexMaxCaptures': 20,
        ... 'byteorder': 'little', 'autoShutdown': False, 'serverInfoNoSplash': False,
        ... 'maxArrayDims': 1, 'ARROW_VERSION': '19.0.0'}
    >>> ak.ones(10000, dtype=int)
    array([1 1 1 ... 1 1 1])
    >>> nums = ak.randint(0,500,10000, seed=1)
    >>> ak.argsort(nums)
    array([1984 2186 3574 ... 9298 9600 9651])
    >>> ak.generate_history(num_commands=5, command_filter='ak.')    # doctest: +SKIP
    ['ak.connect()', 'ak.get_config()', 'ak.ones(10000, dtype=int)', 'nums = ak.randint(0,500,10000)',
    'ak.argsort(nums)']

    """
    shell_mode = get_shell_mode()

    if shell_mode == ShellMode.REPL_SHELL:
        from arkouda.history import ShellHistoryRetriever

        return ShellHistoryRetriever().retrieve(command_filter, num_commands)
    else:
        from arkouda.history import NotebookHistoryRetriever

        return NotebookHistoryRetriever().retrieve(command_filter, num_commands)
