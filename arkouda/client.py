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

Main API
--------
__all__ = [
    "connect",
    "disconnect",
    "shutdown",
    "get_config",
    "get_max_array_rank",
    "get_mem_used",
    "get_mem_avail",
    "get_mem_status",
    "get_server_commands",
    "print_server_commands",
    "generate_history",
    "ruok",
]

Classes
-------
Channel
    Abstract base class for communication between the client and the server.

ZmqChannel
    Default implementation of `Channel` using ZeroMQ request-reply pattern.

ClientMode, ShellMode, RequestMode, RequestStatus, ChannelType
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
- The `generic_msg()` function is used internally to handle message transmission.
- Clients must call `connect()` before performing any server operations.

Examples
--------
>>> import arkouda as ak
>>> ak.connect() # doctest: +SKIP
>>> ak.get_config() # doctest: +SKIP
{'serverHostname': 'localhost', 'numLocales': 4, ...}
>>> ak.disconnect() # doctest: +SKIP

See Also
--------
- arkouda.pdarray
- arkouda.dtypes
- arkouda.pandas.dataframe
- arkouda.security

"""

import json
import os
import warnings
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple, Union, cast

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
    "wait_for_async_activity",
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

    If in API mode, it is assumed the
    Arkouda client is being used via an API call instead of a Python shell or notebook.
    """

    UI = "UI"
    API = "API"

    def __str__(self) -> str:
        """Overridden method returns value."""
        return self.value

    def __repr__(self) -> str:
        """Overridden method returns value."""
        return self.value


class ShellMode(Enum):
    """Indicate whether the Python shell corresponds to a Jupyter notebook, or REPL or IPython shell."""

    IPYTHON_NOTEBOOK = "TerminalInteractiveShell"
    JUPYTER_NOTEBOOK = "ZMQInteractiveShell"
    REPL_SHELL = "REPL_SHELL"

    def __str__(self) -> str:
        """Overridden method returns value."""
        return self.value

    def __repr__(self) -> str:
        """Overridden method returns value."""
        return self.value


class RequestMode(Enum):
    """The Arkouda client-server communication pattern (synchronous or asynchronous)."""

    SYNCHRONOUS = "SYNCHRONOUS"
    ASYNCHRONOUS = "ASYNCHRONOUS"

    def __str__(self) -> str:
        """Overridden method returns value."""
        return self.value

    def __repr__(self) -> str:
        """Overridden method returns value."""
        return self.value


class RequestStatus(Enum):
    """The RequestStatus Enum indicates whether an asynchronous method invocation has completed."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"

    def __str__(self) -> str:
        """Overridden method returns value."""
        return self.value

    def __repr__(self) -> str:
        """Overridden method returns value."""
        return self.value


def get_shell_mode():
    """
    Determine the Python shell type and returns the corresponding ShellMode enum.

    Returns
    -------
    ShellMode
        The shell mode corresponding to a Python shell, Jupyter notebook,
        or IPython notebook

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

    print("{}".format(pyfiglet.figlet_format("Arkouda")))
    print(f"Client Version: {__version__}")  # type: ignore


# reset settings to default values
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
        """Overridden method returns value."""
        return self.value

    def __repr__(self) -> str:
        """Overridden method returns value."""
        return self.value


class Channel:
    """
    The Channel class defines methods for connecting to and communicating with the Arkouda server.

    Attributes
    ----------
    url : str
        Channel url used to connect to the Arkouda server which is either set
        to the connect_url or generated from supplied server and port values
    user : str
        Arkouda user who will use the Channel to connect to the arkouda_server
    token : Union[str, None]
        Token used to connect to the arkouda_server if authentication is enabled
    logger : ArkoudaLogger
        ArkoudaLogger used for logging

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
        Initiate chanel for connecting to and communicating with the Arkouda server.

        Parameters
        ----------
        user : str
            Arkouda user who will use the Channel to connect to the arkouda_server
        server : str, default='localhost'
            The hostname of the server (must be visible to the current machine).
        port : int, default=5555
            The port of the server.
        token : str, optional
            Token used to connect to the arkouda_server if authentication is enabled
        connect_url : str, optional
            The complete url in the format of tcp://server:port?token=<token_value>
            where the token is optional

        """
        self._set_url(server, port, connect_url)
        self.user = user
        self._set_access_token(server, port, token)
        self.logger = getArkoudaLogger(name="Arkouda Client")

    def _set_url(self, server: str, port: int, connect_url: Optional[str] = None) -> None:
        """
        Generate and sets the Channel url.

        If the connect_url is None, generates and sets the Channel url per the
        Channel protocol as well as host and port. Otherwise, sets the Channel url
        to the supplied connect_url value.

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
        Set the access token.

        Set the token for the Channel by doing the following:

        1. retrieves the token configured for the connect_string from the
           .arkouda/tokens.txt file, if any
        2. if token is None, returns the retrieved token
        3. if token is not None, replaces retrieved token with the token to account
           for situations where the token can change for a url (for example,
           the arkouda_server is restarted and a corresponding new token is generated).

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
            If there's an error writing host:port -> access_token mapping to
            the user's tokens.txt file or retrieving the user's tokens

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
    ) -> Union[str, memoryview]:
        """
        Generate and send a RequestMessage encapsulating to the Arkouda server.

        Generate a RequestMessage encapsulating command and requesting
        user information, sends it to the Arkouda server, and returns
        either a string or binary depending upon the message format.

        Parameters
        ----------
        cmd : str
            The name of the command to be executed by the Arkouda server
        recv_binary : bool, default=False
            Indicates if the return message will be a string or binary data
        args : str, default=None
            A delimited string containing 1..n command arguments
        size : int, default=-1
            Number of parameters contained in args. Only set if args is json.
        request_id : str, default=None
            Specifies an identifier for each request submitted to Arkouda

        Returns  # noqa: DAR202
        -------
        Union[str,memoryview]
            The response string or binary data sent back from the Arkouda server

        Raises
        ------
        RuntimeError
            Raised if the return message contains the word "Error", indicating
            a server-side error was thrown
        ValueError
            Raised if the return message is malformed JSON or is missing 1..n
            expected fields

        Notes
        -----
        s- Size is not yet utilized. It is being provided in preparation for further development.

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
    ) -> Union[str, memoryview]:
        """
        Generate and send a RequestMessage binary request to the Arkouda server.

        Generate a RequestMessage encapsulating command and requesting user information,
        information prepends the binary payload, sends the binary request to the Arkouda
        server, and returns either a string or binary depending upon the message format.

        Parameters
        ----------
        cmd : str
            The name of the command to be executed by the Arkouda server
        payload : memoryview
            The binary data to be converted to a pdarray, Strings, or Categorical
            object on the Arkouda server
        recv_binary : bool, default=False
            Indicates if the return message will be a string or binary data
        args : str, default=None
            A delimited string containing 1..n command arguments
        size : int, default=-1
            Number of parameters contained in args. Only set if args is json.
        request_id : str, default=None
            Specifies an identifier for each request submitted to Arkouda

        Returns  # noqa: DAR202
        -------
        Union[str,memoryview]
            The response string or binary data sent back from the Arkouda server

        Raises
        ------
        RuntimeError
            Raised if the return message contains the word "Error", indicating
            a server-side error was thrown
        ValueError
            Raised if the return message is malformed JSON or is missing 1..n
            expected fields

        """
        raise NotImplementedError("send_binary_message must be implemented in derived class")

    def connect(self, timeout: int = 0) -> None:
        """
        Establish a connection to the Arkouda server.

        Parameters
        ----------
        timeout : int, default=0
            Connection timeout

        Raises
        ------
        RuntimeError
            Raised if the return message contains the word "Error", indicating
            a server-side error was thrown

        """
        raise NotImplementedError("connect must be implemented in derived class")

    def disconnect(self) -> None:
        """
        Disconnects from the Arkouda server.

        Raises
        ------
        RuntimeError
            Raised if the return message contains the word "Error", indicating
            a server-side error was thrown

        """
        raise NotImplementedError("connect must be implemented in derived class")


class ZmqChannel(Channel):
    """
    Implement the Channel methods for ZMQ request/reply communication patterns.

    ZMQ the Arkouda-Chapel default.
    """

    __slots__ = "socket"

    def send_string_message(
        self,
        cmd: str,
        recv_binary: bool = False,
        args: Optional[str] = None,
        size: int = -1,
        request_id: Optional[str] = None,
    ) -> Union[str, memoryview]:
        message = RequestMessage(
            user=username, token=self.token, cmd=cmd, format=MessageFormat.STRING, args=args, size=size
        )
        # Note - Size is a placeholder here because Binary msg not yet support json args and
        # request_id is a noop for now
        logger.debug(f"sending message {json.dumps(message.asdict())}")

        self.socket.send_string(json.dumps(message.asdict()))

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
        # Note - Size is a placeholder here because Binary msg not yet support json args and
        # request_id is a noop for now
        message = RequestMessage(
            user=username, token=self.token, cmd=cmd, format=MessageFormat.BINARY, args=args, size=size
        )
        import zmq

        self.logger.debug(f"sending message {message}")

        self.socket.send(f"{json.dumps(message.asdict())}BINARY_PAYLOAD".encode(), flags=zmq.SNDMORE)
        self.socket.send(payload, copy=False)

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
        # create and configure socket for connections to arkouda server
        import zmq

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)

        logger.debug(f"ZMQ version: {zmq.zmq_version()}")

        # if timeout is specified, set send and receive timeout params
        if timeout > 0:
            self.socket.setsockopt(zmq.SNDTIMEO, timeout * 1000)
            self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)

        # connect to arkouda server
        try:
            self.socket.connect(self.url)
        except Exception as e:
            raise ConnectionError(e)

    def disconnect(self) -> None:
        try:
            self.socket.disconnect(self.url)
        except Exception as e:
            raise RuntimeError(e)


# Global Channel object reference
channel = None


# Get ChannelType, defaulting to ZMQ
channelType = ChannelType(os.getenv("ARKOUDA_CHANNEL_TYPE", "ZMQ").upper())


def get_channel(
    server: str = "localhost",
    port: int = 5555,
    token: Optional[str] = None,
    connect_url: Optional[str] = None,
) -> Channel:
    """
    Return the configured Channel implementation.

    Parameters
    ----------
    server : str, default="localhost"
        The hostname of the server (must be visible to the current
        machine).
    port : int, default=5555
        The port of the server.
    token : str, optional
        The token used to connect to an existing socket to enable access to
        an Arkouda server where authentication is enabled. Defaults to None.
    connect_url : str, optional
        The complete url in the format of tcp://server:port?token=<token_value>
        where the token is optional

    Returns
    -------
    Channel
        The Channel implementation configured with the ARKOUDA_CHANNEL_TYPE
        env variable

    Raises
    ------
    EnvironmentError
        Raised if the ARKOUDA_CHANNEL_TYPE references an invalid ChannelType

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
    Connect to a running arkouda server.

    Parameters
    ----------
    server : str, default="localhost"
        The hostname of the server (must be visible to the current
        machine).
    port : int, default=5555
        The port of the server.
    timeout : int, default=0
        The timeout in seconds for client send and receive operations.
        Defaults to 0 seconds, which is interpreted as no timeout.
    access_token : str, optional
        The token used to connect to an existing socket to enable access to
        an Arkouda server where authentication is enabled. Defaults to None.
    connect_url : str, optional
        The complete url in the format of tcp://server:port?token=<token_value>
        where the token is optional
    access_channel : Channel, optional
        The desired Channel implementation that differs from the default ZmqChannel

    Raises
    ------
    ConnectionError
        Raised if there's an error in connecting to the Arkouda server
    ValueError
        Raised if there's an error in parsing the connect_url parameter
    RuntimeError
        Raised if there is a server-side error

    Notes
    -----
    On success, prints the connected address, as seen by the server. If called
    with an existing connection, the socket will be re-initialized.

    """
    global channel
    global connected, serverConfig, regexMaxCaptures, registrationConfig

    # send the connect message
    cmd = "connect"
    logger.debug(f"[Python] Sending request: {cmd}")

    """
    If access-channel is not None, set global channel to access_channel. If not,
    set the global channel object via the get_channel factory method.
    """
    if access_channel:
        channel = access_channel
    else:
        channel = get_channel(server=server, port=port, token=access_token, connect_url=connect_url)

    # connect via the channel
    channel.connect(timeout)

    # send connect request to server and get the response confirming if
    # the connect request succeeded and, if not not, the error message
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
    Parse the url.

    Parse the url in the following format if authentication enabled:

    tcp://<hostname/url>:<port>?token=<token>

    If authentication is not enabled, the url is expected to be in the format:

    tcp://<hostname/url>:<port>

    Parameters
    ----------
    url : str
        The url string

    Returns
    -------
    Tuple[str,int,Optional[str]]
        A tuple containing the host, port, and token, the latter of which is None
        if authentication is not enabled for the Arkouda server being accessed

    Raises
    ------
    ValueError
        if the url does not match one of the above formats, if the port is not an
        integer, or if there's a general string parse error raised in the parsing
        of the url parameter

    """
    try:
        # split on tcp:// and if missing or malformmed, raise ValueError
        no_protocol_stub = url.split("tcp://")
        if len(no_protocol_stub) < 2:
            raise ValueError(
                "url must be in form tcp://<hostname/url>:<port> or "
                "tcp://<hostname/url>:<port>?token=<token>"
            )

        # split on : to separate host from port or port?token=<token>
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
    Start an ssh tunnel.

    Parameters
    ----------
    addr : str
        The address (host:port) of the Arkouda server to which the tunnel should connect
    tunnel_server : str
        The ssh server url

    Returns
    -------
    Tuple[str, object]
        str: The new tunneled-version of connect string
        object: The ssh tunnel object

    Raises
    ------
    ConnectionError
        If the ssh tunnel could not be created given the tunnel_server
        url and credentials (either password or key file)

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


# message arkouda server the client is disconnecting from the server
def disconnect() -> None:
    """
    Disconnects the client from the Arkouda server.

    Raises
    ------
    ConnectionError
        Raised if there's an error disconnecting from the Arkouda server

    """
    global connected, serverConfig, regexMaxCaptures, registrationConfig

    if connected:
        # send disconnect message to server
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
    Send a shutdown message.

    Send a shutdown message to the Arkouda server that does the
    following:

    1. Delete all objects in the SymTable
    2. Shuts down the Arkouda server
    3. Disconnects the client from the stopped Arkouda Server

    Raises
    ------
    RuntimeError
        Raised if the client is not connected to the Arkouda server or
        there is an error in disconnecting from the server

    """
    global connected, serverConfig, regexMaxCaptures, registrationConfig

    if not connected:
        raise RuntimeError("not connected, cannot shutdown server")
    # send shutdown message to server
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
    Convert Python Dictionary into a JSON formatted string.

    Convert Python Dictionary into a JSON formatted string that can be parsed by the msg
    processing system on the Arkouda Server.

    Parameters
    ----------
    json_obj : dict, optional
        Python dictionary of key:val representing command arguments

    Return
    ------
    Tuple - the number of parameters found and the json formatted string

    Raises
    ------
    TypeError
        - Keys are a type other than str
        - Any value is a dictionary.
        - A list contains values of multiple types.

    Notes
    -----
    - Nested dictionaries are not yet supported, but are planned for future support.
    - Support for lists of pdarray or Strings objects does not yet exist.

    """
    j: List[str] = []
    if json_obj is None:
        # early return when none
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
    Send a binary or string message to the arkouda_server, returning the response sent by the server.

    Send a binary or string message composed of a command and corresponding
    arguments to the arkouda_server, returning the response sent by the server.

    Parameters
    ----------
    cmd : str
        The server-side command to be executed
    args : dict, optional
        Python dictionary of key:val representing command arguments
    payload : memoryview, optional
        The payload when sending binary data
    send_binary : bool, default=False
        Indicates if the message to be sent is a string or binary
    recv_binary : bool, default=False
        Indicates if the return message will be a string or binary

    Returns
    -------
    Union[str, memoryview]
        The string or binary return message

    Raises
    ------
    KeyboardInterrupt
        Raised if the user interrupts during command execution
    RuntimeError
        Raised if the client is not connected to the server or if
        there is a server-side error thrown

    Notes
    -----
    If the server response is a string, the string corresponds to a success
    confirmation, warn message, or error message. A memoryview response
    corresponds to an Arkouda array output as a numpy array.

    """
    if not connected:
        raise RuntimeError("client is not connected to a server")

    size, msg_args = _json_args_to_str(args)
    from typing import cast as type_cast

    try:
        if send_binary:
            assert payload is not None
            return type_cast(Channel, channel).send_binary_message(
                cmd=cmd, payload=payload, recv_binary=recv_binary, args=msg_args, size=size
            )
        else:
            assert payload is None
            return type_cast(Channel, channel).send_string_message(
                cmd=cmd, args=msg_args, size=size, recv_binary=recv_binary
            )
    except KeyboardInterrupt as e:
        # if the user interrupts during command execution, the socket gets out
        # of sync reset the socket before raising the interrupt exception
        type_cast(Channel, channel).connect(timeout=0)
        raise e


def get_config() -> Mapping[str, Union[str, int, float]]:
    """
    Get runtime information about the server.

    Returns
    -------
    Mapping[str, Union[str, int, float]]
        serverHostname
        serverPort
        numLocales
        numPUs (number of processor units per locale)
        maxTaskPar (maximum number of tasks per locale)
        physicalMemory

    Raises
    ------
    RuntimeError
        Raised if the client is not connected to a server

    """
    if serverConfig is None:
        raise RuntimeError("client is not connected to a server, no 'serverConfig'")

    return serverConfig


def get_registration_config():
    """
    Get the registration settings that the server was built with.

    These registration settings are defined in the file `registration-config.json`
    and snapshot at server build time.

    Returns
    -------
    A mapping from parameter name to nested mappings, matching the json structure.

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

    This value corresponds to the maximum number in
    parameter_classes -> array -> nd in the `registration-config.json`
    file when the server was compiled.

    Returns
    -------
    int
        The maximum pdarray rank supported by the server

    """
    if serverConfig is None:
        raise RuntimeError("client is not connected to a server, no 'serverConfig'")

    return max(get_array_ranks())


def get_array_ranks() -> list[int]:
    """
    Get the list of pdarray ranks the server was compiled to support.

    This value corresponds to
    parameter_classes -> array -> nd in the `registration-config.json`
    file when the server was compiled.

    Returns
    -------
    list of int
        The pdarray ranks supported by the server

    """
    if registrationConfig is None:
        raise RuntimeError("client is not connected to a server, no 'registrationConfig'")

    return registrationConfig["parameter_classes"]["array"]["nd"]


def _get_config_msg() -> Mapping[str, Union[str, int, float]]:
    """
    Get runtime information about the server.

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in getting memory used
    ValueError
        Raised if there's an error in parsing the JSON-formatted server config

    """
    try:
        raw_message = cast(str, generic_msg(cmd="getconfig"))
        return json.loads(raw_message)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"Returned config is not valid JSON: {raw_message}")
    except Exception as e:
        raise RuntimeError(f"{e} in retrieving Arkouda server config")


def wait_for_async_activity() -> None:
    """
    Wait for the completion of asynchronous activities on the server.

    Intended to help with testing of automatic checkpointing.
    The server will consider itself "idle" despite serving this message.

    """
    generic_msg("wait_for_async_activity")


def _get_registration_config_msg() -> dict:
    """
    Get runtime information about the command registration configuration.

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in getting memory used
    ValueError
        Raised if there's an error in parsing the JSON-formatted server config

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
    unit : str {'b', 'kb', 'mb', 'gb', 'tb', 'pb'}
        unit of return ('b' by default)
    as_percent : bool
        If True, return the percent (as an int) of the available memory that's been used
        False by default

    Returns
    -------
    int
        Indicates the amount of memory allocated to symbol table objects.

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in getting memory used
    ValueError
        Raised if the returned value is not an int-formatted string

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
    unit : str {'b', 'kb', 'mb', 'gb', 'tb', 'pb'}
        unit of return ('b' by default)
    as_percent : bool
        If True, return the percent (as an int) of the memory that's available to be used
        False by default

    Returns
    -------
    int
        Indicates the amount of memory available to be used.

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in getting memory available
    ValueError
        Raised if the returned value is not an int-formatted string

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
    List[Mapping[str, Union[str, int, float]]]
        total_mem: total physical memory on locale host
        avail_mem: current available memory on locale host
        arkouda_mem_alloc: memory allocated to Arkouda chapel process on locale host
        pct_avail_mem: percentage of physical memory currently available on locale host
        locale_id: locale id which is between 0 and numLocales-1
        locale_hostname: host name of locale host

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in getting per-locale
        memory status information

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
    Return a dictionary of available server commands and the functions they map to.

    Returns
    -------
    dict
        String to String mapping of available server commands to functions

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
    """Print the list of the available Server commands."""
    cmdMap = get_server_commands()
    cmds = [k for k in sorted(cmdMap.keys())]
    print(f"Total available server commands: {len(cmds)}")
    for cmd in cmds:
        print(f"\t{cmd}")


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


def ruok() -> str:
    """
    Send a "ruok" message to the arkouda_server.

    Simply sends a "ruok" message to the server and, if the return message is
    "imok", this means the arkouda_server is up and operating normally. A return
    message of "imnotok" indicates an error occurred or the connection timed out.

    This method is basically a way to do a quick healthcheck in a way that does
    not require error handling.

    Returns
    -------
    str
        A string indicating if the server is operating normally (imok), if there's
        an error server-side, or if ruok did not return a response (imnotok) in
        both of the latter cases

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
    Generate list of commands executed.

    Generate list of commands executed within the Python shell, Jupyter notebook,
    or IPython notebook, with an optional cmd_filter and number of commands to return.

    Parameters
    ----------
    num_commands : int, optional
        The number of commands from history to retrieve
    command_filter : str, optional
        String containing characters used to select a subset of commands.

    Returns
    -------
    List[str]
        A list of commands from the Python shell, Jupyter notebook, or IPython notebook

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.connect()    # doctest: +SKIP
    connected to arkouda server tcp://*:5555
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
