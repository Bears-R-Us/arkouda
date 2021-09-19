import json, os
from typing import cast, Mapping, Optional, Tuple, Union
import warnings, pkg_resources
import zmq # type: ignore
import pyfiglet # type: ignore
from arkouda import security, io_util, __version__
from arkouda.logger import getArkoudaLogger
from arkouda.message import RequestMessage, MessageFormat, ReplyMessage, \
     MessageType

__all__ = ["connect", "disconnect", "shutdown", "get_config", "get_mem_used", "ruok"]

# stuff for zmq connection
pspStr = ''
context = zmq.Context()
socket = context.socket(zmq.REQ)
connected = False
serverConfig = None
# username and token for when basic authentication is enabled
username = ''
token = ''
# verbose flag for arkouda module
verboseDefVal = False
verbose = verboseDefVal
# threshold for __iter__() to limit comms to arkouda_server
pdarrayIterThreshDefVal = 100
pdarrayIterThresh  = pdarrayIterThreshDefVal
maxTransferBytesDefVal = 2**30
maxTransferBytes = maxTransferBytesDefVal

logger = getArkoudaLogger(name='Arkouda Client') 
clientLogger = getArkoudaLogger(name='Arkouda User Logger', logFormat='%(message)s')   

# Print splash message
print('{}'.format(pyfiglet.figlet_format('Arkouda')))
print('Client Version: {}'.format(__version__)) # type: ignore

# reset settings to default values
def set_defaults() -> None:
    """
    Sets client variables including verbose, maxTransferBytes and 
    pdarrayIterThresh to default values.
    
    Returns
    -------
    None
    """
    global verbose, maxTransferBytes, pdarrayIterThresh
    verbose = verboseDefVal
    pdarrayIterThresh  = pdarrayIterThreshDefVal
    maxTransferBytes = maxTransferBytesDefVal

# create context, request end of socket, and connect to it
def connect(server : str="localhost", port : int=5555, timeout : int=0, 
                           access_token : str=None, connect_url=None) -> None:
    """
    Connect to a running arkouda server.

    Parameters
    ----------
    server : str, optional
        The hostname of the server (must be visible to the current
        machine). Defaults to `localhost`.
    port : int, optional
        The port of the server. Defaults to 5555.
    timeout : int, optional
        The timeout in seconds for client send and receive operations.
        Defaults to 0 seconds, whicn is interpreted as no timeout.
    access_token : str, optional
        The token used to connect to an existing socket to enable access to 
        an Arkouda server where authentication is enabled. Defaults to None.
    connect_url : str, optional
        The complete url in the format of tcp://server:port?token=<token_value>
        where the token is optional

    Returns
    -------
    None
    
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
    global context, socket, pspStr, connected, serverConfig, verbose, username, token

    logger.debug("ZMQ version: {}".format(zmq.zmq_version()))

    if connect_url:
        url_values = _parse_url(connect_url)
        server = url_values[0]
        port = url_values[1]
        if len(url_values) == 3:
            access_token=url_values[2]
    
    # "protocol://server:port"
    pspStr = "tcp://{}:{}".format(server,port)
    
    # check to see if tunnelled connection is desired. If so, start tunnel
    tunnel_server = os.getenv('ARKOUDA_TUNNEL_SERVER')
    if tunnel_server:
        (pspStr, _) = _start_tunnel(addr=pspStr, tunnel_server=tunnel_server)
    
    logger.debug("psp = {}".format(pspStr))

    # create and configure socket for connections to arkouda server
    socket = context.socket(zmq.REQ) # request end of the zmq connection

    # if timeout is specified, set send and receive timeout params
    if timeout > 0:
        socket.setsockopt(zmq.SNDTIMEO, timeout*1000)
        socket.setsockopt(zmq.RCVTIMEO, timeout*1000)
    
    # set token and username global variables
    username = security.get_username()
    token = cast(str, _set_access_token(access_token=access_token, 
                                        connect_string=pspStr))

    # connect to arkouda server
    try:
        socket.connect(pspStr)
    except Exception as e:
        raise ConnectionError(e)

    # send the connect message
    cmd = "connect"
    logger.debug("[Python] Sending request: {}".format(cmd))

    # send connect request to server and get the response confirming if
    # the connect request succeeded and, if not not, the error message
    return_message = _send_string_message(cmd=cmd)
    logger.debug("[Python] Received response: {}".format(str(return_message)))
    connected = True

    serverConfig = _get_config_msg()
    if serverConfig['arkoudaVersion'] != __version__:
        warnings.warn(('Version mismatch between client ({}) and server ({}); ' +
                      'this may cause some commands to fail or behave ' +
                      'incorrectly! Updating arkouda is strongly recommended.').\
                      format(__version__, serverConfig['arkoudaVersion']), RuntimeWarning)
    clientLogger.info(return_message)

def _parse_url(url : str) -> Tuple[str,int,Optional[str]]:
    """
    Parses the url in the following format if authentication enabled:

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
        no_protocol_stub = url.split('tcp://')
        if len(no_protocol_stub) < 2:
            raise ValueError(('url must be in form tcp://<hostname/url>:<port>' +
                  ' or tcp://<hostname/url>:<port>?token=<token>'))

        # split on : to separate host from port or port?token=<token>
        host_stub = no_protocol_stub[1].split(':')
        if len(host_stub) < 2:
            raise ValueError(('url must be in form tcp://<hostname/url>:<port>' +
                  ' or tcp://<hostname/url>:<port>?token=<token>'))
        host = host_stub[0]
        port_stub = host_stub[1]
   
        if '?token=' in port_stub:
            port_token_stub = port_stub.split('?token=')
            return (host, int(port_token_stub[0]), port_token_stub[1])
        else:
            return (host, int(port_stub), None)
    except Exception as e:
        raise ValueError(e)

def _set_access_token(access_token : Optional[str], 
                                connect_string : str='localhost:5555') -> Optional[str]:
    """
    Sets the access_token for the connect request by doing the following:

    1. retrieves the token configured for the connect_string from the 
       .arkouda/tokens.txt file, if any
    2. if access_token is None, returns the retrieved token
    3. if access_token is not None, replaces retrieved token with the access_token
       to account for situations where the token can change for a url (for example,
       the arkouda_server is restarted and a corresponding new token is generated).

    Parameters
    ----------
    username : str
        The username retrieved from the user's home directory    
    access_token : str
        The access_token supplied by the user, which is required if authentication
        is enabled, defaults to None
    connect_string : str
        The arkouda_server host:port connect string, defaults to localhost:5555
    
    Returns
    -------
    str
        The access token configured for the host:port, None if there is no
        token configured for the host:port
    
    Raises
    ------
    IOError
        If there's an error writing host:port -> access_token mapping to
        the user's tokens.txt file or retrieving the user's tokens
    """
    path = '{}/tokens.txt'.format(security.get_arkouda_client_directory())
    try:
        tokens = io_util.delimited_file_to_dict(path)
    except Exception as e:
        raise IOError(e)

    if cast(str,access_token) and cast(str,access_token) not in {'','None'}:
        saved_token = tokens.get(connect_string)
        if saved_token is None or saved_token != access_token:
            tokens[connect_string] = cast(str,access_token)
            try:
                io_util.dict_to_delimited_file(values=tokens, path=path, 
                                               delimiter=',')
            except Exception as e:
                raise IOError(e)
        return access_token   
    else:
        try:
            tokens = io_util.delimited_file_to_dict(path)
        except Exception as e:
            raise IOError(e)
        return tokens.get(connect_string)

def _start_tunnel(addr : str, tunnel_server : str) -> Tuple[str,object]:
    """
    Starts ssh tunnel

    Parameters
    ----------
    tunnel_server : str
        The ssh server url   
    
    Returns
    -------
    str
        The new tunneled-version of connect string
    object
        The ssh tunnel object

    Raises
    ------
    ConnectionError 
        If the ssh tunnel could not be created given the tunnel_server 
        url and credentials (either password or key file)
    """
    from zmq import ssh
    kwargs = {'addr' : addr,
              'server' : tunnel_server}
    keyfile = os.getenv('ARKOUDA_KEY_FILE')
    password = os.getenv('ARKOUDA_PASSWORD')

    if keyfile:
        kwargs['keyfile'] = keyfile
    if password:
        kwargs['password'] = password

    try: 
        return ssh.tunnel.open_tunnel(**kwargs)
    except Exception as e:
        raise ConnectionError(e)

def _send_string_message(cmd : str, recv_binary : bool=False,
                         args : str=None) -> Union[str, memoryview]:
    """
    Generates a RequestMessage encapsulating command and requesting
    user information, sends it to the Arkouda server, and returns 
    either a string or binary depending upon the message format.

    Parameters
    ----------
    cmd : str
        The name of the command to be executed by the Arkouda server
    recv_binary : bool, defaults to False
        Indicates if the return message will be a string or binary data
    args : str
        A delimited string containing 1..n command arguments

    Returns
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
    message = RequestMessage(user=username, token=token, cmd=cmd, 
                          format=MessageFormat.STRING, args=args)

    logger.debug('sending message {}'.format(message))

    socket.send_string(json.dumps(message.asdict()))

    if recv_binary:
        frame = socket.recv(copy=False)
        view = frame.buffer
        # raise errors sent back from the server
        if bytes(view[0:len(b"Error:")]) == b"Error:":
            raise RuntimeError(frame.bytes.decode())
        return view
    else:
        raw_message = socket.recv_string()
        try:
            return_message = ReplyMessage.fromdict(json.loads(raw_message))

            # raise errors or warnings sent back from the server
            if return_message.msgType == MessageType.ERROR:
                raise RuntimeError(return_message.msg)
            elif return_message.msgType == MessageType.WARNING:
                warnings.warn(return_message.msg)
            return return_message.msg
        except KeyError as ke:
            raise ValueError('Return message is missing the {} field'.format(ke))
        except json.decoder.JSONDecodeError:
            raise ValueError('Return message is not valid JSON: {}'.\
                             format(raw_message))


def _send_binary_message(cmd : str, payload : memoryview, recv_binary : bool=False,
                         args : str=None) -> Union[str, memoryview]:
    """
    Generates a RequestMessage encapsulating command and requesting user information,
    information prepends the binary payload, sends the binary request to the Arkouda 
    server, and returns either a string or binary depending upon the message format.

    Parameters
    ----------
    cmd : str
        The name of the command to be executed by the Arkouda server    
    payload : memoryview
        The binary data to be converted to a pdarray, Strings, or Categorical
        object on the Arkouda server
    recv_binary : bool, defaults to False
        Indicates if the return message will be a string or binary data
    args : str
        A delimited string containing 1..n command arguments

    Returns
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
    message = RequestMessage(user=username, token=token, cmd=cmd, 
                                format=MessageFormat.BINARY, args=args)

    logger.debug('sending message {}'.format(message))

    socket.send('{}BINARY_PAYLOAD'.format(json.dumps(message.asdict())).encode(),
                flags=zmq.SNDMORE)
    socket.send(payload, copy=False)

    if recv_binary:
        frame = socket.recv(copy=False)
        view = frame.buffer
        # raise errors sent back from the server
        if bytes(view[0:len(b"Error:")]) == b"Error:":
            raise RuntimeError(frame.bytes.decode())
        return view
    else:
        raw_message = socket.recv_string()
        try:
            return_message = ReplyMessage.fromdict(json.loads(raw_message))

            # raise errors or warnings sent back from the server
            if return_message.msgType == MessageType.ERROR:
                raise RuntimeError(return_message.msg)
            elif return_message.msgType == MessageType.WARNING:
                warnings.warn(return_message.msg)
            return return_message.msg
        except KeyError as ke:
            raise ValueError('Return message is missing the {} field'.format(ke))
        except json.decoder.JSONDecodeError:
            raise ValueError('{} is not valid JSON, may be server-side error'.\
                             format(raw_message))

# message arkouda server the client is disconnecting from the server
def disconnect() -> None:
    """
    Disconnects the client from the Arkouda server

    Returns
    -------
    None
    
    Raises
    ------
    ConnectionError
        Raised if there's an error disconnecting from the Arkouda server
    """
    global socket, pspStr, connected, serverConfig, verbose, token

    if connected:
        # send disconnect message to server
        message = "disconnect"
        logger.debug("[Python] Sending request: {}".format(message))
        return_message = cast(str,_send_string_message(message))
        logger.debug("[Python] Received response: {}".format(return_message))
        try:
            socket.disconnect(pspStr)
        except Exception as e:
            raise ConnectionError(e)
        connected = False
        serverConfig = None
        clientLogger.info(return_message)
    else:
        clientLogger.info("not connected; cannot disconnect")
    

def shutdown() -> None:
    """
    Sends a shutdown message to the Arkouda server that does the
    following:
    
    1. Delete all objects in the SymTable
    2. Shuts down the Arkouda server
    3. Disconnects the client from the stopped Arkouda Server

    Returns
    -------
    None
    
    Raises
    ------
    RuntimeError
        Raised if the client is not connected to the Arkouda server or
        there is an error in disconnecting from the server
    """
    global socket, pspStr, connected, serverConfig, verbose

    if not connected:
        raise RuntimeError('not connected, cannot shutdown server')
    # send shutdown message to server
    message = "shutdown"

    logger.debug("[Python] Sending request: {}".format(message))
    return_message = cast(str,_send_string_message(message))
    logger.debug("[Python] Received response: {}".format(return_message))

    try:
        socket.disconnect(pspStr)
    except Exception as e:
        raise RuntimeError(e)
    connected = False
    serverConfig = None

def generic_msg(cmd : str, args : str=None, payload : memoryview=None, send_binary : bool=False,
                recv_binary : bool=False) -> Union[str, memoryview]:
    """
    Sends a binary or string message composed of a command and corresponding 
    arguments to the arkouda_server, returning the response sent by the server.

    Parameters
    ----------
    cmd : str
        The server-side command to be executed
    args : str
        A space-delimited list of command arguments
    payload : memoryview
        The payload when sending binary data
    send_binary : bool
        Indicates if the message to be sent is a string or binary
    recv_binary : bool
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
    global socket, pspStr, connected, verbose

    if not connected:
        raise RuntimeError("client is not connected to a server")
    
    try:
        if send_binary:
            assert payload is not None
            return _send_binary_message(cmd=cmd, payload=payload,
                                        recv_binary=recv_binary, args=args)
        else:
            assert payload is None
            return _send_string_message(cmd=cmd, args=args, recv_binary=recv_binary)

    except KeyboardInterrupt as e:
        # if the user interrupts during command execution, the socket gets out 
        # of sync reset the socket before raising the interrupt exception
        socket = context.socket(zmq.REQ)
        socket.connect(pspStr)
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
        raise RuntimeError("client is not connected to a server")

    return serverConfig

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
        raw_message = cast(str,generic_msg(cmd="getconfig"))
        return json.loads(raw_message)
    except json.decoder.JSONDecodeError:
        raise ValueError('Returned config is not valid JSON: {}'.format(raw_message))
    except Exception as e:
        raise RuntimeError('{} in retrieving Arkouda server config'.format(e))

def get_mem_used() -> int:
    """
    Compute the amount of memory used by objects in the server's symbol table.

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
    mem_used_message = cast(str,generic_msg(cmd="getmemused"))
    return int(mem_used_message)

def _no_op() -> str:
    """
    Send a no-op message just to gather round trip time

    Returns
    -------
    str
        The noop command result
        
    Raises
    ------  
    RuntimeError
        Raised if there is a server-side error in executing noop request
    """
    return cast(str,generic_msg(cmd="noop"))
  
def ruok() -> str:
    """
    Simply sends an "ruok" message to the server and, if the return message is
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
        res = cast(str,generic_msg(cmd='ruok'))
        if res == 'imok':
            return 'imok'
        else:
            return 'imnotok because: {}'.format(res)
    except Exception as e:
        return 'ruok did not return response: {}'.format(str(e))
