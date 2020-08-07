import zmq, json, os, secrets
from typing import Mapping, Optional, Tuple, Union
import warnings, pkg_resources
from arkouda import security, io_util

__all__ = ["verbose", "pdarrayIterThresh", "maxTransferBytes",
           "AllSymbols", "set_defaults", "connect", "disconnect",
           "shutdown", "get_config", "get_mem_used", "__version__",
           "ruok"]

# Try to read the version from the file located at ../VERSION
VERSIONFILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION")
if os.path.isfile(VERSIONFILE):
    with open(VERSIONFILE, 'r') as f:
        __version__ = f.read().strip()
else:
    # Fall back to the version defined at build time in setup.py
    # pkg_resources is a subpackage of setuptools
    # __package__ is the name of the current package, i.e. "arkouda"
    __version__ = pkg_resources.require(__package__)[0].version

# stuff for zmq connection
pspStr = None
context = zmq.Context()
socket = None
connected = False
# username and token for when basic authentication is enabled
username = None
token = None
# verbose flag for arkouda module
verboseDefVal = False
verbose = verboseDefVal
# threshold for __iter__() to limit comms to arkouda_server
pdarrayIterThreshDefVal = 100
pdarrayIterThresh  = pdarrayIterThreshDefVal
maxTransferBytesDefVal = 2**30
maxTransferBytes = maxTransferBytesDefVal
AllSymbols = "__AllSymbols__"

# reset settings to default values
def set_defaults():
    """
    Sets global variables to defaults

    :return: None
    """
    global verbose, verboseDefVal, pdarrayIterThresh, pdarrayIterThreshDefVal
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
        an arkouda server where authentication is enabled. Defaults to None.
    connect_url : str, optional
        The complete url in the format of tcp://server:port?token=<token_value>
        where the token is optional
    Returns
    -------
    None

    Notes
    -----
    On success, prints the connected address, as seen by the server. If called
    with an existing connection, the socket will be re-initialized.
    """
    global context, socket, pspStr, connected, verbose, username, token

    if verbose: print("ZMQ version: {}".format(zmq.zmq_version()))

    if connect_url:
        url_values = _parse_url(connect_url)
        host = url_values[0]
        port = url_values[1]
        if len(url_values) == 3:
            access_token=url_values[2]
    
    # "protocol://server:port"
    pspStr = "tcp://{}:{}".format(server,port)
    
    # check to see if tunnelled connection is desired. If so, start tunnel
    tunnel_server = os.getenv('ARKOUDA_TUNNEL_SERVER')
    if tunnel_server:
        (pspStr, _) = _start_tunnel(addr=pspStr, tunnel_server=tunnel_server)
    
    if verbose: print("psp = ",pspStr);

    # create and configure socket for connections to arkouda server
    socket = context.socket(zmq.REQ) # request end of the zmq connection

    # if timeout is specified, set send and receive timeout params
    if timeout > 0:
        socket.setsockopt(zmq.SNDTIMEO, timeout*1000)
        socket.setsockopt(zmq.RCVTIMEO, timeout*1000)
    
    # set token and username global variables
    username = security.get_username()
    token = _set_access_token(username,access_token,pspStr)

    # connect to arkouda server
    try:
        socket.connect(pspStr)
    except Exception as e:
        raise ConnectionError(e)

    # send the connect message
    message = "connect"
    if verbose: print("[Python] Sending request: %s" % message)

    # send connect request to server and get the response confirming if
    # the connect request succeeded and, if not not, the error message
    message = _send_string_message(message)
    if verbose: print("[Python] Received response: %s" % message)
    connected = True

    conf = get_config()
    if conf['arkoudaVersion'] != __version__:
        warnings.warn(('Version mismatch between client ({}) and server ({}); ' +
                      'this may cause some commands to fail or behave ' +
                      'incorrectly! Updating arkouda is strongly recommended.').\
                      format(__version__, conf['arkoudaVersion']), RuntimeWarning)

def _parse_url(url : str) -> Tuple[str]:
    """
    Parses the url in the following format if authentication enabled:

    tcp://<hostname/url>:<port>?token=<token>

    If authentication is not enabled, the url is expected to be in the format:

    tcp://<hostname/url>:<port>

    :return: tuple containing host, port, and token, the latter of which
             can be None if authentication is not enabled
    :rtype: Tuple[str,int,Optional[str]]
    :raise: ValueError if the url does not match one of the above formats,
            if the port is not an integer, or if there's a general string 
            parse error raised
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
            port = int(port_token_stub[0])
            param_token = port_token_stub[1]
        else:
            port = int(port_stub)
            param_token = None
        return (host,port,param_token)
    except Exception as e:
        raise ValueError(e)

def _set_access_token(username : str, access_token : str, 
                                          connect_string : str) -> Optional[str]:
    """
    Sets the access_token for the connect request by doing the following:

    1. retrieves the token configured for the connect_string from the 
       .arkouda/tokens.txt file, if any
    2. if access_token is None, returns the retrieved token
    3. if access_token is not None, replaces retrieved token with the access_token
       to account for situations where the token can change for a url (for example,
       the arkouda_server is restarted and a corresponding new token is generated).

    :param str username: the username retrieved from the user's home directory
    :param str access_token: the access_token supplied by the user, defaults to None
    :param str connect_string: the arkouda_server host:port connect string
    :return: the retrieved or supplied access_token, defaults to None
    """
    path = '{}/tokens.txt'.format(security.get_arkouda_client_directory())
    tokens = io_util.delimited_file_to_dict(path)

    if access_token and access_token not in {'','None'}:
        saved_token = tokens.get(connect_string)
        if saved_token is None or saved_token != access_token:
            tokens[connect_string] = access_token
            io_util.dict_to_delimited_file(values=tokens,path=path,delimiter=',')
        return access_token   
    else:
        tokens = io_util.delimited_file_to_dict(path)
        return tokens.get(connect_string)

def _start_tunnel(addr : str, tunnel_server : str) -> str:
    """
    Starts ssh tunnel
    
    :param str tunnel_server: the ssh url
    :return: new tunneled-version of connect string
    :rtype: str
    :raise: ConnectionError if the ssh tunnel could not be created given the
            tunnel_server url and credentials (either password or key file)
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

def _send_string_message(message : str, recv_bytes=False) -> str:
    """
    Prepends the message string with Arkouda infrastructure elements 
    including username and authentication data and then sends the 
    resulting, composite string to the Arkouda server.

    :param str message: the message including arkouda command to be sent
    :param bool recv_bytes: indicates whether bytes vs str are expected
    :return: the response string sent back from the Arkouda server
    :rtype: str
    """
    message = '{}:{}:{}'.format(username,token,message)

    socket.send_string(message)

    if recv_bytes:
        return_message = socket.recv()
        # raise errors sent back from the server
        if return_message.startswith(b"Error:"): \
                                   raise RuntimeError(return_message.decode())
        elif return_message.startswith(b"Warning:"): warnings.warn(return_message)
    else:
        return_message = socket.recv_string()
        # raise errors sent back from the server
        if return_message.startswith("Error:"): raise RuntimeError(return_message)
        elif return_message.startswith("Warning:"): warnings.warn(return_message)
    return return_message

def _send_binary_message(message : str, recv_bytes=False) -> str:
    """
    Prepends the binary message with Arkouda infrastructure elements
    including username and authentication token and then sends the
    resulting, composite string to the Arkouda server.

    :param str message: the message including arkouda command to be sent
    :return: the response string sent back from the Arkouda server
    :rtype: str
    """
    socket.send('{}:{}:'.format(username,token,).encode() + message)
    if recv_bytes:
        return socket.recv()
    else:
        return socket.recv_string()

# message arkouda server the client is disconnecting from the server
def disconnect() -> None:
    """
    Disconnects the client from the arkouda server

    :return: None
    """
    global socket, pspStr, connected, verbose, token

    if socket is not None:
        # send disconnect message to server
        message = "disconnect"
        if verbose: print("[Python] Sending request: %s" % message)
        message = _send_string_message(message)
        if verbose: print("[Python] Received response: %s" % message)
        socket.disconnect(pspStr)
        connected = False
    else:
        print("not connected; cannot disconnect")

# message arkouda server to shutdown server
def shutdown() -> None:
    """
    Tell the arkouda server to delete all objects and shut itself down.

    :return: None
    """
    global socket, pspStr, connected, verbose

    if not connected:
        raise RuntimeError('not connected, cannot shutdown server')
    # send shutdown message to server
    message = "shutdown"
    if verbose: print("[Python] Sending request: %s" % message)
    return_message = _send_string_message(message)
    if verbose: print("[Python] Received response: {}".format(return_message))
    socket.disconnect(pspStr)
    connected = False

# send message to arkouda server and check for server side error
def generic_msg(message, send_bytes=False, recv_bytes=False) -> Union[str,bytes]:
    """
    Sends the binary or string message to the arkouda_server and returns 
    the response sent by the server which is either a success confirmation
    or error message
 
    :param Union[str,Array[byte] message : message to be sent
    :param bool send_bytes: indicates if binary message to be sent
    :param bool recv_bytes: indicates if binary message to be returned
    :return: the string or binary return message
    :rtype: Union[str,bytes]
    :raise: KeyboardInterrupt if user interrupts during commmand execution 
    """
    global socket, pspStr, connected, verbose

    if not connected:
        raise RuntimeError("client is not connected to a server")

    try:
        if send_bytes:
            return _send_binary_message(message=message, 
                                            recv_bytes=recv_bytes)
        else:
            if verbose: print("[Python] Sending request: %s" % message)
            return _send_string_message(message=message, 
                                            recv_bytes=recv_bytes)
    except KeyboardInterrupt as e:
        # if the user interrupts during command execution, the socket gets out 
        # of sync reset the socket before raising the interrupt exception
        socket = context.socket(zmq.REQ)
        socket.connect(pspStr)
        raise e

# query the server to get configuration
def get_config() -> Mapping[str, Union[str, int, float]]:
    """
    Get runtime information about the server.
    Returns
    -------
    dict
        serverHostname
        serverPort
        numLocales
        numPUs (number of processor units per locale)
        maxTaskPar (maximum number of tasks per locale)
        physicalMemory
    """
    return json.loads(generic_msg("getconfig"))

# query the server to get pda memory used
def get_mem_used() -> int:
    """
    Compute the amount of memory used by objects in the server's symbol table.

    :return: int indicating the amount of memory allocated to symbol table objects.
    :rtype: int
    """
    return int(generic_msg("getmemused"))

def _no_op() -> str:
    """
    Send a no-op message just to gather round trip time

    :return: noop command result
    :rtype: str
    """
    return generic_msg("noop")
  
def ruok() -> str:
    """
    Simply sends an "ruok" message to the server and, if the return message is "imok",
    this means the arkouda_server is up and operating normally. If the message is
    "imnotok" means an error occurred or the connection timed out.
    
    This method is basically a way to do a quick healthcheck in a way that does 
    not require error handling.
    
    :return: string indicating if the server is ok (operating normally), if there's
             an error server-side, or if ruok did not return a response
    :rtype: str
    """
    try:
        res = generic_msg('ruok')
        if res == 'imok':
            return 'imok'
        else:
            return 'imnotok because: {}'.format(res)
    except Exception as e:
        return 'ruok did not return response: {}'.format(str(e))
