import zmq, json, os, secrets
from typing import Mapping, Tuple, Union
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
    global verbose, verboseDefVal, pdarrayIterThresh, pdarrayIterThreshDefVal
    verbose = verboseDefVal
    pdarrayIterThresh  = pdarrayIterThreshDefVal
    maxTransferBytes = maxTransferBytesDefVal

# create context, request end of socket, and connect to it
def connect(server = "localhost", port = 5555, timeout = 0, 
                                                  access_token=None) -> None:
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
        Defaults to 0 seconds, which is interpreted as no timeout
    access_token : str, optional 
        The token used to connect to an existing socket to enable access to
        session-scoped resources. Defaults to None

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

    # "protocol://server:port"
    pspStr = "tcp://{}:{}".format(server,port)
    
    # check to see if tunnelled connection is desired. If so, start tunnel
    tunnel_server = os.getenv('ARKOUDA_TUNNEL_SERVER')
    if tunnel_server:
        _start_tunnel(addr=pspStr, tunnel_server=tunnel_server)

    if verbose: print("psp = ",pspStr);

    # create and configure socket for connections to arkouda server
    socket = context.socket(zmq.REQ) # request end of the zmq connection

    # if timeout is specified, set send and receive timeout params
    if timeout > 0:
        socket.setsockopt(zmq.SNDTIMEO, timeout*1000)
        socket.setsockopt(zmq.RCVTIMEO, timeout*1000)
    
    # set token and username global variables
    token = access_token
    username = security.get_username()

    # connect to arkouda server
    try:
        socket.connect(pspStr)
    except Exception as e:
        raise ConnectionError(e)

    # send the connect message
    message = "connect"
    if verbose: print("[Python] Sending request: %s" % message)

    # send connect request to server and get the response confirming i
    # if the connect request succeeded and, if not not, the error message
    message = _send_string_message(message)
    if verbose: print("[Python] Received response: %s" % message)
    connected = True

    conf = get_config()
    if conf['arkoudaVersion'] != __version__:
        warnings.warn(('Version mismatch between client ({}) and server ({}); ' +
                      'this may cause some commands to fail or behave ' +
                      'incorrectly! Updating arkouda is strongly recommended.').\
                      format(__version__, conf['arkoudaVersion']), RuntimeWarning)

def _set_access_token(username : str, access_token : str, connect_string : str) -> None:

    tokens = io_util.csv_file_to_dict('{}/.arkouda/tokens.txt'.\
                                                format(security.get_home()))



def _start_tunnel(addr : str, tunnel_server : str) -> None:
    """
    Starts ssh tunnel
    
    :param str tunnel_server: the ssh url
    :return: None
    :raise: ConnectionError if the ssh tunnel could not be created given the tunnel_server 
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
        ssh.tunnel.open_tunnel(**kwargs)
    except Exception as e:
        raise ConnectionError(e)

def _get_arkouda_directory() -> str:

    io_util.get_directory('{}/.arkouda'.format(security.get_home_directory()))

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
        if return_message.startswith(b"Error:"): raise RuntimeError(return_message.decode())
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
    including username and authentication data and then sends the
    resulting, composite string to the Arkouda server.

    :param str message: the message including arkouda command to be sent
    :return: the response string sent back from the Arkouda server
    :rtype: str
    """
    #print('SENDING BINARY MESSAGE {}'.format(b'{}:{}:BINARY_PAYLOAD{}'.format(username,token,message))
    socket.send('{}:{}:'.format(username,token,).encode() + message)
    #socket.send(message)
    if recv_bytes:
        return socket.recv()
    else:
        return socket.recv_string()

# message arkouda server the client is disconnecting from the server
def disconnect() -> None:
    """
    Disconnect from the arkouda server.
    """
    global socket, pspStr, connected, verbose, token

    if socket is not None:
        if token:
            print('the token is {}'.format(token))
        # send disconnect message to server
        message = "disconnect"
        if verbose: print("[Python] Sending request: %s" % message)
        message = _send_string_message(message)
        if verbose: print("[Python] Received response: %s" % message)
        socket.disconnect(pspStr)
        print(message)
        connected = False
    else:
        print("not connected; cannot disconnect")

# message arkouda server to shutdown server
def shutdown() -> None:
    """
    Tell the arkouda server to delete all objects and shut itself down.
    """

    global socket, pspStr, connected, verbose

    # send shutdown message to server
    message = "shutdown"
    if verbose: print("[Python] Sending request: %s" % message)
    return_message = _send_string_message(message)
    if verbose: print("[Python] Received response: {}".format(return_message))
    socket.disconnect(pspStr)
    print(return_message)
    connected = False

# send message to arkouda server and check for server side error
def generic_msg(message, send_bytes=False, recv_bytes=False) -> str:
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
def get_config() -> Mapping[str, Union[str, int]]:
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

    Returns
    -------
    int
        Amount of memory allocated to symbol table objects.
    """
    return int(generic_msg("getmemused"))

def _no_op() -> str:
    """
    Send a no-op message just to gather round trip time
    """
    return generic_msg("noop")
  
def ruok() -> str:
    """
    Simply sends an "ruok" message to the server and, if the return message is "imok",
    this means the arkouda_server is up and operating normally. If the message is
    "imnotok" means an error occurred or the connection timed out.
    
    This method is basically a way to do a quick healthcheck in a way that does 
    not require error handling.
    
    :return: string indicating if the server is ok (operating normally), if there's an
             error server-side, or if ruok did not return a response
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
