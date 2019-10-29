import zmq, json
import warnings

__all__ = ["verbose", "pdarrayIterThresh", "maxTransferBytes",
           "AllSymbols", "set_defaults", "connect", "disconnect",
           "shutdown", "get_config", "get_mem_used"]

# stuff for zmq connection
pspStr = None
context = zmq.Context()
socket = None
connected = False

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
def connect(server = "localhost", port = 5555):
    """
    Connect to a running arkouda server.

    Parameters
    ----------
    server : str, optional
        The hostname of the server (must be visible to the current 
        machine). Defaults to `localhost`.
    port : int, optional
        The port of the server. Defaults to 5555.

    Returns
    -------
    None

    Notes
    -----
    On success, prints the connected address, as seen by the server. If called
    with an existing connection, the socket will be re-initialized.
    """

    global context, socket, pspStr, connected, verbose

    if verbose: print("ZMQ version: {}".format(zmq.zmq_version()))

    # "protocol://server:port"
    pspStr = "tcp://{}:{}".format(server,port)
    if verbose: print("psp = ",pspStr);

    # setup connection to arkouda server
    socket = context.socket(zmq.REQ) # request end of the zmq connection
    socket.connect(pspStr)

    #send the connect message
    message = "connect"
    if verbose: print("[Python] Sending request: %s" % message)
    socket.send_string(message)

    # get the response that the server has started
    message = socket.recv_string()
    if verbose: print("[Python] Received response: %s" % message)

    print(message)
    connected = True

# message arkouda server to shutdown server
def disconnect():
    """
    Disconnect from the arkouda server.
    """

    global socket, pspStr, connected, verbose
    
    if socket is not None:
        # send disconnect message to server
        message = "disconnect"
        if verbose: print("[Python] Sending request: %s" % message)
        socket.send_string(message)
        message = socket.recv_string()
        if verbose: print("[Python] Received response: %s" % message)
        socket.disconnect(pspStr)
        print(message)
        connected = False
    else:
        print("not connected; cannot disconnect")
    
# message arkouda server to shutdown server
def shutdown():
    """
    Tell the arkouda server to delete all objects and shut itself down.
    """

    global socket, pspStr, connected, verbose
    
    # send shutdown message to server
    message = "shutdown"
    if verbose: print("[Python] Sending request: %s" % message)
    socket.send_string(message)
    message = socket.recv_string()
    if verbose: print("[Python] Received response: %s" % message)
    socket.disconnect(pspStr)
    print(message)
    connected = False
    
# send message to arkouda server and check for server side error
def generic_msg(message, send_bytes=False, recv_bytes=False):
    global socket, pspStr, connected, verbose

    if not connected:
        raise RuntimeError("Not connected to a server")
    
    if send_bytes:
        socket.send(message)
    else:
        if verbose: print("[Python] Sending request: %s" % message)
        socket.send_string(message)
    try:
        if recv_bytes:
            message = socket.recv()
            if message.startswith(b"Error:"): raise RuntimeError(message.decode())
            elif message.startswith(b"Warning:"): warnings.warn(message)
        else:
            message = socket.recv_string()
            if verbose: print("[Python] Received response: %s" % message)
            # raise errors sent back from the server
            if message.startswith("Error:"): raise RuntimeError(message)
            elif message.startswith("Warning:"): warnings.warn(message)
    except KeyboardInterrupt as e:
        # if the user interrupts during command execution, the socket gets out of sync
        # reset the socket before raising the interrupt exception
        socket = context.socket(zmq.REQ)
        socket.connect(pspStr)
        raise e
    return message

# query the server to get configuration 
def get_config():
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
def get_mem_used():
    """
    Compute the amount of memory used by objects in the server's symbol table.

    Returns
    -------
    int
        Amount of memory allocated to symbol table objects.
    """
    return int(generic_msg("getmemused"))
