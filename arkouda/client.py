import zmq, json
import warnings

__all__ = ["connected", "verbose", "pdarrayIterThresh", "maxTransferBytes",
           "AllSymbols", "set_defaults", "connect", "disconnect",
           "shutdown", "get_config", "get_mem_used"]

# stuff for zmq connection
pspStr = None
context = None
socket = None
serverPid = None
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
        On success, prints ``connected to tcp://<hostname>:<port>``
    """
    global verbose, context, socket, pspStr, serverPid, connected

    if connected == False:
        print(zmq.zmq_version())
        
        # "protocol://server:port"
        pspStr = "tcp://{}:{}".format(server,port)
        print("psp = ",pspStr);
    
        # setup connection to arkouda server
        context = zmq.Context()
        socket = context.socket(zmq.REQ) # request end of the zmq connection
        socket.connect(pspStr)
        connected = True
        
        #send the connect message
        message = "connect"
        if verbose: print("[Python] Sending request: %s" % message)
        socket.send_string(message)
        
        # get the response that the server has started
        message = socket.recv_string()
        if verbose: print("[Python] Received response: %s" % message)

        print("connected to {}".format(pspStr))

# message arkouda server to shutdown server
def disconnect():
    global verbose, context, socket, pspStr, connected

    if connected == True:
        # send disconnect message to server
        message = "disconnect"
        if verbose: print("[Python] Sending request: %s" % message)
        socket.send_string(message)
        message = socket.recv_string()
        if verbose: print("[Python] Received response: %s" % message)
        socket.disconnect(pspStr)
        connected = False

        print("disconnected from {}".format(pspStr))
    
# message arkouda server to shutdown server
def shutdown():
    global verbose, context, socket, pspStr, connected
    
    # send shutdown message to server
    message = "shutdown"
    if verbose: print("[Python] Sending request: %s" % message)
    socket.send_string(message)
    message = socket.recv_string()
    if verbose: print("[Python] Received response: %s" % message)
    connected = False
    socket.disconnect(pspStr)

# send message to arkouda server and check for server side error
def generic_msg(message, send_bytes=False, recv_bytes=False):
    global verbose, context, socket
    if send_bytes:
        socket.send(message)
    else:
        if verbose: print("[Python] Sending request: %s" % message)
        socket.send_string(message)
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
