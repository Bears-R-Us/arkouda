#!/usr/bin/env python3

# arkouda python module -- python wrapper and comms
# arkouda is numpy like high perf dist arrays

# import zero mq
import zmq
import os
import subprocess
import warnings
import json, struct
import numpy as np

# stuff for zmq connection
pspStr = None
context = None
socket = None
serverPid = None
connected = False

# verbose flag for arkouda module
vDefVal = False
v = vDefVal
# threshold for __iter__() to limit comms to arkouda_server
pdarrayIterThreshDefVal = 100
pdarrayIterThresh  = pdarrayIterThreshDefVal
maxTransferBytesDefVal = 2**30
maxTransferBytes = maxTransferBytesDefVal

# reset settings to default values
def set_defaults():
    global v, vDefVal, pdarrayIterThresh, pdarrayIterThreshDefVal 
    v = vDefVal
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
    global v, context, socket, pspStr, serverPid, connected

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
        if v: print("[Python] Sending request: %s" % message)
        socket.send_string(message)
        
        # get the response that the server has started
        message = socket.recv_string()
        if v: print("[Python] Received response: %s" % message)

        print("connected to {}".format(pspStr))

# message arkouda server to shutdown server
def disconnect():
    global v, context, socket, pspStr, connected

    if connected == True:
        # send disconnect message to server
        message = "disconnect"
        if v: print("[Python] Sending request: %s" % message)
        socket.send_string(message)
        message = socket.recv_string()
        if v: print("[Python] Received response: %s" % message)
        socket.disconnect(pspStr)
        connected = False

        print("disconnected from {}".format(pspStr))
    
# message arkouda server to shutdown server
def shutdown():
    global v, context, socket, pspStr, connected
    
    # send shutdown message to server
    message = "shutdown"
    if v: print("[Python] Sending request: %s" % message)
    socket.send_string(message)
    message = socket.recv_string()
    if v: print("[Python] Received response: %s" % message)
    connected = False
    socket.disconnect(pspStr)

# send message to arkouda server and check for server side error
def generic_msg(message, send_bytes=False, recv_bytes=False):
    global v, context, socket
    if send_bytes:
        socket.send(message)
    else:
        if v: print("[Python] Sending request: %s" % message)
        socket.send_string(message)
    if recv_bytes:
        message = socket.recv()
        if message.startswith(b"Error:"): raise RuntimeError(message.decode())
        elif message.startswith(b"Warning:"): warnings.warn(message)
    else:
        message = socket.recv_string()
        if v: print("[Python] Received response: %s" % message)
        # raise errors sent back from the server
        if message.startswith("Error:"): raise RuntimeError(message)
        elif message.startswith("Warning:"): warnings.warn(message)
    return message

# supported dtypes
structDtypeCodes = {'int64': 'q',
                    'float64': 'd',
                    'bool': '?'}
DTypes = frozenset(structDtypeCodes.keys())
NUMBER_FORMAT_STRINGS = {'bool': '{}',
                         'int64': '{:n}',
                         'float64': '{:.17f}'}
bool = np.bool
int64 = np.int64
float64 = np.float64

def check_np_dtype(dt):
    """
    Assert that numpy dtype dt is one of the dtypes supported by arkouda, 
    otherwise raise TypeError.
    """
    if dt.name not in DTypes:
        raise TypeError("Unsupported type: {}".format(dt))

def translate_np_dtype(dt):
    """
    Split numpy dtype dt into its kind and byte size, raising TypeError 
    for unsupported dtypes.
    """
    # Assert that dt is one of the arkouda supported dtypes
    check_np_dtype(dt)
    trans = {'i': 'int', 'f': 'float', 'b': 'bool'}
    kind = trans[dt.kind]
    return kind, dt.itemsize

def resolve_scalar_dtype(val):
    """
    Try to infer what dtype arkouda_server should treat val as.
    """
    # Python bool or np.bool
    if isinstance(val, bool) or (hasattr(val, 'dtype') and val.dtype.kind == 'b'):
        return 'bool'
    # Python int or np.int* or np.uint*
    elif isinstance(val, int) or (hasattr(val, 'dtype') and val.dtype.kind in 'ui'):
        return 'int64'
    # Python float or np.float*
    elif isinstance(val, float) or (hasattr(val, 'dtype') and val.dtype.kind == 'f'):
        return 'float64'
    # Other numpy dtype
    elif hasattr(val, 'dtype'):
        return dtype.name
    # Other python type
    else:
        return str(type(val))

BinOps = frozenset(["+", "-", "*", "/", "//", "%", "<", ">", "<=", ">=", "!=", "==", "&", "|", "^", "<<", ">>","**"])
OpEqOps = frozenset(["+=", "-=", "*=", "/=", "//=", "&=", "|=", "^=", "<<=", ">>=","**="])

# class for the pdarray
class pdarray:
    """
    The basic arkouda array class. This class contains only the 
    attributies of the array; the data resides on the arkouda 
    server. When a server operation results in a new array, arkouda 
    will create a pdarray instance that points to the array data on 
    the server. As such, the user should not initialize pdarray 
    instances directly.

    Attributes
    ----------
    name : str
        The server-side identifier for the array
    dtype : np.dtype
        The element type of the array
    size : int
        The number of elements in the array
    ndim : int
        The rank of the array (currently only rank 1 arrays supported)
    shape : tuple
        The sizes of each dimension of the array
    itemsize : int
        The size in bytes of each element
    """
    def __init__(self, name, dtype, size, ndim, shape, itemsize):
        self.name = name
        self.dtype = np.dtype(dtype)
        self.size = size
        self.ndim = ndim
        self.shape = shape
        self.itemsize = itemsize

    def __del__(self):
        global connected
        if connected:
            generic_msg("delete {}".format(self.name))

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        global pdarrayIterThresh
        return generic_msg("str {} {}".format(self.name,pdarrayIterThresh) )

    def __repr__(self):
        global pdarrayIterTresh
        return generic_msg("repr {} {}".format(self.name,pdarrayIterThresh))
 
    def format_other(self, other):
        """
        Attempt to cast scalar other to the element dtype of this pdarray, 
        and print the resulting value to a string (e.g. for sending to a
        server command). The user should not call this function directly.
        """
        try:
            other = self.dtype.type(other)
        except:
            raise TypeError("Unable to convert {} to {}".format(other, self.dtype.name))
        if self.dtype == np.bool:
            return str(other)
        fmt = NUMBER_FORMAT_STRINGS[self.dtype.name]
        return fmt.format(other)
        
    # binary operators
    def binop(self, other, op):
        if op not in BinOps:
            raise ValueError("bad operator {}".format(op))
        # pdarray binop pdarray
        if isinstance(other, pdarray):
            if self.size != other.size:
                raise ValueError("size mismatch {} {}".format(self.size,other.size))
            msg = "binopvv {} {} {}".format(op, self.name, other.name)
            repMsg = generic_msg(msg)
            return create_pdarray(repMsg)
        # pdarray binop array-like is not implemented
        if hasattr(other, '__len__'): 
            return NotImplemented
        # pdarray binop scalar
        dt = resolve_scalar_dtype(other)
        if dt not in DTypes:
            raise TypeError("Unhandled scalar type: {} ({})".format(other, dt))
        msg = "binopvs {} {} {} {}".format(op, self.name, dt, NUMBER_FORMAT_STRINGS[dt].format(other))
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    # reverse binary operators
    # pdarray binop pdarray: taken care of by binop function
    def r_binop(self, other, op):
        if op not in BinOps:
            raise ValueError("bad operator {}".format(op))
        # pdarray binop array-like is not implemented
        if hasattr(other, '__len__'): 
            return NotImplemented
        # pdarray binop scalar
        dt = resolve_scalar_dtype(other)
        if dt not in DTypes:
            raise TypeError("Unhandled scalar type: {} ({})".format(other, dt))
        msg = "binopsv {} {} {} {}".format(op, dt, NUMBER_FORMAT_STRINGS[dt].format(other), self.name)
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    # overload + for pdarray, other can be {pdarray, int, float}
    def __add__(self, other):
        return self.binop(other, "+")

    def __radd__(self, other):
        return self.r_binop(other, "+")

    # overload - for pdarray, other can be {pdarray, int, float}
    def __sub__(self, other):
        return self.binop(other, "-")

    def __rsub__(self, other):
        return self.r_binop(other, "-")

    # overload * for pdarray, other can be {pdarray, int, float}
    def __mul__(self, other):
        return self.binop(other, "*")

    def __rmul__(self, other):
        return self.r_binop(other, "*")

    # overload / for pdarray, other can be {pdarray, int, float}
    def __truediv__(self, other):
        return self.binop(other, "/")

    def __rtruediv__(self, other):
        return self.r_binop(other, "/")

    # overload // for pdarray, other can be {pdarray, int, float}
    def __floordiv__(self, other):
        return self.binop(other, "//")

    def __rfloordiv__(self, other):
        return self.r_binop(other, "//")

    def __mod__(self, other):
        return self.binop(other, "%")

    def __rmod__(self, other):
        return self.r_binop(other, "%")

    # overload << for pdarray, other can be {pdarray, int}
    def __lshift__(self, other):
        return self.binop(other, "<<")

    def __rlshift__(self, other):
        return self.r_binop(other, "<<")

    # overload >> for pdarray, other can be {pdarray, int}
    def __rshift__(self, other):
        return self.binop(other, ">>")

    def __rrshift__(self, other):
        return self.r_binop(other, ">>")

    # overload & for pdarray, other can be {pdarray, int}
    def __and__(self, other):
        return self.binop(other, "&")

    def __rand__(self, other):
        return self.r_binop(other, "&")

    # overload | for pdarray, other can be {pdarray, int}
    def __or__(self, other):
        return self.binop(other, "|")

    def __ror__(self, other):
        return self.r_binop(other, "|")

    # overload | for pdarray, other can be {pdarray, int}
    def __xor__(self, other):
        return self.binop(other, "^")

    def __rxor__(self, other):
        return self.r_binop(other, "^")

    def __pow__(self,other): 
        return self.binop(other,"**")
    
    def __rpow__(self,other): 
        return self.r_binop(other,"**")

    # overloaded comparison operators
    def __lt__(self, other):
        return self.binop(other, "<")

    def __gt__(self, other):
        return self.binop(other, ">")

    def __le__(self, other):
        return self.binop(other, "<=")

    def __ge__(self, other):
        return self.binop(other, ">=")

    def __eq__(self, other):
        return self.binop(other, "==")

    def __ne__(self, other):
        return self.binop(other, "!=")

    # overload unary- for pdarray implemented as pdarray*(-1)
    def __neg__(self):
        return self.binop(-1, "*")

    # overload unary~ for pdarray implemented as pdarray^(~0)
    def __invert__(self):
        if self.dtype == np.int64:
            return self.binop(~0, "^")
        if self.dtype == np.bool:
            return self.binop(True, "^")
        return NotImplemented

    # op= operators
    def opeq(self, other, op):
        if op not in OpEqOps:
            raise ValueError("bad operator {}".format(op))
        # pdarray op= pdarray
        if isinstance(other, pdarray):
            if self.size != other.size:
                raise ValueError("size mismatch {} {}".format(self.size,other.size))
            generic_msg("opeqvv {} {} {}".format(op, self.name, other.name))
            return self
        # pdarray op= array-like is not implemented
        if hasattr(other, '__len__'): 
            return NotImplemented
        # pdarray binop scalar
        # opeq requires scalar to be cast as pdarray dtype
        try:
            other = self.dtype.type(other)
        except: # Can't cast other as dtype of pdarray
            return NotImplemented
        msg = "opeqvs {} {} {} {}".format(op, self.name, self.dtype.name, self.format_other(other))
        generic_msg(msg)
        return self

    # overload += pdarray, other can be {pdarray, int, float}
    def __iadd__(self, other):
        return self.opeq(other, "+=")

    # overload -= pdarray, other can be {pdarray, int, float}
    def __isub__(self, other):
        return self.opeq(other, "-=")

    # overload *= pdarray, other can be {pdarray, int, float}
    def __imul__(self, other):
        return self.opeq(other, "*=")

    # overload /= pdarray, other can be {pdarray, int, float}
    def __itruediv__(self, other):
        return self.opeq(other, "/=")

    # overload //= pdarray, other can be {pdarray, int, float}
    def __ifloordiv__(self, other):
        return self.opeq(other, "//=")

    # overload <<= pdarray, other can be {pdarray, int, float}
    def __ilshift__(self, other):
        return self.opeq(other, "<<=")

    # overload >>= pdarray, other can be {pdarray, int, float}
    def __irshift__(self, other):
        return self.opeq(other, ">>=")

    # overload &= pdarray, other can be {pdarray, int, float}
    def __iand__(self, other):
        return self.opeq(other, "&=")

    # overload |= pdarray, other can be {pdarray, int, float}
    def __ior__(self, other):
        return self.opeq(other, "|=")

    # overload ^= pdarray, other can be {pdarray, int, float}
    def __ixor__(self, other):
        return self.opeq(other, "^=")
    def __ipow__(self, other):
        return self.opeq(other,"**=")

    # overload a[] to treat like list
    def __getitem__(self, key):
        if np.isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            if (key >= 0 and key < self.size):
                repMsg = generic_msg("[int] {} {}".format(self.name, key))
                fields = repMsg.split()
                # value = fields[2]
                return parse_single_value(' '.join(fields[1:]))
            else:
                raise IndexError("[int] {} is out of bounds with size {}".format(key,self.size))
        if isinstance(key, slice):
            (start,stop,stride) = key.indices(self.size)
            if v: print(start,stop,stride)
            repMsg = generic_msg("[slice] {} {} {} {}".format(self.name, start, stop, stride))
            return create_pdarray(repMsg);
        if isinstance(key, pdarray):
            kind, itemsize = translate_np_dtype(key.dtype)
            if kind not in ("bool", "int"):
                raise TypeError("unsupported pdarray index type {}".format(key.dtype))
            if kind == "bool" and self.size != key.size:
                raise ValueError("size mismatch {} {}".format(self.size,key.size))
            repMsg = generic_msg("[pdarray] {} {}".format(self.name, key.name))
            return create_pdarray(repMsg);
        else:
            return NotImplemented

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if (key >= 0 and key < self.size):
                generic_msg("[int]=val {} {} {} {}".format(self.name, key, self.dtype.name, self.format_other(value)))
            else:
                raise IndexError("index {} is out of bounds with size {}".format(key,self.size))
        elif isinstance(key, pdarray):
            if isinstance(value, pdarray):
                generic_msg("[pdarray]=pdarray {} {} {}".format(self.name,key.name,value.name))
            else:
                generic_msg("[pdarray]=val {} {} {} {}".format(self.name, key.name, self.dtype.name, self.format_other(value)))
        elif isinstance(key, slice):
            (start,stop,stride) = key.indices(self.size)
            if v: print(start,stop,stride)
            if isinstance(value, pdarray):
                generic_msg("[slice]=pdarray {} {} {} {} {}".format(self.name,start,stop,stride,value.name))
            else:
                generic_msg("[slice]=val {} {} {} {} {} {}".format(self.name, start, stop, stride, self.dtype.name, self.format_other(value)))
        else:
            return NotImplemented

    def __iter__(self):
        # to_ndarray will error if array is too large to bring back
        a = self.to_ndarray()
        for x in a:
            yield x
            
    def fill(self, value):
        """
        Fill the array (in place) with a constant value.
        """
        generic_msg("set {} {} {}".format(self.name, self.dtype.name, self.format_other(value)))

    def any(self):
        """
        Return True iff any element of the array evaluates to True.
        """
        return any(self)
    
    def all(self):
        """
        Return True iff all elements of the array evaluate to True.
        """
        return all(self)

    def is_sorted(self):
        """
        Return True iff the array is monotonically non-decreasing.
        """
        return is_sorted(self)
    
    def sum(self):
        """
        Return the sum of all elements in the array.
        """
        return sum(self)
    
    def prod(self):
        """
        Return the product of all elements in the array. Return value is
        always a float.
        """
        return prod(self)
    
    def min(self):
        """
        Return the minimum value of the array.
        """
        return min(self)
    
    def max(self):
        """
        Return the maximum value of the array.
        """
        return max(self)
    
    def argmin(self):
        """
        Return the index of the first minimum value of the array.
        """
        return argmin(self)
    
    def argmax(self):
        """
        Return the index of the first maximum value of the array.
        """
        return argmax(self)
    
    def mean(self):
        """
        Return the mean of the array.
        """
        return mean(self)
    
    def var(self, ddof=0):
        """
        Compute the variance. See ``arkouda.var`` for details.
        """
        return var(self, ddof=ddof)
    
    def std(self, ddof=0):
        """
        Compute the standard deviation. See ``arkouda.std`` for details.
        """
        return std(self, ddof=ddof)

    def to_ndarray(self):
        """
        Convert the array to a np.ndarray, transferring array data from the
        arkouda server to Python. If the array exceeds a builtin size limit, 
        a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            A numpy ndarray with the same attributes and data as the pdarray

        Notes
        -----
        The number of bytes in the array cannot exceed ``arkouda.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        array

        Examples
        --------
        >>> a = ak.arange(0, 5, 1)
        >>> a.to_ndarray()
        array([0, 1, 2, 3, 4])

        >>> type(a.to_ndarray())
        numpy.ndarray
        """
        # Total number of bytes in the array data
        arraybytes = self.size * self.dtype.itemsize
        # Guard against overflowing client memory
        if arraybytes > maxTransferBytes:
            raise RuntimeError("Array exceeds allowed size for transfer. Increase ak.maxTransferBytes to allow")
        # The reply from the server will be a bytes object
        rep_msg = generic_msg("tondarray {}".format(self.name), recv_bytes=True)
        # Make sure the received data has the expected length
        if len(rep_msg) != self.size*self.dtype.itemsize:
            raise RuntimeError("Expected {} bytes but received {}".format(self.size*self.dtype.itemsize, len(rep_msg)))
        # Use struct to interpret bytes as a big-endian numeric array
        fmt = '>{:n}{}'.format(self.size, structDtypeCodes[self.dtype.name])
        # Return a numpy ndarray
        return np.array(struct.unpack(fmt, rep_msg))

    def save(self, prefix_path, dataset='array', mode='truncate'):
        """
        Save the pdarray to HDF5. The result is a collection of HDF5 files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in HDF5 files (must not already exist)
        mode : {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.

        See Also
        --------
        save_all, load, read_hdf, read_all

        Notes
        -----
        The prefix_path must be visible to the arkouda server and the user must have
        write permission.

        Output files have names of the form ``<prefix_path>_LOCALE<i>.hdf``, where ``<i>``
        ranges from 0 to ``numLocales``. If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.

        Examples
        --------
        >>> a = ak.arange(0, 100, 1)
        >>> a.save('arkouda_range', dataset='array')

        Array is saved in numLocales files with names like ``tmp/arkouda_range_LOCALE0.hdf``

        The array can be read back in as follows

        >>> b = ak.load('arkouda_range', dataset='array')
        >>> (a == b).all()
        True
        """
        if mode.lower() in 'append':
            m = 1
        elif mode.lower() in 'truncate':
            m = 0
        else:
            raise ValueError("Allowed modes are 'truncate' and 'append'")
        rep_msg = generic_msg("tohdf {} {} {} {}".format(self.name, dataset, m, json.dumps([prefix_path])))
        
# flag to info all arrays from arkouda server
AllSymbols = "__AllSymbols__"

#################################
# functions which create pdarrays
#################################

# creates pdarray object
#   only after:
#       all values have been checked by python module and...
#       server has created pdarray already befroe this is called
def create_pdarray(repMsg):
    """
    Return a pdarray instance pointing to an array created by the arkouda server.
    The user should not call this function directly.
    """
    fields = repMsg.split()
    name = fields[1]
    dtype = fields[2]
    size = int(fields[3])
    ndim = int(fields[4])
    shape = [int(el) for el in fields[5][1:-1].split(',')]
    itemsize = int(fields[6])
    if v: print("{} {} {} {} {} {}".format(name,dtype,size,ndim,shape,itemsize))
    return pdarray(name,dtype,size,ndim,shape,itemsize)

def parse_single_value(msg):
    """
    Attempt to convert a scalar return value from the arkouda server to a numpy
    scalar in Python. The user should not call this function directly.
    """
    dtname, value = msg.split()
    dtype = np.dtype(dtname)
    if dtype == np.bool:
        if value == "True":
            return np.bool(True)
        elif value == "False":
            return np.bool(False)
        else:
            raise ValueError("unsupported value from server {} {}".format(dtype.name, value))
    try:
        return dtype.type(value)
    except:
        raise ValueError("unsupported value from server {} {}".format(dtype.name, value))

def ls_hdf(filename):
    """
    This function calls the h5ls utility on a filename visible to the arkouda 
    server.

    Parameters
    ----------
    filename : str
        The name of the file to pass to h5ls

    Returns
    -------
    str 
        The string output of `h5ls <filename>` from the server
    """
    return generic_msg("lshdf {}".format(json.dumps([filename])))

def read_hdf(dsetName, filenames):
    """
    Read a single dataset from multiple HDF5 files into an arkouda pdarray. 

    Parameters
    ----------
    dsetName : str
        The name of the dataset (must be the same across all files)
    filenames : list or str
        Either a list of filenames or shell expression

    Returns
    -------
    pdarray
        A pdarray instance pointing to the server-side data read in

    See Also
    --------
    get_datasets, ls_hdf, read_all, load, save

    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression 
    (a single filename is a valid expression, so it will work) and is 
    expanded with glob to read all matching files. Use ``get_datasets`` to 
    show the names of datasets in HDF5 files.

    If dsetName is not present in all files, a RuntimeError is raised.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    rep_msg = generic_msg("readhdf {} {:n} {}".format(dsetName, len(filenames), json.dumps(filenames)))
    return create_pdarray(rep_msg)

def read_all(filenames, datasets=None):
    """
    Read multiple datasets from multiple HDF5 files.
    
    Parameters
    ----------
    filenames : list or str
        Either a list of filenames or shell expression
    datasets : list or str or None
        (List of) name(s) of dataset(s) to read (default: all available)

    Returns
    -------
    dict of pdarrays
        Dictionary of {datasetName: pdarray}

    See Also
    --------
    read_hdf, get_datasets, ls_hdf
    
    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression 
    (a single filename is a valid expression, so it will work) and is 
    expanded with glob to read all matching files. This is done separately
    for each dataset, so if new matching files appear during ``read_all``,
    some datasets will contain more data than others. 

    If datasets is None, infer the names of datasets from the first file
    and read all of them. Use ``get_datasets`` to show the names of datasets in 
    HDF5 files.

    If not all datasets are present in all HDF5 files, a RuntimeError
    is raised.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    alldatasets = get_datasets(filenames[0])
    if datasets is None:
        datasets = alldatasets
    else: # ensure dataset(s) exist
        if isinstance(datasets, str):
            datasets = [datasets]
        nonexistent = set(datasets) - set(get_datasets(filenames[0]))
        if len(nonexistent) > 0:
            raise ValueError("Dataset(s) not found: {}".format(nonexistent))
    return {dset:read_hdf(dset, filenames) for dset in datasets}

def load(path_prefix, dataset='array'):
    """
    Load a pdarray previously saved with ``pdarray.save()``. 
    
    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray
    dataset : str
        Dataset name where the pdarray was saved

    Returns
    -------
    pdarray
        The pdarray that was previously saved

    See Also
    --------
    save, load_all, read_hdf, read_all
    """
    prefix, extension = os.path.splitext(path_prefix)
    globstr = "{}_LOCALE*{}".format(prefix, extension)
    return read_hdf(dataset, globstr)

def get_datasets(filename):
    """
    Get the names of datasets in an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of an HDF5 file visible to the arkouda server

    Returns
    -------
    list of str
        Names of the datasets in the file
    
    See Also
    --------
    ls_hdf
    """
    rep_msg = ls_hdf(filename)
    datasets = [line.split()[0] for line in rep_msg.splitlines()]
    return datasets
            
def load_all(path_prefix):
    """
    Load multiple pdarray previously saved with ``save_all()``. 
    
    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray

    Returns
    -------
    dict of pdarrays
        Dictionary of {datsetName: pdarray} with the previously saved pdarrays

    See Also
    --------
    save_all, load, read_hdf, read_all
    """
    prefix, extension = os.path.splitext(path_prefix)
    firstname = "{}_LOCALE0{}".format(prefix, extension)
    return {dataset: load(path_prefix, dataset=dataset) for dataset in get_datasets(firstname)}

def save_all(columns, path_prefix, names=None, mode='truncate'):
    """
    Save multiple named pdarrays to HDF5 files.

    Parameters
    ----------
    columns : dict or list of pdarrays
        Collection of arrays to save
    path_prefix : str
        Directory and filename prefix for output files
    names : list of str
        Dataset names for the pdarrays
    mode : {'truncate' | 'append'}
        By default, truncate (overwrite) the output files if they exist. 
        If 'append', attempt to create new dataset in existing files.

    See Also
    --------
    save, load_all

    Notes
    -----
    Creates one file per locale containing that locale's chunk of each pdarray.
    If columns is a dictionary, the keys are used as the HDF5 dataset names. 
    Otherwise, if no names are supplied, 0-up integers are used. By default, 
    any existing files at path_prefix will be overwritten, unless the user 
    specifies the 'append' mode, in which case arkouda will attempt to add 
    <columns> as new datasets to existing files. If the wrong number of files
    is present or dataset names already exist, a RuntimeError is raised.
    """
    if names is not None and len(names) != len(columns):
        raise ValueError("Number of names does not match number of columns")
    if isinstance(columns, dict):
        pdarrays = columns.values()
        if names is None:
            names = columns.keys()
    elif isinstance(columns, list):
        pdarrays = columns
        if names is None:
            names = range(len(columns))
    if (mode.lower() not in 'append') and (mode.lower() not in 'truncate'):
        raise ValueError("Allowed modes are 'truncate' and 'append'")
    first_iter = True
    for arr, name in zip(pdarrays, names):
        # Append all pdarrays to existing files as new datasets EXCEPT the first one, and only if user requests truncation
        if mode.lower() not in 'append' and first_iter:
            arr.save(path_prefix, dataset=name, mode='truncate')
            first_iter = False
        else:
            arr.save(path_prefix, dataset=name, mode='append')

def array(a):
    """
    Convert an iterable to a pdarray, sending data to the arkouda server.

    Parameters
    ----------
    a : array_like
        Rank-1 array of a supported dtype

    Returns
    -------
    pdarray
        Instance of pdarray stored on arkouda server

    See Also
    --------
    pdarray.to_ndarray

    Notes
    -----
    The number of bytes in the input array cannot exceed `arkouda.maxTransferBytes`,
    otherwise a RuntimeError will be raised. This is to protect the user
    from overwhelming the connection between the Python client and the arkouda
    server, under the assumption that it is a low-bandwidth connection. The user
    may override this limit by setting ak.maxTransferBytes to a larger value, 
    but should proceed with caution.

    Examples
    --------
    >>> a = [3, 5, 7]
    >>> b = ak.array(a)
    >>> b
    array([3, 5, 7])
   
    >>> type(b)
    arkouda.pdarray    
    """
    # If a is already a pdarray, do nothing
    if isinstance(a, pdarray):
        return a
    # If a is not already a numpy.ndarray, convert it
    if not isinstance(a, np.ndarray):
        try:
            a = np.array(a)
        except:
            raise TypeError("Argument must be array-like")
    # Only rank 1 arrays currently supported
    if a.ndim != 1:
        raise RuntimeError("Only rank-1 arrays supported")
    # Check that dtype is supported in arkouda
    if a.dtype.name not in DTypes:
        raise RuntimeError("Unhandled dtype {}".format(a.dtype))
    # Do not allow arrays that are too large
    size = a.size
    if (size * a.itemsize) > maxTransferBytes:
        raise RuntimeError("Array exceeds allowed transfer size. Increase ak.maxTransferBytes to allow")
    # Pack binary array data into a bytes object with a command header
    # including the dtype and size
    fmt = ">{:n}{}".format(size, structDtypeCodes[a.dtype.name])
    req_msg = "array {} {:n} ".format(a.dtype.name, size).encode() + struct.pack(fmt, *a)
    rep_msg = generic_msg(req_msg, send_bytes=True)
    return create_pdarray(rep_msg)

def zeros(size, dtype=np.float64):
    """
    Create a pdarray filled with zeros.

    Parameters
    ----------
    size : int
        Size of the array (only rank-1 arrays supported)
    dtype : {float64, int64, bool}
        Type of resulting array, default float64

    Returns
    -------
    pdarray
        Zeros of the requested size and dtype

    See Also
    --------
    ones, zeros_like

    Examples
    --------
    >>> ak.zeros(5, dtype=ak.int64)
    array([0, 0, 0, 0, 0])
    >>> ak.zeros(5, dtype=ak.float64)
    array([0, 0, 0, 0, 0])
    >>> ak.zeros(5, dtype=ak.bool)
    array([False, False, False, False, False])
    """
    dtype = np.dtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    kind, itemsize = translate_np_dtype(dtype)
    repMsg = generic_msg("create {} {}".format(dtype.name, size))
    return create_pdarray(repMsg)

def ones(size, dtype=float64):
    """
    Create a pdarray filled with ones.

    Parameters
    ----------
    size : int
        Size of the array (only rank-1 arrays supported)
    dtype : {float64, int64, bool}
        Resulting array type, default float64

    Returns
    -------
    pdarray
        Ones of the requested size and dtype

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> ak.ones(5, dtype=ak.int64)
    array([1, 1, 1, 1, 1])
    >>> ak.ones(5, dtype=ak.float64)
    array([1, 1, 1, 1, 1])
    >>> ak.ones(5, dtype=ak.bool)
    array([True, True, True, True, True])
    """
    dtype = np.dtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    kind, itemsize = translate_np_dtype(dtype)
    repMsg = generic_msg("create {} {}".format(dtype.name, size))
    a = create_pdarray(repMsg)
    a.fill(1)
    return a

def zeros_like(pda):
    """
    Create a zero-filled pdarray of the same size and dtype as an existing pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for size and dtype

    Returns
    -------
    pdarray
        Equivalent to ak.zeros(pda.size, pda.dtype)

    See Also
    --------
    zeros, ones_like
    """
    if isinstance(pda, pdarray):
        return zeros(pda.size, pda.dtype)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def ones_like(pda):
    """
    Create a one-filled pdarray of the same size and dtype as an existing pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for size and dtype

    Returns
    -------
    pdarray
        Equivalent to ak.ones(pda.size, pda.dtype)

    See Also
    --------
    ones, zeros_like
    """
    if isinstance(pda, pdarray):
        return ones(pda.size, pda.dtype)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def arange(*args):
    """
    arange([start,] stop[, stride])

    Create a pdarray of consecutive integers within the interval [start, stop).
    If only one arg is given then arg is the stop parameter. If two args are given
    then the first arg is start and second is stop. If three args are given
    then the first arg is start, second is stop, third is stride.

    Parameters
    ----------
    start : int, optional
        Starting value (inclusive), the default starting value is 0
    stop : int
        Stopping value (exclusive)
    stride : int, optional
        The difference between consecutive elements, the default stride is 1,
        if stride is specified then start must also be specified

    Returns
    -------
    pdarray, int64
        Integers from start (inclusive) to stop (exclusive) by stride

    See Also
    --------
    linspace, zeros, ones, randint
    
    Notes
    -----
    Negative strides result in decreasing values. Currently, only int64 pdarrays
    can be created with this function. For float64 arrays, use linspace.

    Examples
    --------
    >>> ak.arange(0, 5, 1)
    array([0, 1, 2, 3, 4])

    >>> ak.arange(5, 0, -1)
    array([5, 4, 3, 2, 1])

    >>> ak.arange(0, 10, 2)
    array([0, 2, 4, 6, 8])
    """
   
    #if one arg is given then arg is stop
    if len(args) == 1:
        start = 0
        stop = args[0]
        stride = 1

    #if two args are given then first arg is start and second is stop
    if len(args) == 2:
        start = args[0]
        stop = args[1]
        stride = 1

    #if three args are given then first arg is start,
    #second is stop, third is stride
    if len(args) == 3:
        start = args[0]
        stop = args[1]
        stride = args[2]

    if stride == 0:
        raise ZeroDivisionError("division by zero")

    if isinstance(start, int) and isinstance(stop, int) and isinstance(stride, int):
        # TO DO: fix bug in server that goes 2 steps too far for negative strides
        if stride < 0:
            stop = stop + 2
        repMsg = generic_msg("arange {} {} {}".format(start, stop, stride))
        return create_pdarray(repMsg)
    else:
        raise TypeError("start,stop,stride must be type int {} {} {}".format(start,stop,stride))

def linspace(start, stop, length):
    """
    Create a pdarray of linearly spaced points in a closed interval.

    Parameters
    ----------
    start : scalar
        Start of interval (inclusive)
    stop : scalar
        End of interval (inclusive)
    length : int
        Number of points

    Returns
    -------
    pdarray, float64
        Array of evenly spaced points along the interval

    See Also
    --------
    arange

    Examples
    --------
    >>> ak.linspace(0, 1, 5)
    array([0, 0.25, 0.5, 0.75, 1])
    """
    starttype = resolve_scalar_dtype(start)
    startstr = NUMBER_FORMAT_STRINGS[starttype].format(start)
    stoptype = resolve_scalar_dtype(stop)
    stopstr = NUMBER_FORMAT_STRINGS[stoptype].format(stop)
    lentype = resolve_scalar_dtype(length)
    if lentype != 'int64':
        raise TypeError("Length must be int64")
    lenstr = NUMBER_FORMAT_STRINGS[lentype].format(length)
    repMsg = generic_msg("linspace {} {} {}".format(startstr, stopstr, lenstr))
    return create_pdarray(repMsg)

def histogram(pda, bins=10):
    """
    Compute a histogram of evenly spaced bins over the range of an array.
    
    Parameters
    ----------
    pda : pdarray
        The values to histogram

    bins : int
        The number of equal-size bins to use (default: 10)

    Returns
    -------
    pdarray
        The number of values present in each bin

    See Also
    --------
    value_counts

    Notes
    -----
    The bins are evenly spaced in the interval [pda.min(), pda.max()]. Currently,
    the user must re-compute the bin edges, e.g. with np.linspace (see below) 
    in order to plot the histogram.

    Examples
    --------
    >>> A = ak.arange(0, 10, 1)
    >>> nbins = 3
    >>> h = ak.histogram(A, bins=nbins)
    >>> h
    array([3, 3, 4])
    # Recreate the bin edges in NumPy
    >>> binEdges = np.linspace(A.min(), A.max(), nbins+1)
    >>> binEdges
    array([0., 3., 6., 9.])
    # To plot, use only the left edges, and export the histogram to NumPy
    >>> plt.plot(binEdges[:-1], h.to_ndarray())
    """
    if isinstance(pda, pdarray) and isinstance(bins, int):
        repMsg = generic_msg("histogram {} {}".format(pda.name, bins))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {} and bins must be an int {}".format(pda,bins))

def in1d(pda1, pda2, invert=False):
    """
    Test whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as `pda1` that is True
    where an element of `pda1` is in `pda2` and False otherwise.

    Parameters
    ----------
    pda1 : pdarray
        Input array.
    pda2 : pdarray
        The values against which to test each value of `pda1`.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of `pda1` is in `pda2` and True otherwise).
        Default is False. ``ak.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``~ak.in1d(a, b)``.

    Returns
    -------
    pdarray, bool
        The values `pda1[in1d]` are in `pda2`.

    See Also
    --------
    unique, intersect1d, union1d

    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is logically
    equivalent to ``ak.array([item in b for item in a])``, but is much
    faster and scales to arbitrarily large ``a``.
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        repMsg = generic_msg("in1d {} {} {}".format(pda1.name, pda2.name, invert))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

def unique(pda, return_counts=False):
    """
    Find the unique elements of an array.

    Returns the sorted unique elements of an array. There is an optional
    output in addition to the unique elements: the number of times each 
    unique value comes up in the input array.

    Parameters
    ----------
    pda : pdarray
        Input array.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `pda`.

    Returns
    -------
    unique : pdarray
        The sorted unique values.
    unique_counts : pdarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    Notes
    -----
    Internally, this function checks to see whether `pda` is sorted and, if so,
    whether it is already unique. This step can save considerable computation.
    Otherwise, this function will sort `pda`.

    Examples
    --------
    >>> A = ak.array([3, 2, 1, 1, 2, 3])
    >>> ak.unique(A)
    array([1, 2, 3])
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("unique {} {}".format(pda.name, return_counts))
        if return_counts:
            vc = repMsg.split("+")
            if v: print(vc)
            return create_pdarray(vc[0]), create_pdarray(vc[1])
        else:
            return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def value_counts(pda):
    """
    Count the occurrences of the unique values of an array.

    Parameters
    ----------
    pda : pdarray, int64
        The array of values to count

    Returns
    -------
    unique_values : pdarray, int64
        The unique values, sorted in ascending order

    counts : pdarray, int64
        The number of times the corresponding unique value occurs

    See Also
    --------
    unique, histogram

    Notes
    -----
    This function differs from ``histogram()`` in that it only returns counts 
    for values that are present, leaving out empty "bins".

    Examples
    --------
    >>> A = ak.array([2, 0, 2, 4, 0, 0])
    >>> ak.value_counts(A)
    (array([0, 2, 4]), array([3, 2, 1]))
    """
    return unique(pda, return_counts=True)

def randint(low, high, size, dtype=np.int64):
    """
    Generate a pdarray with random values in a specified range.

    Parameters
    ----------
    low : int
        The low value (inclusive) of the range
    high : int
        The high value (exclusive for int, inclusive for float) of the range
    size : int
        The length of the returned array
    dtype : (np.int64 | np.float64 | np.bool)
        The dtype of the array

    Returns
    -------
    pdarray
        Values drawn uniformly from the specified range having the desired dtype

    Notes
    -----
    Calling randint with dtype=float64 will result in uniform non-integral
    floating point values.

    Examples
    --------
    >>> ak.randint(0, 10, 5)
    array([5, 7, 4, 8, 3])

    >>> ak.randint(0, 1, 3, dtype=ak.float64)
    array([0.92176432277231968, 0.083130710959903542, 0.68894208386667544])

    >>> ak.randint(0, 1, 5, dtype=ak.bool)
    array([True, False, True, True, True])
    """
    # TO DO: separate out into int and float versions
    # TO DO: float version should accept non-integer low and high
    dtype = np.dtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    if isinstance(low, int) and isinstance(high, int) and isinstance(size, int):
        kind, itemsize = translate_np_dtype(dtype)
        repMsg = generic_msg("randint {} {} {} {}".format(low,high,size,dtype.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("min,max,size must be int {} {} {}".format(low,high,size));

def argsort(pda):
    """
    Return the permutation that sorts the array.
    
    Parameters
    ----------
    pda : pdarray
        The array to sort (int64 or float64)

    Returns
    -------
    pdarray, int64
        The indices such that ``pda[indices]`` is sorted

    See Also
    --------
    coargsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive.

    Examples
    --------
    >>> a = ak.randint(0, 10, 10)
    >>> perm = ak.argsort(a)
    >>> a[perm]
    array([0, 1, 1, 3, 4, 5, 7, 8, 8, 9])
    """
    if isinstance(pda, pdarray):
        if pda.size == 0:
            return zeros(0, dtype=int64)
        repMsg = generic_msg("argsort {}".format(pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def coargsort(arrays):
    """
    Return the permutation that sorts the rows (left-to-right), if the
    input arrays are treated as columns.
    
    Parameters
    ----------
    arrays : iterable of pdarray
        The columns (int64 or float64) to sort by row

    Returns
    -------
    pdarray, int64
        The indices that permute the rows to sorted order

    See Also
    --------
    argsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive. Starts with the
    last array and moves forward.

    Examples
    --------
    >>> a = ak.array([0, 1, 0, 1])
    >>> b = ak.array([1, 1, 0, 0])
    >>> perm = ak.coargsort([a, b])
    >>> perm
    array([2, 0, 3, 1])
    >>> a[perm]
    array([0, 0, 1, 1])
    >>> b[perm]
    array([0, 1, 0, 1])
    """
    size = -1
    for a in arrays:
        if not isinstance(a, pdarray):
            raise ValueError("Argument must be an iterable of pdarrays")
        if size == -1:
            size = a.size
        elif size != a.size:
            raise ValueError("All pdarrays must have same size")
    if size == 0:
        return zeros(0, dtype=int64)
    repMsg = generic_msg("coargsort {} {}".format(len(arrays), ' '.join([a.name for a in arrays])))
    return create_pdarray(repMsg)

def concatenate(arrays):
    """
    Concatenate an iterable of ``pdarray`` objects into one ``pdarray``.

    Parameters
    ----------
    arrays : iterable of ``pdarray``
        The arrays to concatenate. Must all have same dtype.

    Returns
    -------
    pdarray
        Single array containing all values, in original order

    Examples
    --------
    >>> ak.concatenate([ak.array([1, 2, 3]), ak.array([4, 5, 6])])
    array([1, 2, 3, 4, 5, 6])
    """
    size = 0
    dtype = None
    for a in arrays:
        if not isinstance(a, pdarray):
            raise ValueError("Argument must be an iterable of pdarrays")
        if dtype == None:
            dtype = a.dtype
        elif dtype != a.dtype:
            raise ValueError("All pdarrays must have same dtype")
        size += a.size
    if size == 0:
        return zeros(0, dtype=int64)
    repMsg = generic_msg("concatenate {} {}".format(len(arrays), ' '.join([a.name for a in arrays])))
    return create_pdarray(repMsg)

# (A1 | A2) Set Union: elements are in one or the other or both
def union1d(pda1, pda2):
    """
    Find the union of two arrays.

    Return the unique, sorted array of values that are in either of the two
    input arrays.

    Parameters
    ----------
    pda1 : pdarray
        Input array
    pda2 : pdarray
        Input array

    Returns
    -------
    pdarray
        Unique, sorted union of the input arrays.

    See Also
    --------
    intersect1d, unique

    Examples
    --------
    >>> ak.union1d([-1, 0, 1], [-2, 0, 2])
    array([-2, -1,  0,  1,  2])
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda2 # union is pda2
        if pda2.size == 0:
            return pda1 # union is pda1
        return unique(concatenate((unique(pda1), unique(pda2))))
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

# (A1 & A2) Set Intersection: elements have to be in both arrays
def intersect1d(pda1, pda2, assume_unique=False):
    """
    Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    pda1 : pdarray
        Input array
    pda2 : pdarray
        Input array
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray
        Sorted 1D array of common and unique elements.

    See Also
    --------
    unique, union1d

    Examples
    --------
    >>> ak.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda1 # nothing in the intersection
        if pda2.size == 0:
            return pda2 # nothing in the intersection
        if not assume_unique:
            pda1 = unique(pda1)
            pda2 = unique(pda2)
        aux = concatenate((pda1, pda2))
        aux_sort_indices = argsort(aux)
        aux = aux[aux_sort_indices]
        mask = aux[1:] == aux[:-1]
        int1d = aux[:-1][mask]
        return int1d
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

# (A1 - A2) Set Difference: elements have to be in first array but not second
def setdiff1d(pda1, pda2, assume_unique=False):
    """
    Find the set difference of two arrays.

    Return the sorted, unique values in `pda1` that are not in `pda2`.

    Parameters
    ----------
    pda1 : pdarray
        Input array.
    pda2 : pdarray
        Input comparison array.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray
        Sorted 1D array of values in `pda1` that are not in `pda2`.

    See Also
    --------
    unique, setxor1d

    Examples
    --------
    >>> a = ak.array([1, 2, 3, 2, 4, 1])
    >>> b = ak.array([3, 4, 5, 6])
    >>> ak.setdiff1d(a, b)
    array([1, 2])
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda1 # return a zero length pdarray
        if pda2.size == 0:
            return pda1 # subtracting nothing return orig pdarray
        if not assume_unique:
            pda1 = unique(pda1)
            pda2 = unique(pda2)
        return pda1[in1d(pda1, pda2, invert=True)]
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

# (A1 ^ A2) Set Symmetric Difference: elements are not in the intersection
def setxor1d(pda1, pda2, assume_unique=False):
    """
    Find the set exclusive-or (symmetric difference) of two arrays.

    Return the sorted, unique values that are in only one (not both) of the
    input arrays.

    Parameters
    ----------
    pda1 : pdarray
        Input array.
    pda2 : pdarray
        Input array.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray
        Sorted 1D array of unique values that are in only one of the input
        arrays.

    Examples
    --------
    >>> a = ak.array([1, 2, 3, 2, 4])
    >>> b = ak.array([2, 3, 5, 7, 5])
    >>> ak.setxor1d(a,b)
    array([1, 4, 5, 7])
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda2 # return other pdarray if pda1 is empty
        if pda2.size == 0:
            return pda1 # return other pdarray if pda2 is empty
        if not assume_unique:
            pda1 = unique(pda1)
            pda2 = unique(pda2)
        aux = concatenate((pda1, pda2))
        aux_sort_indices = argsort(aux)
        aux = aux[aux_sort_indices]
        flag = concatenate((array([True]), aux[1:] != aux[:-1], array([True])))
        return aux[flag[1:] & flag[:-1]]
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))


def local_argsort(pda):
    if isinstance(pda, pdarray):
        if pda.size == 0:
            return zeros(0, dtype=int64)
        repMsg = generic_msg("localArgsort {}".format(pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def abs(pda):
    """
    Return the element-wise absolute value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("abs", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def log(pda):
    """
    Return the element-wise natural log of the array. 

    Notes
    -----
    Logarithms with other bases can be computed as follows:

    >>> A = ak.array([1, 10, 100])
    # Natural log
    >>> ak.log(A)
    array([0, 2.3025850929940459, 4.6051701859880918])
    # Log base 10
    >>> ak.log(A) / np.log(10)
    array([0, 1, 2])
    # Log base 2
    >>> ak.log(A) / np.log(2)
    array([0, 3.3219280948873626, 6.6438561897747253])
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("log", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def exp(pda):
    """
    Return the element-wise exponential of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("exp", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def cumsum(pda):
    """
    Return the cumulative sum over the array. 

    The sum is inclusive, such that the ``i`` th element of the 
    result is the sum of elements up to and including ``i``.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("cumsum", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def cumprod(pda):
    """
    Return the cumulative product over the array. 

    The product is inclusive, such that the ``i`` th element of the 
    result is the product of elements up to and including ``i``.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("cumprod", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def sin(pda):
    """
    Return the element-wise sine of the array.
    """
    if isinstance(pda,pdarray):
        repMsg = generic_msg("efunc {} {}".format("sin",pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def cos(pda):
    """
    Return the element-wise cosine of the array.
    """
    if isinstance(pda,pdarray):
        repMsg = generic_msg("efunc {} {}".format("cos",pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def any(pda):
    """
    Return True iff any element of the array evaluates to True.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("any", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def all(pda):
    """
    Return True iff all elements of the array evaluate to True.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("all", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
    
def is_sorted(pda):
    """
    Return True iff the array is monotonically non-decreasing.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("is_sorted", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def sum(pda):
    """
    Return the sum of all elements in the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("sum", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def prod(pda):
    """
    Return the product of all elements in the array. Return value is
    always a float.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("prod", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def min(pda):
    """
    Return the minimum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("min", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def max(pda):
    """
    Return the maximum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("max", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
    
def argmin(pda):
    """
    Return the index of the first minimum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmin", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def argmax(pda):
    """
    Return the index of the first maximum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmax", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def mean(pda):
    """
    Return the mean of the array.
    """
    return pda.sum() / pda.size

def var(pda, ddof=0):
    """
    Return the variance of values in the array.

    Parameters
    ----------
    pda : pdarray
        Values for which to find the variance
    ddof : int
        "Delta Degrees of Freedom" used in calculating mean

    Returns
    -------
    float
        The scalar variance of the array

    See Also
    --------
    mean, std

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean((x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.
    """
    if not isinstance(pda, pdarray):
        raise TypeError("must be pdarray {}".format(pda))
    if ddof >= pda.size:
        raise ValueError("var: ddof must be less than number of values")
    m = mean(pda)
    return ((pda - m)**2).sum() / (pda.size - ddof)

def std(pda, ddof=0):
    """
    Return the standard deviation of values in the array. The standard
    deviation is implemented as the square root of the variance.

    Parameters
    ----------
    pda : pdarray
        values for which to find the variance
    ddof : int
        "Delta Degrees of Freedom" used in calculating mean

    Returns
    -------
    float
        The scalar standard deviation of the array

    See Also
    --------
    mean, var

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean, i.e., ``std = sqrt(mean((x - x.mean())**2))``.

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
    the divisor ``N - ddof`` is used instead. In standard statistical
    practice, ``ddof=1`` provides an unbiased estimator of the variance
    of the infinite population. ``ddof=0`` provides a maximum likelihood
    estimate of the variance for normally distributed variables. The
    standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.
    """
    return np.sqrt(var(pda, ddof=ddof))
    
def where(condition, A, B):
    """
    Return an array with elements chosen from A and B based on a conditioning array.
    
    Parameters
    ----------
    condition : pdarray
        Used to choose values from A or B
    A : scalar or pdarray
        Value(s) used when condition is True
    B : scalar or pdarray
        Value(s) used when condition is False

    Returns
    -------
    pdarray
        Values chosen from A and B according to condition

    Notes
    -----
    A and B must have the same dtype.
    """
    if not isinstance(condition, pdarray):
        raise TypeError("must be pdarray {}".format(condition))
    if isinstance(A, pdarray) and isinstance(B, pdarray):
        repMsg = generic_msg("efunc3vv {} {} {} {}".format("where",
                                                           condition.name,
                                                           A.name,
                                                           B.name))
    # For scalars, try to convert it to the array's dtype
    elif isinstance(A, pdarray) and np.isscalar(B):
        repMsg = generic_msg("efunc3vs {} {} {} {} {}".format("where",
                                                              condition.name,
                                                              A.name,
                                                              A.dtype.name,
                                                              A.format_other(B)))
    elif isinstance(B, pdarray) and np.isscalar(A):
        repMsg = generic_msg("efunc3sv {} {} {} {} {}".format("where",
                                                              condition.name,
                                                              B.dtype.name,
                                                              B.format_other(A),
                                                              B.name))
    elif np.isscalar(A) and np.isscalar(B):
        # Scalars must share a common dtype (or be cast)
        dtA = resolve_scalar_dtype(A)
        dtB = resolve_scalar_dtype(B)
        # Make sure at least one of the dtypes is supported
        if not (dtA in DTypes or dtB in DTypes):
            raise TypeError("Not implemented for scalar types {} and {}".format(dtA, dtB))
        # If the dtypes are the same, do not cast
        if dtA == dtB:
            dt = dtA
        # If the dtypes are different, try casting one direction then the other
        elif dtB in DTypes and np.can_cast(A, dtB):
            A = np.dtype(dtB).type(A)
            dt = dtB
        elif dtA in DTypes and np.can_cast(B, dtA):
            B = np.dtype(dtA).type(B)
            dt = dtA
        # Cannot safely cast
        else:
            raise TypeError("Cannot cast between scalars {} and {} to supported dtype".format(A, B))
        repMsg = generic_msg("efunc3ss {} {} {} {} {} {}".format("where",
                                                                 condition.name,
                                                                 dt,
                                                                 A,
                                                                 dt,
                                                                 B))
    return create_pdarray(repMsg)
                             
            
class GroupBy:
    """
    Group an array by value, usually in preparation for aggregating the 
    within-group values of another array.

    Parameters
    ----------
    keys : (list of) pdarray, int64
        The array to group by value, or if list, the column arrays to group by row

    Attributes
    ----------
    nkeys : int
        The number of key arrays (columns)
    size : int
        The length of the array(s), i.e. number of rows
    permutation : pdarray
        The permutation that sorts the keys array(s) by value (row)
    unique_keys : pdarray
        The unique values of the keys array(s), in cosorted order
    segments : pdarray
        The start index of each group in the sorted array(s)
    unique_key_indices : pdarray
        The first index in the unsorted keys array(s) where each unique value (row) occurs

    Notes
    -----
    Only accepts pdarrays of int64 dtype.

    """
    Reductions = frozenset(['sum', 'prod', 'mean',
                            'min', 'max', 'argmin', 'argmax',
                            'nunique', 'any', 'all'])
    def __init__(self, keys):    
        self.per_locale = False
        self.keys = keys
        if isinstance(keys, pdarray):
            self.nkeys = 1
            self.size = keys.size
            if self.per_locale:
                self.permutation = local_argsort(keys)
            else:
                self.permutation = argsort(keys)
        else:
            self.nkeys = len(keys)
            self.size = keys[0].size
            for k in keys:
                if k.size != self.size:
                    raise ValueError("Key arrays must all be same size")
            self.permutation = coargsort(keys)
            
        # self.permuted_keys = self.keys[self.permutation]
        self.find_segments()
            
    def find_segments(self):
        if self.per_locale:
            cmd = "findLocalSegments"
        else:
            cmd = "findSegments"
        if self.nkeys == 1:
            keynames = self.keys.name
        else:
            keynames = ' '.join([k.name for k in self.keys])
        reqMsg = "{} {} {:n} {:n} {}".format(cmd,
                                             self.permutation.name,
                                             self.nkeys,
                                             self.size,
                                             keynames)
        repMsg = generic_msg(reqMsg)
        segAttr, uniqAttr = repMsg.split("+")
        if v: print(segAttr, uniqAttr)
        self.segments = create_pdarray(segAttr)
        self.unique_key_indices = create_pdarray(uniqAttr)
        if self.nkeys == 1:
            self.unique_keys = self.keys[self.unique_key_indices]
        else:
            self.unique_keys = [k[self.unique_key_indices] for k in self.keys]


    def count(self):
        '''
        Count the number of elements in each group, i.e. the number of times
        each key appears.

        Parameters
        ----------
        none

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        counts : pdarray, int64
            The number of times each unique key appears
        
        '''
        if self.per_locale:
            cmd = "countLocalRdx"
        else:
            cmd = "countReduction"
        reqMsg = "{} {} {}".format(cmd, self.segments.name, self.size)
        repMsg = generic_msg(reqMsg)
        if v: print(repMsg)
        return self.unique_keys, create_pdarray(repMsg)
        
    def aggregate(self, values, operator):
        '''
        Using the permutation stored in the GroupBy instance, group another array 
        of values and apply a reduction to each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and reduce
        operator: str
            The name of the reduction operator to use

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        aggregates : pdarray
            One aggregate value per unique key in the GroupBy instance

        '''
        if not isinstance(values, pdarray):
            raise TypeError("<values> must be a pdarray")
        if values.size != self.size:
            raise ValueError("Attempt to group array using key array of different length")
        if operator not in self.Reductions:
            raise ValueError("Unsupported reduction: {}\nMust be one of {}".format(operator, self.Reductions))
        permuted_values = values[self.permutation]
        if self.per_locale:
            cmd = "segmentedLocalRdx"
        else:
            cmd = "segmentedReduction"
        reqMsg = "{} {} {} {}".format(cmd,
                                         permuted_values.name,
                                         self.segments.name,
                                         operator)
        repMsg = generic_msg(reqMsg)
        if v: print(repMsg)
        if operator.startswith('arg'):
            return self.unique_keys, self.permutation[create_pdarray(repMsg)]
        else:
            return self.unique_keys, create_pdarray(repMsg)

    def sum(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and sum each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and sum

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_sums : pdarray
            One sum per unique key in the GroupBy instance

        Notes
        -----
        The grouped sum of a boolean ``pdarray`` returns integers.
        """
        return self.aggregate(values, "sum")
    
    def prod(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and compute the product of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and multiply

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_products : pdarray, float64
            One product per unique key in the GroupBy instance

        Notes
        -----
        The return dtype is always float64.
        """
        return self.aggregate(values, "prod")
    
    def mean(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and compute the mean of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and average

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_means : pdarray, float64
            One mean value per unique key in the GroupBy instance

        Notes
        -----
        The return dtype is always float64.
        """
        return self.aggregate(values, "mean")
    
    def min(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the minimum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find minima

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_minima : pdarray
            One minimum per unique key in the GroupBy instance

        """
        return self.aggregate(values, "min")
    
    def max(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the maximum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find maxima

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_maxima : pdarray
            One maximum per unique key in the GroupBy instance

        """
        return self.aggregate(values, "max")
    
    def argmin(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the location of the first minimum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find argmin

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_argminima : pdarray, int64
            One index per unique key in the GroupBy instance

        Notes
        -----
        The returned indices refer to the original values array as passed in, not
        the permutation applied by the GroupBy instance.

        Examples
        --------
        >>> A = ak.array([0, 1, 0, 1, 0, 1])
        >>> B = ak.array([0, 1, 1, 0, 0, 1])
        >>> byA = ak.GroupBy(A)
        >>> byA.argmin(B)
        (array([0, 1]), array([0, 3]))
        """
        return self.aggregate(values, "argmin")
    
    def argmax(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the location of the first maximum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find argmax

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_argmaxima : pdarray, int64
            One index per unique key in the GroupBy instance

        Notes
        -----
        The returned indices refer to the original values array as passed in, not
        the permutation applied by the GroupBy instance.

        Examples
        --------
        >>> A = ak.array([0, 1, 0, 1, 0, 1])
        >>> B = ak.array([0, 1, 1, 0, 0, 1])
        >>> byA = ak.GroupBy(A)
        >>> byA.argmax(B)
        (array([0, 1]), array([2, 1]))
        """
        return self.aggregate(values, "argmax")
    
    def nunique(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the number of unique values in each group. 

        Parameters
        ----------
        values : pdarray
            The values to group and find unique values

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_nunique : pdarray, int64
            Number of unique values per unique key in the GroupBy instance
        """
        return self.aggregate(values, "nunique")
    
    def any(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and perform an "or" reduction on each group. 

        Parameters
        ----------
        values : pdarray
            The values to group and reduce with "or"

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_any : pdarray, bool
            One bool per unique key in the GroupBy instance
        """
        return self.aggregate(values, "any")
    
    def all(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and perform an "and" reduction on each group. 

        Parameters
        ----------
        values : pdarray
            The values to group and reduce with "and"

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_any : pdarray, bool
            One bool per unique key in the GroupBy instance
        """
        return self.aggregate(values, "all")

    
    
# functions which query the server for information
def info(pda):
    if isinstance(pda, pdarray):
        return generic_msg("info {}".format(pda.name))
    elif isinstance(pda, str):
        return generic_msg("info {}".format(pda))
    else:
        raise TypeError("info: must be pdarray or string {}".format(pda))

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


################################################
# end of arkouda python definitions
################################################

########################
# a little quick testing
if __name__ == "__main__":
    v = True
    connect()


    # create some arrays and other things
    # and see the effect of the python __del__ method
    a = zeros(8, dtype=np.int64)
    a = zeros(10) # defaults to float64
    b = ones(8) # defaults to float64
    a = ones(8,np.int64)
    c = a + b + ones(8,dtype=np.int64)
    d = arange(1,10,2)
    e = arange(10, 30, 5)
    f = linspace(0, 2, 9)
    
    # print out some array names
    print(a.name)
    print(b.name)
    print(c.name)
    print(d.name)
    print(e.name)
    print(f.name)
    
    # check out assignment
    z = a
    print(z.name,a.name)
    a = ones(10,dtype=np.int64)
    print(z.name,a.name)
    
    # fill an array with a value
    a.fill(247)
    # get info of specific array
    info(a)
    
    info(AllSymbols)
    
    # shutdown arkouda server
    shutdown()

