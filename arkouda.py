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
    if dt.name not in DTypes:
        raise TypeError("Unsupported type: {}".format(dt))

def translate_np_dtype(dt):
    check_np_dtype(dt)
    trans = {'i': 'int', 'f': 'float', 'b': 'bool'}
    kind = trans[dt.kind]
    return kind, dt.itemsize

def resolve_scalar_dtype(val):
    '''Try to infer what dtype arkouda_server should treat <val> as.'''
    if isinstance(val, bool) or (hasattr(val, 'dtype') and val.dtype.kind == 'b'):
        return 'bool'
    elif isinstance(val, int) or (hasattr(val, 'dtype') and val.dtype.kind in 'ui'):
        return 'int64'
    elif isinstance(val, float) or (hasattr(val, 'dtype') and val.dtype.kind == 'f'):
        return 'float64'
    elif hasattr(val, 'dtype'):
        return dtype.name
    else:
        return str(type(val))

BinOps = frozenset(["+", "-", "*", "/", "//", "%", "<", ">", "<=", ">=", "!=", "==", "&", "|", "^", "<<", ">>","**"])
OpEqOps = frozenset(["+=", "-=", "*=", "/=", "//=", "&=", "|=", "^=", "<<=", ">>=","**="])

# class for the pdarray
class pdarray:
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
        ## s = repr([e for e in self])
        ## s = s.replace(",","")
        ## s = s.replace("Ellipsis","...")
        ## return s

    def __repr__(self):
        global pdarrayIterTresh
        return generic_msg("repr {} {}".format(self.name,pdarrayIterThresh))
        ## s = repr([e for e in self])
        ## s = s.replace("Ellipsis","...")
        ## s = "array(" + s + ")"
        ## return s

    def format_other(self, other):
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
        return self.binop(~0, "^")

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
        generic_msg("set {} {} {}".format(self.name, self.dtype.name, self.format_other(value)))

    def any(self):
        return any(self)
    def all(self):
        return all(self)
    def sum(self):
        return sum(self)
    def prod(self):
        return prod(self)
    def min(self):
        return min(self)
    def max(self):
        return max(self)
    def argmin(self):
        return argmin(self)
    def argmax(self):
        return argmax(self)

    def to_ndarray(self):
        arraybytes = self.size * self.dtype.itemsize
        if arraybytes > maxTransferBytes:
            raise RuntimeError("Array exceeds allowed size for transfer. Increase ak.maxTransferBytes to allow")
        rep_msg = generic_msg("tondarray {}".format(self.name), recv_bytes=True)
        if len(rep_msg) != self.size*self.dtype.itemsize:
            raise RuntimeError("Expected {} bytes but received {}".format(self.size*self.dtype.itemsize, len(rep_msg)))
        fmt = '>{:n}{}'.format(self.size, structDtypeCodes[self.dtype.name])
        return np.array(struct.unpack(fmt, rep_msg))

    def save(self, prefix_path, dataset='array', mode='truncate'):
        if mode.lower() == 'append':
            m = 1
        else:
            m = 0
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
    '''Return the string output of `h5ls <filename>` from the server.
    '''
    return generic_msg("lshdf {}".format(json.dumps([filename])))

def read_hdf(dsetName, filenames):
    '''Read a single dataset named <dsetName> from all HDF5 files in <filenames>. If <filenames> is a string, it will be interpreted as a glob expression (a single filename is a valid glob expression, so it will work). Returns a pdarray.
    '''
    if isinstance(filenames, str):
        filenames = [filenames]
    rep_msg = generic_msg("readhdf {} {:n} {}".format(dsetName, len(filenames), json.dumps(filenames)))
    return create_pdarray(rep_msg)

def read_all(filenames, datasets=None):
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
    '''Load a pdarray previously saved with .save(<path_prefix>). If <dataset> is supplied, it must match the name used to save the pdarray.
    '''
    prefix, extension = os.path.splitext(path_prefix)
    globstr = "{}_LOCALE*{}".format(prefix, extension)
    return read_hdf(dataset, globstr)

def get_datasets(filename):
    '''Return the list of dataset names contained in the HDF5 file <filename>.
    '''
    rep_msg = ls_hdf(filename)
    datasets = [line.split()[0] for line in rep_msg.splitlines()]
    return datasets
            
def load_all(path_prefix):
    prefix, extension = os.path.splitext(path_prefix)
    firstname = "{}_LOCALE0{}".format(prefix, extension)
    return {dataset: load(path_prefix, dataset=dataset) for dataset in get_datasets(firstname)}

def save_all(columns, path_prefix, names=None, mode='truncate'):
    '''Save all pdarrays in <columns> to files under <path_prefix>. Arkouda will create one file per locale but will keep pdarrays together in the same file as datasets called <names>. If <names> is not supplied and <columns> is a dict, arkouda will try to use the keys as dataset names; otherwise, dataset names are 0-up integers. By default, any existing files at <path_prefix> will be overwritten, unless the user supplies 'append' for the <mode>, in which case arkouda will attempt to add <columns> as new datasets to existing files.
    '''
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
    first_iter = True
    for arr, name in zip(pdarrays, names):
        # Append all pdarrays to existing files as new datasets EXCEPT the first one, and only if user requests truncation
        if mode.lower != 'append' and first_iter:
            arr.save(path_prefix, dataset=name, mode='truncate')
            first_iter = False
        else:
            arr.save(path_prefix, dataset=name, mode='append')

def array(a):
    if isinstance(a, pdarray):
        return a
    if not isinstance(a, np.ndarray):
        try:
            a = np.array(a)
        except:
            raise TypeError("Argument must be array-like")
    if a.ndim != 1:
        raise RuntimeError("Only rank-1 arrays supported")
    if a.dtype.name not in DTypes:
        raise RuntimeError("Unhandled dtype {}".format(a.dtype))
    size = a.size
    if (size * a.itemsize) > maxTransferBytes:
        raise RuntimeError("Array exceeds allowed transfer size. Increase ak.maxTransferBytes to allow")
    fmt = ">{:n}{}".format(size, structDtypeCodes[a.dtype.name])
    req_msg = "array {} {:n} ".format(a.dtype.name, size).encode() + struct.pack(fmt, *a)
    rep_msg = generic_msg(req_msg, send_bytes=True)
    return create_pdarray(rep_msg)

def zeros(size, dtype=np.float64):
    dtype = np.dtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    kind, itemsize = translate_np_dtype(dtype)
    repMsg = generic_msg("create {} {}".format(dtype.name, size))
    return create_pdarray(repMsg)

def ones(size, dtype=np.float64):
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
    if isinstance(pda, pdarray):
        return zeros(pda.size, pda.dtype)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def ones_like(pda):
    if isinstance(pda, pdarray):
        return ones(pda.size, pda.dtype)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def arange(start, stop, stride):
    if isinstance(start, int) and isinstance(stop, int) and isinstance(stride, int):
        repMsg = generic_msg("arange {} {} {}".format(start, stop, stride))
        return create_pdarray(repMsg)
    else:
        raise TypeError("start,stop,stride must be type int {} {} {}".format(start,stop,stride))

def linspace(start, stop, length):
    if (isinstance(start, int) or isinstance(start, float)) and (isinstance(stop, int) or isinstance(stop, float)) and (isinstance(length, int) or isinstance(length, float)) :
        repMsg = generic_msg("linspace {} {} {}".format(start, stop, length))
        return create_pdarray(repMsg)
    else:
        raise TypeError("start,stop,length must be type int or float {} {} {}".format(start,stop,length))

def histogram(pda, bins=10):
    if isinstance(pda, pdarray) and isinstance(bins, int):
        repMsg = generic_msg("histogram {} {}".format(pda.name, bins))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {} and bins must be an int {}".format(pda,bins))

def in1d(pda1, pda2):
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        repMsg = generic_msg("in1d {} {}".format(pda1.name, pda2.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {} and bins must be an int {}".format(pda,bins))

def unique(pda, return_counts=False):
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
    if isinstance(pda, pdarray):
        repMsg = generic_msg("value_counts {}".format(pda.name))
        vc = repMsg.split("+")
        if v: print(vc)
        return create_pdarray(vc[0]), create_pdarray(vc[1])
    else:
        raise TypeError("must be pdarray {}".format(pda))

def randint(low, high, size, dtype=np.int64):
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
    if isinstance(pda, pdarray):
        if pda.size == 0:
            return zeros(0, dtype=int64)
        repMsg = generic_msg("argsort {}".format(pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def coargsort(arrays):
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

def local_argsort(pda):
    if isinstance(pda, pdarray):
        if pda.size == 0:
            return zeros(0, dtype=int64)
        repMsg = generic_msg("localArgsort {}".format(pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def abs(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("abs", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def log(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("log", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def exp(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("exp", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def cumsum(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("cumsum", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def cumprod(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("efunc {} {}".format("cumprod", pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def sin(pda):
    if isinstance(pda,pdarray):
        repMsg = generic_msg("efunc {} {}".format("sin",pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def cos(pda):
    if isinstance(pda,pdarray):
        repMsg = generic_msg("efunc {} {}".format("cos",pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def any(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("any", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def all(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("all", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
    
def is_sorted(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("is_sorted", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def sum(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("sum", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def prod(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("prod", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def min(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("min", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def max(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("max", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
    
def argmin(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmin", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def argmax(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmax", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def where(condition, A, B):
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
    Reductions = frozenset(['sum', 'prod', 'mean',
                            'min', 'max', 'argmin', 'argmax',
                            'nunique', 'any', 'all'])
    def __init__(self, keys, per_locale=False):
        '''Group <keys> by value, usually in preparation for grouping
        and aggregating the values of another array via the
        .aggregate() method. Return a GroupBy object that stores the
        information for how to group values.
        '''
            
        self.per_locale = False
        self.keys = keys
        if isinstance(keys, pdarray):
            self.nkeys = 1
            self.size = keys.size
            if per_locale:
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
        # steps = zeros(self.size-1, dtype=bool)
        # if self.nkeys == 1:
        #     keys = [self.keys]
        # else:
        #     keys = self.keys
        # for k in keys:
        #     kperm = k[self.permutation]
        #     steps |= (kperm[:-1] != kperm[1:])
        # ukeyinds = zeros(self.size, dtype=bool)
        # ukeyinds[0] = True
        # ukeyinds[1:] = steps
        # #nsegments = ukeyinds.sum()
        # self.segments = arange(0, self.size, 1)[ukeyinds]
        # self.unique_key_indices = self.permutation[ukeyinds]

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
        '''Return the number of elements in each group, i.e. the number of times each key occurs.
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
        '''Using the grouping implied by self.keys, group <values> and reduce each group with <operator>. The result is one aggregate value per key, so the function returns the pdarray of keys and the pdarray of aggregate values.
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
        return self.aggregate(values, "sum")
    def prod(self, values):
        return self.aggregate(values, "prod")
    def mean(self, values):
        return self.aggregate(values, "mean")
    def min(self, values):
        return self.aggregate(values, "min")
    def max(self, values):
        return self.aggregate(values, "max")
    def argmin(self, values):
        return self.aggregate(values, "argmin")
    def argmax(self, values):
        return self.aggregate(values, "argmax")
    def nunique(self, values):
        return self.aggregate(values, "nunique")
    def any(self, values):
        return self.aggregate(values, "any")
    def all(self, values):
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
    return json.loads(generic_msg("getconfig"))

# query the server to get pda memory used 
def get_mem_used():
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

