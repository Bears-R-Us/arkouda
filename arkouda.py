#!/usr/bin/env python3

# arkouda python module -- python wrapper and comms
# arkouda is numpy like high perf dist arrays

# import zero mq
import zmq
import os
import subprocess
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
    else:
        message = socket.recv_string()
        if v: print("[Python] Received response: %s" % message)
        # raise errors sent back from the server
        if message.startswith("Error:"): raise RuntimeError(message)
    return message

# supported dtypes
structDtypeCodes = {'int64': 'q',
                    'float64': 'd',
                    'bool': '?'}
DTypes = frozenset(structDtypeCodes.keys())
NUMBER_FORMAT_STRINGS = {'int64': '{:n}',
                         'float64': '{:.17f}'}

def check_np_dtype(dt):
    if dt.name not in DTypes:
        raise TypeError("Unsupported type: {}".format(dt))

def translate_np_dtype(dt):
    check_np_dtype(dt)
    trans = {'i': 'int', 'f': 'float', 'b': 'bool'}
    kind = trans[dt.kind]
    return kind, dt.itemsize

BinOps = frozenset(["+", "-", "*", "/", "//", "<", ">", "<=", ">=", "!=", "==", "&", "|", "^", "<<", ">>"])
OpEqOps = frozenset(["+=", "-=", "*=", "/=", "//=", "&=", "|=", "^=", "<<=", ">>="])

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
            return str(other).lower()
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
        try:
            other = self.dtype.type(other)
        except: # Can't cast other as dtype of pdarray
            return NotImplemented
        msg = "binopvs {} {} {} {}".format(op, self.name, self.dtype.name, self.format_other(other))
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
        try:
            other = self.dtype.type(other)
        except: # Can't cast other as dtype of pdarray
            return NotImplemented
        msg = "binopvs {} {} {} {}".format(op, self.dtype.name, self.format_other(other), self.name)
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

    def parse_single_value(self, value):
        if self.dtype == np.bool:
            if value == "True":
                val = True
            elif value == "False":
                val = False
            else:
                raise ValueError("unsupported value from server {}".format(value))
        else:
            val = self.dtype.type(value)
        return val
    
    # overload a[] to treat like list
    def __getitem__(self, key):
        if isinstance(key, int):
            if (key >= 0 and key < self.size):
                repMsg = generic_msg("[int] {} {}".format(self.name, key))
                fields = repMsg.split()
                value = fields[2]
                return self.parse_single_value(value)
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
                generic_msg("[slice]=val {} {} {} {} {}".format(self.name, start, stop, stride, self.dtype.name, self.format_other(value)))
        else:
            return NotImplemented

    # needs better impl but ok for now
    def __iter__(self):
        global pdarrayIterThresh
        if (self.size <= pdarrayIterThresh) or (self.size <= 6):
            for i in range(0, self.size):
                yield self[i]
        else:
            for i in range(0, 3):
                yield self[i]
            yield ...
            for i in range(self.size-3, self.size):
                yield self[i]
            
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
    def unique(self, return_counts=False):
        return unique(self, return_counts)
    def value_counts(self):
        return value_counts(self)

    def to_ndarray(self):
        arraybytes = self.size * self.dtype.itemsize
        if arraybytes > maxTransferBytes:
            raise RuntimeError("Array exceeds allowed size for transfer. Increase ak.maxTransferBytes to allow")
        rep_msg = generic_msg("tondarray {}".format(self.name), recv_bytes=True)
        if len(rep_msg) != self.size*self.dtype.itemsize:
            raise RuntimeError("Expected {} bytes but received {}".format(self.size*self.dtype.itemsize, len(rep_msg)))
        fmt = '>{:n}{}'.format(self.size, structDtypeCodes[self.dtype.name])
        return np.array(struct.unpack(fmt, rep_msg))
        
# flag to info and dump all arrays from arkouda server
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

def read_hdf(dsetName, filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    rep_msg = generic_msg("readhdf {} {:n} {}".format(dsetName, len(filenames), json.dumps(filenames)))
    return create_pdarray(rep_msg)

def array(a):
    if isinstance(a, pdarray):
        return a
    try:
        a = np.array(a)
    except:
        raise TypeError("Argument must be array-like")
    if a.ndim != 1:
        raise RuntimeError("Only rank-1 arrays supported")
    if a.dtype.name not in DTypes:
        raise RuntimeError("Unhandled dtype {}".format(a.dtype))
    size = a.shape[0]
    if size > maxTransferBytes:
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

def any(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("any", pda.name))
        return pda.parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def all(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("all", pda.name))
        return pda.parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
    
def sum(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("sum", pda.name))
        return pda.parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def prod(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("prod", pda.name))
        return pda.parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def min(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("min", pda.name))
        return pda.parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def max(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("max", pda.name))
        return pda.parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
    
def argmin(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmin", pda.name))
        return pda.parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def argmax(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmax", pda.name))
        return pda.parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

# functions which query the server for information
def info(pda):
    if isinstance(pda, pdarray):
        return generic_msg("info {}".format(pda.name))
    elif isinstance(pda, str):
        return generic_msg("info {}".format(pda))
    else:
        raise TypeError("info: must be pdarray or string {}".format(pda))

def dump(pda):
    if isinstance(pda, pdarray):
        return generic_msg("dump {}".format(pda.name))
    elif isinstance(pda, str):
        return generic_msg("dump {}".format(pda))
    else:
        raise TypeError("dump: must be pdarray or string {}".format(pda))


################################################
# end of arkouda python definitions
################################################

########################
# a little quick testing
if __name__ == "__main__":
    v = True
    connect()

    dump(AllSymbols)

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
    # dump a specific array
    dump(a)
    
    info(AllSymbols)
    
    dump(AllSymbols)
    
    # shutdown arkouda server
    shutdown()

