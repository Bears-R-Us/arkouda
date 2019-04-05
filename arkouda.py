#!/usr/bin/env python3

# arkouda python module -- python wrapper and comms
# arkouda is numpy like high perf dist arrays

# import zero mq
import zmq
import os
import subprocess
import json

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

# reset settings to default values
def set_defaults():
    global v, vDefVal, pdarrayIterThresh, pdarrayIterThreshDefVal 
    v = vDefVal
    pdarrayIterThresh  = pdarrayIterThreshDefVal 

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
def generic_msg(message):
    global v, context, socket
    if v: print("[Python] Sending request: %s" % message)
    socket.send_string(message)
    message = socket.recv_string()
    if v: print("[Python] Received response: %s" % message)
    # raise errors sent back from the server
    if (message.split())[0] == "Error:": raise RuntimeError(message)
    return message

# supported dtypes
bool_ = type(True) # save bool type into bool_
bool = "bool" # seems dangerous but numpy redefines bool also # how do you call bool() in this context?
int64 = "int64"
float64 = "float64"
DTypes = frozenset([int64, float64, bool]) # remember bool is a string here blah!
BinOps = frozenset(["+", "-", "*", "/", "//", "<", ">", "<=", ">=", "!=", "==", "&", "|", "^", "<<", ">>"])
OpEqOps = frozenset(["+=", "-=", "*=", "/=", "//=", "&=", "|=", "^=", "<<=", ">>="])

# class for the pdarray
class pdarray:
    def __init__(self, name, dtype, size, ndim, shape, itemsize):
        self.name = name
        self.dtype = dtype
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
        # pdarray binop int
        elif isinstance(other, int):
            msg = "binopvs {} {} {} {}".format(op, self.name, int64, other)
            repMsg = generic_msg(msg)
            return create_pdarray(repMsg)
        # pdarray binop float
        elif isinstance(other, float):
            msg = "binopvs {} {} {} {}".format(op, self.name, float64, other)
            repMsg = generic_msg(msg)            
            return create_pdarray(repMsg)
        else:
            return NotImplemented

    # reverse binary operators
    # pdarray binop pdarray: taken care of by binop function
    def r_binop(self, other, op):
        if op not in BinOps:
            raise ValueError("bad operator {}".format(op))
        # int binop pdarray
        if isinstance(other, int):
            msg = "binopsv {} {} {} {}".format(op, int64, other, self.name)
            repMsg = generic_msg(msg)
            return create_pdarray(repMsg)
        # float binop pdarray
        elif isinstance(other, float):
            msg = "binopsv {} {} {} {}".format(op, float64, other, self.name)
            repMsg = generic_msg(msg)            
            return create_pdarray(repMsg)
        else:
            return NotImplemented

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
        # pdarray op= int
        elif isinstance(other, int):
            generic_msg("opeqvs {} {} {}".format(op, self.name, int64, other))
            return self
        # pdarray op= float
        elif isinstance(other, float):
            generic_msg("opeqvs {} {} {}".format(op, self.name, float64, other))
            return self
        else:
            return NotImplemented

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

    # overload a[] to treat like list
    def __getitem__(self, key):
        if isinstance(key, int):
            if (key >= 0 and key < self.size):
                repMsg = generic_msg("[int] {} {}".format(self.name, key))
                fields = repMsg.split()
                value = fields[2]
                if self.dtype == int64:
                    return int(value)
                elif self.dtype == float64:
                    return float(value)
                elif self.dtype == bool: # remember bool is a string here blah!
                    val = False
                    if value == "True": val = True
                    elif value == "False": val = False
                    else: ValueError("unsupported value from server {}".format(value))
                    return val
                else:
                    raise TypeError("unsupported value type from server {}".format(self.dtype))
            else:
                raise IndexError("[int] {} is out of bounds with size {}".format(key,self.size))
        if isinstance(key, slice):
            (start,stop,stride) = key.indices(self.size)
            if v: print(start,stop,stride)
            repMsg = generic_msg("[slice] {} {} {} {}".format(self.name, start, stop, stride))
            return create_pdarray(repMsg);
        if isinstance(key, pdarray):
            if key.dtype == int64:
                repMsg = generic_msg("[pdarray] {} {}".format(self.name, key.name))
                return create_pdarray(repMsg);
            elif key.dtype == bool: # remember bool is a string here blah!
                if self.size != key.size:
                    raise ValueError("size mismatch {} {}".format(self.size,key.size))
                repMsg = generic_msg("[pdarray] {} {}".format(self.name, key.name))
                return create_pdarray(repMsg);
            else:
                raise TypeError("unsupported pdarray index type {}".format(key.dtype))
        else:
            return NotImplemented

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if (key >= 0 and key < self.size):
                if isinstance(value, bool_): # we set bool_ = type(True) to test for bool
                    # remember bool is a string here blah!
                    generic_msg("[int]=val {} {} {} {}".format(self.name,key,bool,value))
                elif isinstance(value, int):
                    generic_msg("[int]=val {} {} {} {}".format(self.name,key,int64,value))
                elif isinstance(value, float):
                    generic_msg("[int]=val {} {} {} {}".format(self.name,key,float64,value))
                else:
                    raise TypeError("unsupported value type")
            else:
                raise IndexError("index {} is out of bounds with size {}".format(key,self.size))
        elif isinstance(key, pdarray):
            if isinstance(value, bool_): # we set bool_ = type(True) to test for bool
                # remember bool is a string here blah!
                generic_msg("[pdarray]=val {} {} {} {}".format(self.name,key.name,bool,value))
            elif isinstance(value, int):
                generic_msg("[pdarray]=val {} {} {} {}".format(self.name,key.name,int64,value))
            elif isinstance(value, float):
                generic_msg("[pdarray]=val {} {} {} {}".format(self.name,key.name,float64,value))
            elif isinstance(value, pdarray):
                generic_msg("[pdarray]=pdarray {} {} {}".format(self.name,key.name,value.name))
        elif isinstance(key, slice):
            (start,stop,stride) = key.indices(self.size)
            if v: print(start,stop,stride)
            if isinstance(value, bool_): # we set bool_ = type(True) to test for bool
                # remember bool is a string here blah!
                generic_msg("[slice]=val {} {} {} {} {} {}".format(self.name,start,stop,stride,bool,value))
            elif isinstance(value, int):
                generic_msg("[slice]=val {} {} {} {} {} {}".format(self.name,start,stop,stride,int64,value))
            elif isinstance(value, float):
                generic_msg("[slice]=val {} {} {} {} {} {}".format(self.name,start,stop,stride,float64,value))
            elif isinstance(value, pdarray):
                generic_msg("[slice]=pdarray {} {} {} {} {}".format(self.name,start,stop,stride,value.name))
            else:
                raise TypeError("unsupported value type")
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
        # error check and set dtype
        if isinstance(value, int): dtype = int64
        elif isinstance(value, float): dtype = float64
        elif isinstance(value, bool_): dtype = bool # remember bool is a string here blah! test with bool_
        else: raise TypeError("unsupported value type {}".format(type(value)))
        generic_msg("set {} {} {}".format(self.name, dtype, value))

    def any(self):
        return any(self)
    def all(self):
        return all(self)
    def sum(self):
        return sum(self)
    def prod(self):
        return prod(self)
    def argmin(self):
        return argmin(self)
    def argmax(self):
        return argmax(self)
    def unique(self, return_counts=False):
        return unique(self, return_counts)
    def value_counts(self):
        return value_counts(self)

    def to_ndarray(self):
        pass
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
    if a.ndim != 1:
        print("Only rank-1 arrays supported")
        return None
    codes = {'int64': 'q',
             'float64': 'd',
             'bool': '?'}
    if a.dtype.name not in codes:
        print("Unhandled dtype {}".format(a.dtype))
        return None
    size = a.shape[0]
    fmt = "<{:n}{}".format(size, codes[a.dtype.name])
    rep_msg = "array {} {:n} ".format(a.dtype.name, size).encode() + struct.pack(fmt, *a)
    return create_pdarray(rep_msg)

def zeros(size, dtype=float64):
    # check dtype for error
    if dtype not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    repMsg = generic_msg("create {} {}".format(dtype, size))
    return create_pdarray(repMsg)

def ones(size, dtype=float64):
    # check dtype for error
    if dtype not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    repMsg = generic_msg("create {} {}".format(dtype, size))
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

def randint(low, high, size, dtype=int64):
    # check dtype for error
    if dtype not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    if isinstance(low, int) and isinstance(high, int) and isinstance(size, int):
        repMsg = generic_msg("randint {} {} {} {}".format(low,high,size,dtype))
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
        fields = repMsg.split()
        dtype = fields[0]
        value = fields[1]
        val = None
        if value == "True": val = True
        elif value == "False": val = False
        else: ValueError("unsupported value from server {}".format(value))
        return val
    else:
        raise TypeError("must be pdarray {}".format(pda))

def all(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("all", pda.name))
        fields = repMsg.split()
        dtype = fields[0]
        value = fields[1]
        val = None
        if value == "True": val = True
        elif value == "False": val = False
        else: ValueError("unsupported value from server {}".format(value))
        return val
    else:
        raise TypeError("must be pdarray {}".format(pda))

def sum(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("sum", pda.name))
        fields = repMsg.split()
        dtype = fields[0]
        value = fields[1]
        val = None
        if dtype == int64: val = int(value)
        elif dtype == float64: val = float(value)
        elif dtype == bool: # remember bool is a string here blah!
            if value == "True": val = True
            elif value == "False": val = False
            else: raise ValueError("unsupported value from server {}".format(value))
        else: raise TypeError("unsupported dtype from server {}".format(dtype))
        return val
    else:
        raise TypeError("must be pdarray {}".format(pda))

def prod(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("prod", pda.name))
        fields = repMsg.split()
        dtype = fields[0]
        value = fields[1]
        val = None
        if dtype == int64: val = int(value)
        elif dtype == float64: val = float(value)
        elif dtype == bool: # remember bool is a string here blah!
            if value == "True": val = True
            elif value == "False": val = False
            else: raise ValueError("unsupported value from server {}".format(value))
        else: raise TypeError("unsupported dtype from server {}".format(dtype))
        return val
    else:
        raise TypeError("must be pdarray {}".format(pda))

def argmin(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmin", pda.name))
        fields = repMsg.split()
        dtype = fields[0]
        value = fields[1]
        val = None
        if dtype == int64: val = int(value)
        elif dtype == float64: val = float(value)
        elif dtype == bool: # remember bool is a string here blah!
            if value == "True": val = True
            elif value == "False": val = False
            else: raise ValueError("unsupported value from server {}".format(value))
        else: raise TypeError("unsupported dtype from server {}".format(dtype))
        return val
    else:
        raise TypeError("must be pdarray {}".format(pda))

def argmax(pda):
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmax", pda.name))
        fields = repMsg.split()
        dtype = fields[0]
        value = fields[1]
        val = None
        if dtype == int64: val = int(value)
        elif dtype == float64: val = float(value)
        elif dtype == bool: # remember bool is a string here blah!
            if value == "True": val = True
            elif value == "False": val = False
            else: raise ValueError("unsupported value from server {}".format(value))
        else: raise TypeError("unsupported dtype from server {}".format(dtype))
        return val
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
    a = zeros(8, dtype=int64)
    a = zeros(10) # defaults to float64
    b = ones(8) # defaults to float64
    a = ones(8,int64)
    c = a + b + ones(8,dtype=int64)
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
    a = ones(10,dtype=int64)
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

