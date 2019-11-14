from arkouda.client import generic_msg, verbose, pdarrayIterThresh
from arkouda.pdarrayclass import pdarray, create_pdarray, parse_single_value
from arkouda.dtypes import *
from numpy import isscalar

global verbose
global pdarrayIterThresh

__all__ = ['Strings']

class Strings:
    BinOps = frozenset(["==", "!="])
    objtype = "str"
    
    def __init__(self, offset_attrib, bytes_attrib):
        self.offsets = create_pdarray(offset_attrib)
        self.bytes = create_pdarray(bytes_attrib)
        self.size = self.offsets.size
        self.nbytes = self.bytes.size
        self.ndim = self.offsets.ndim
        self.shape = self.offsets.shape

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        if self.size <= pdarrayIterThresh:
            vals = [self[i] for i in range(self.size)]
        else:
            vals = [self[i] for i in range(3)]
            vals.append(' ... ')
            vals.extend([self[i] for i in range(self.size-3, self.size)])
        return "[{}]".format(', '.join(vals))

    def __repr__(self):
        return "array({})".format(self.__str__())

    def binop(self, other, op):
        if op not in self.BinOps:
            raise ValueError("Strings: unsupported operator: {}".format(op))
        if isinstance(other, Strings):
            if self.size != other.size:
                raise ValueError("Strings: size mismatch {} {}".format(self.size, other.size))
            msg = "segmentedBinopvv {} {} {} {} {} {} {}".format(op,
                                                                 self.objtype,
                                                                 self.offsets.name,
                                                                 self.bytes.name,
                                                                 other.dtype.name,
                                                                 other.offsets.name,
                                                                 other.bytes.name)
        elif resolve_scalar_dtype(other) == 'str':
            msg = "segmentedBinopvs {} {} {} {} {} {}".format(op,
                                                              self.objtype,
                                                              self.offsets.name,
                                                              self.bytes.name,
                                                              self.objtype,
                                                              other)
        else:
            raise ValueError("Strings: {} not supported between Strings and {}".format(op, type(other)))
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)
        
    def __eq__(self, other):
        return self.binop(other, "==")

    def __neq__(self, other):
        return self.binop(other, "!=")

    def __getitem__(self, key):
        if isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            if (key >= 0 and key < self.size):
                msg = "segmentedIndex {} {} {} {} {}".format('intIndex',
                                                             self.objtype,
                                                             self.offsets.name,
                                                             self.bytes.name,
                                                             key)
                repMsg = generic_msg(msg)
                fields = repMsg.split()
                # value = fields[2]
                return parse_single_value(' '.join(fields[1:]))
            else:
                raise IndexError("[int] {} is out of bounds with size {}".format(key,self.size))
        elif isinstance(key, slice):
            (start,stop,stride) = key.indices(self.size)
            if verbose: print(start,stop,stride)
            msg = "segmentedIndex {} {} {} {} {} {} {}".format('sliceIndex',
                                                               self.objtype,
                                                               self.offsets.name,
                                                               self.bytes.name,
                                                               start,
                                                               stop,
                                                               stride)
            repMsg = generic_msg(msg)
            offsets, values = repMsg.split('+')
            return Strings(offsets, values);
        elif isinstance(key, pdarray):
            kind, itemsize = translate_np_dtype(key.dtype)
            if kind not in ("bool", "int"):
                raise TypeError("unsupported pdarray index type {}".format(key.dtype))
            if kind == "bool" and self.size != key.size:
                raise ValueError("size mismatch {} {}".format(self.size,key.size))
            msg = "segmentedIndex {} {} {} {} {}".format('pdarrayIndex',
                                                         self.objtype,
                                                         self.offsets.name,
                                                         self.bytes.name,
                                                         other.name)
            repMsg = generic_msg(msg)
            offsets, values = repMsg.split('+')
            return Strings(offsets, values)
        else:
            return NotImplemented

    def group(self):
        msg = "segmentedGroup {} {} {}".format(self.objtype, self.offsets.name, self.bytes.name)
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)
