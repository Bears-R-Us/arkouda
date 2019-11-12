from arkouda.pdarrayclass import pdarray, create_pdarray

__all__ = ['Strings']

class Strings:
    BinOps = frozenset(["==", "!="])
    
    def __init__(self, offset_attrib, bytes_attrib):
        self.offsets = create_pdarray(offset_attrib)
        self.bytes = create_pdarray(bytes_attrib)
        self.size = self.offsets.size
        self.nbytes = self.bytes.size
        self.dtype = np.dtype(np.str_)
        self.ndim = self.offsets.ndim
        self.shape = self.offsets.shape

    def __len__(self):
        return self.shape[0]

    def binop(self, other):
        if op not in self.BinOps:
            raise ValueError("Strings: unsupported operator: {}".format(op))
        if isinstance(other, Strings):
            if self.size != other.size:
                raise ValueError("Strings: size mismatch {} {}".format(self.size, other.size))
            msg = "segBinopvv {} {} {} {} {} {} {}".format(op,
                                                           self.dtype.name,
                                                           self.offsets.name,
                                                           self.bytes.name,
                                                           other.dtype.name,
                                                           other.offsets.name,
                                                           other.bytes.name)
        elif hasattr(other, '__len__'):
            return NotImplemented
        elif resolve_scalar_dtype(other) == 'str':
            msg = "segBinopvs {} {} {} {} {} {}".format(op,
                                                        self.dtype.name,
                                                        self.offsets.name,
                                                        self.bytes.name,
                                                        self.dtype.name,
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
        if np.isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            if (key >= 0 and key < self.size):
                msg = "segmentedIndex {} {} {} {} {}".format('intIndex',
                                                             self.dtype.name,
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
                                                               self.dtype.name,
                                                               self.offsets.name,
                                                               self.bytes.name,
                                                               start,
                                                               stop,
                                                               stride)
            repMsg = generic_msg(msg)
            return create_pdarray(repMsg);
        elif isinstance(key, pdarray):
            kind, itemsize = translate_np_dtype(key.dtype)
            if kind not in ("bool", "int"):
                raise TypeError("unsupported pdarray index type {}".format(key.dtype))
            if kind == "bool" and self.size != key.size:
                raise ValueError("size mismatch {} {}".format(self.size,key.size))
            msg = "segmentedIndex {} {} {} {} {}".format('pdarrayIndex',
                                                      self.dtype.name,
                                                      self.offsets.name,
                                                      self.bytes.name,
                                                      other.name)
            repMsg = generic_msg(msg)
            return create_pdarray(repMsg)
        else:
            return NotImplemented

    def group(self):
        msg = "segGroup {} {} {}".format(self.dtype.name, self.offsets.name, self.bytes.name)
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)
