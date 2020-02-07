from arkouda.strings import Strings
from arkouda.pdarrayclass import pdarray
from arkouda.groupbyclass import GroupBy
from arkouda.pdarraycreation import zeros, zeros_like, arange
from arkouda.dtypes import int64, resolve_scalar_dtype
from arkouda.sorting import argsort
from arkouda.client import pdarrayIterThresh
from arkouda.pdarraysetops import unique, concatenate, in1d
import numpy as np

__all__ = ['Categorical']

class Categorical:
    BinOps = frozenset(["==", "!="])
    objtype = "category"
    permutation = None
    segments = None
    
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            items = args[0]
            g = GroupBy(items)
            self.index = g.unique_keys
            self.values = zeros(items.size, dtype=int64)
            self.values[g.permutation] = g.broadcast(arange(self.index.size))
            self.permutation = g.permutation
            self.segments = g.segments
        elif len(args) == 2:
            if not isinstance(args[0], pdarray) or args[0].dtype != int64:
                raise TypeError("Values must be pdarray of int64")
            if not isinstance(args[1], Strings):
                raise TypeError("Index must be Strings")
            self.values = args[0]
            self.index = args[1]
            if 'permutation' in kwargs:
                self.permutation = kwargs['permutation']
            if 'segments' in kwargs:
                self.segments = kwargs['segments']
        self.size = self.values.size
        self.nlevels = self.index.size
        self.ndim = self.values.ndim
        self.shape = self.values.shape

    def to_ndarray(self):
        idx = self.index.to_ndarray()
        valcodes = self.values.to_ndarray()
        return idx[valcodes]

    def __iter__(self):
        return iter(self.to_ndarray())
        
    def __len__(self):
        return self.shape[0]

    def __str__(self):
        if self.size <= pdarrayIterThresh:
            vals = ["'{}'".format(self[i]) for i in range(self.size)]
        else:
            vals = ["'{}'".format(self[i]) for i in range(3)]
            vals.append('... ')
            vals.extend([self[i] for i in range(self.size-3, self.size)])
        return "[{}]".format(', '.join(vals))

    def __repr__(self):
        return "array({})".format(self.__str__())

    def binop(self, other, op):
        if op not in self.BinOps:
            raise NotImplementedError("Categorical: unsupported operator: {}".format(op))
        if np.isscalar(other) and resolve_scalar_dtype(other) == "str":
            idxresult = self.index.binop(other, op)
            return idxresult[self.values]
        if self.size != other.size:
            raise ValueError("Categorical {}: size mismatch {} {}".format(op, self.size, other.size))
        if isinstance(other, Categorical):
            if self.index.name == other.index.name:
                return self.values.binop(other.values, op)
            else:
                raise NotImplementedError("Operations between Categoricals with different indices not yet implemented")
        else:
            raise NotImplementedError("Operations between Categorical and non-Categorical not yet implemented. Consider converting operands to Categorical.")

    def r_binop(self, other, op):
        return self.binop(other, op)

    def __eq__(self, other):
        return self.binop(other, "==")

    def __neq__(self, other):
        return self.binop(other, "!=")

    def __getitem__(self, key):
        if np.isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            return self.index[self.values[key]]
        else:
            return Categorical(self.values[key], self.index)

    def reset_index(self):
        g = GroupBy(self.values)
        idx = self.index[g.unique_keys]
        newvals = zeros(self.values.size, int64)
        newvals[g.permutation] = g.broadcast(arange(idx.size))
        return Categorical(newvals, idx, permutation=g.permutation, segments=g.segments)

    def contains(self, substr):
        indexcontains = self.index.contains(substr)
        return indexcontains[self.values]

    def startswith(self, substr):
        indexstartswith = self.index.startswith(substr)
        return indexstartswith[self.values]

    def endswith(self, substr):
        indexendswith = self.index.endswith(substr)
        return indexendswith[self.values]

    def in1d(self, test):
        indexisin = in1d(self.index, test)
        return indexisin[self.values]

    def unique(self):
        return Categorical(arange(self.index.size), self.index)

    def group(self):
        if self.permutation is None:
            return argsort(self.values)
        else:
            return self.permutation

    def argsort(self):
        idxperm = argsort(self.index)
        inverse = zeros_like(idxperm)
        inverse[idxperm] = arange(idxperm.size)
        newvals = inverse[self.values]
        return argsort(newvals)

    def sort(self):
        idxperm = argsort(self.index)
        inverse = zeros_like(idxperm)
        inverse[idxperm] = arange(idxperm.size)
        newvals = inverse[self.values]
        return Categorical(newvals, self.index[idxperm])
            
    def merge(self, others):
        if isinstance(others, Categorical):
            others = [others]
        elif len(others) < 1:
            return self
        sameindex = True
        for c in others:
            if not isinstance(c, Categorical):
                raise TypeError("Categorical: can only merge/concatenate with other Categoricals")
            if (self.index.size != other.index.size) or not (self.index == other.index).all():
                sameindex = False
        if sameindex:
            newvals = concatenate([self.values] + [o.values for o in others])
            return Categorical(newvals, self.index)
        else:
            g = ak.GroupBy(concatenate([self.index] + [o.index for o in others]))
            newidx = g.unique_keys
            wherediditgo = zeros(newidx.size, dtype=int64)
            wherediditgo[g.permutation] = arange(newidx.size)
            idxsizes = np.array([self.index.size] + [o.index.size for o in others])
            idxoffsets = np.cumsum(idxsizes) - idxsizes
            oldvals = concatenate([c.values + off for c, off in zip([self.values] + [o.values for o in others], idxoffsets)])
            newvals = wherediditgo[oldvals]
            return Categorical(newvals, newidx)
    #     msg = "segmentedMerge {} {} {} {} {} {}".format(self.index.objtype,
    #                                                        self.index.offsets.name,
    #                                                        self.index.bytes.name,
    #                                                        other.index.objtype,
    #                                                        other.index.offsets.name,
    #                                                        other.index.bytes.name)
    #     repMsg = generic_msg(msg)
    #     perm_attrib, seg_attrib, val_attrib = repMsg.split('+')
    #     perm = create_pdarray(perm_attrib)
    #     index = Strings(seg_attrib, val_attrib)
    # #     return create_pdarray(repMsg)
        
    # # def merge(self, other):
    # #     perm = self.argmerge(other)
    # #     index = concatenate((self.index, other.index))[perm]
    #     values = zeros(self.size + other.size, int64)
    #     values[:self.size] = perm[self.values]
    #     values[self.size:] = perm[other.values + self.index.size]
    #     return Categorical(values, index)
