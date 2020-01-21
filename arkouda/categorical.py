from arkouda.strings import Strings
from arkouda.pdarrayclass import pdarray
from arkouda.groupbyclass import GroupBy
from arkouda.pdarraycreation import zeros
from arkouda.dtypes import int64
from arkouda.sorting import argsort

__all__ = ['Categorical']

class Categorical:
    BinOps = frozenset(["==", "!="])
    objtype = "category"
    self.permutation = None
    self.segments = None
    
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
            if not isinstance(args[1], pdarray) and not isinstance(args[1], Strings):
                raise TypeError("Index must be pdarray of int64 or Strings")
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

    def __iter__(self):
        inds = self.values.to_ndarray()
        for i in inds:
            yield self.index[i]

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
        if self.size != other.size:
            raise ValueError("Strings: size mismatch {} {}".format(self.size, other.size))
        if isinstance(other, Categorical):
            if self.index.name == other.index.name:
                return self.values.binop(other.values, op)
            else:
                raise NotImplementedError("Operations between Categoricals with different indices not yet implemented")
        else:
            raise NotImplementedError("Operations between Categorical and non-Categorical not yet implemented")

    def __eq__(self, other):
        return self.binop(other, "==")

    def __neq__(self, other):
        return self.binop(other, "!=")

    def __getitem__(self, key):
        if np.isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            return self.index[self.values[key]]
        else:
            vals = self.values[key]
            g = GroupBy(vals)
            idx = self.index[g.unique_keys]
            newvals = zeros(vals.size, int64)
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

    def argsort(self):
        if self.permutation is None:
            return argsort(self.values)
        else:
            return self.permutation

    def merge(self, other):
        if not isinstance(other, Categorical):
            raise TypeError("Categorical: can only merge with another Categorical")
        msg = "segmentedMerge {} {} {} {} {} {}".format(self.index.objtype,
                                                           self.index.offsets.name,
                                                           self.index.bytes.name,
                                                           other.index.objtype,
                                                           other.index.offsets.name,
                                                           other.index.bytes.name)
        repMsg = generic_msg(msg)
        perm_attrib, seg_attrib, val_attrib = repMsg.split('+')
        perm = create_pdarray(perm_attrib)
        index = Strings(seg_attrib, val_attrib)
    #     return create_pdarray(repMsg)
        
    # def merge(self, other):
    #     perm = self.argmerge(other)
    #     index = concatenate((self.index, other.index))[perm]
        values = zeros(self.size + other.size, int64)
        values[:self.size] = perm[self.values]
        values[self.size:] = perm[other.values + self.index.size]
        return Categorical(values, index)
