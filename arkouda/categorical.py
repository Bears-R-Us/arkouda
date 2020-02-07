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
    """
    Represents an array of values belonging to named categories. Converting a Strings
    object to Categorical often saves memory and speeds up operations, especially
    if there are many repeated values, at the cost of some one-time work in initialization.

    Parameters
    ----------
    values : Strings
        String values to convert to categories 

    Attributes
    ----------
    categories : Strings
        The set of category labels (determined automatically)
    codes : pdarray, int64
        The category indices of the values or -1 for N/A
    permutation : pdarray, int64
        The permutation that groups the values in the same order as categories
    segments : pdarray, int64
        When values are grouped, the starting offset of each group
    size : int
        The number of items in the array
    nlevels : int
        The number of distinct categories
    ndim : int
        The rank of the array (currently only rank 1 arrays supported)
    shape : tuple
        The sizes of each dimension of the array
    """
    BinOps = frozenset(["==", "!="])
    objtype = "category"
    permutation = None
    segments = None
    
    def __init__(self, values, **kwargs):
        if 'codes' in kwargs and 'categories' in kwargs:
            # This initialization is called by Categorical.from_codes()
            # The values arg is ignored
            self.codes = kwargs['codes']
            self.categories = kwargs['categories']            
            if 'permutation' in kwargs:
                self.permutation = kwargs['permutation']
            if 'segments' in kwargs:
                self.segments = kwargs['segments']
        else:
            # Typical initialization, called with values
            if not isinstance(values, Strings):
                raise ValueError("Categorical: inputs other than Strings not yet supported")
            g = GroupBy(values)
            self.categories = g.unique_keys
            self.codes = zeros(values.size, dtype=int64)
            self.codes[g.permutation] = g.broadcast(arange(self.categories.size))
            self.permutation = g.permutation
            self.segments = g.segments
        # Always set these values
        self.size = self.codes.size
        self.nlevels = self.categories.size
        self.ndim = self.codes.ndim
        self.shape = self.codes.shape

    @classmethod
    def from_codes(cls, codes, categories, permutation=None, segments=None):
        """
        Make a Categorical from codes and categories arrays. If codes and categories
        have already been precomputed, this constructor saves time. If not, please
        use the normal constructor.

        Parameters
        ----------
        codes : pdarray, int64
            Category indices of each value
        categories : String
            Unique category labels
        """
        if not isinstance(codes, pdarray) or codes.dtype != int64:
            raise TypeError("Codes must be pdarray of int64")
        if not isinstance(categories, Strings):
            raise TypeError("Categories must be Strings")
        return cls(None, codes=codes, categories=categories, permutation=permutation, segments=segments)

    def to_ndarray(self):
        """
        Convert the array to a np.ndarray, transferring array data from the
        arkouda server to Python. This conversion discards category information
        and produces an ndarray of strings. If the array exceeds a builtin size limit,
        a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            A numpy ndarray of strings corresponding to the values in this array

        Notes
        -----
        The number of bytes in the array cannot exceed ``arkouda.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.maxTransferBytes to a larger
        value, but proceed with caution.
        """
        idx = self.categories.to_ndarray()
        valcodes = self.codes.to_ndarray()
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
            idxresult = self.categories.binop(other, op)
            return idxresult[self.codes]
        if self.size != other.size:
            raise ValueError("Categorical {}: size mismatch {} {}".format(op, self.size, other.size))
        if isinstance(other, Categorical):
            if self.categories.name == other.categories.name:
                return self.codes.binop(other.codes, op)
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
            return self.categories[self.codes[key]]
        else:
            return Categorical.from_codes(self.codes[key], self.categories)

    def reset_categories(self):
        """
        Recompute the category labels, discarding any unused labels. This method
        is often useful after slicing or indexing a Categorical array, when the
        resulting array only contains a subset of the original categories. In
        this case, eliminating unused categories can speed up other operations.
        """
        g = GroupBy(self.codes)
        idx = self.categories[g.unique_keys]
        newvals = zeros(self.codes.size, int64)
        newvals[g.permutation] = g.broadcast(arange(idx.size))
        return Categorical.from_codes(newvals, idx, permutation=g.permutation, segments=g.segments)

    def contains(self, substr):
        """
        Check whether each element contains the given substring.

        Parameters
        ----------
        substr : str
            The substring to search for

        Returns
        -------
        pdarray, bool
            True for elements that contain substr, False otherwise

        Notes
        -----
        This method can be significantly faster than the corresponding method
        on Strings objects, because it searches the unique category labels
        instead of the full array.

        See Also
        --------
        Categorical.startswith, Categorical.endswith
        """
        categoriescontains = self.categories.contains(substr)
        return categoriescontains[self.codes]

    def startswith(self, substr):
        """
        Check whether each element starts with the given substring.

        Parameters
        ----------
        substr : str
            The substring to search for

        Returns
        -------
        pdarray, bool
            True for elements that contain substr, False otherwise

        Notes
        -----
        This method can be significantly faster than the corresponding method
        on Strings objects, because it searches the unique category labels
        instead of the full array.

        See Also
        --------
        Categorical.contains, Categorical.endswith
        """
        categoriesstartswith = self.categories.startswith(substr)
        return categoriesstartswith[self.codes]

    def endswith(self, substr):
        """
        Check whether each element ends with the given substring.

        Parameters
        ----------
        substr : str
            The substring to search for

        Returns
        -------
        pdarray, bool
            True for elements that contain substr, False otherwise

        Notes
        -----
        This method can be significantly faster than the corresponding method
        on Strings objects, because it searches the unique category labels
        instead of the full array.

        See Also
        --------
        Categorical.startswith, Categorical.contains
        """
        categoriesendswith = self.categories.endswith(substr)
        return categoriesendswith[self.codes]

    def in1d(self, test):
        __doc__ = in1d.__doc__
        categoriesisin = in1d(self.categories, test)
        return categoriesisin[self.codes]

    def unique(self):
        __doc__ = unique.__doc__
        return Categorical.from_codes(arange(self.categories.size), self.categories)

    def group(self):
        """
        Return the permutation that groups the array, placing equivalent
        categories together. All instances of the same category are guaranteed to lie
        in one contiguous block of the permuted array, but the blocks are not
        necessarily ordered.

        Returns
        -------
        pdarray
            The permutation that groups the array by value

        See Also
        --------
        GroupBy, unique

        Notes
        -----
        This method is faster than the corresponding Strings method. If the Categorical
        was created from a Strings object, then this function simply returns the
        cached permutation. Even if the Categorical was created using from_codes(),
        this function will be faster than Strings.group() because it sorts dense
        integer values, rather than 128-bit hash values.
        """        
        if self.permutation is None:
            return argsort(self.codes)
        else:
            return self.permutation

    def argsort(self):
        __doc__ = argsort.__doc__
        idxperm = argsort(self.categories)
        inverse = zeros_like(idxperm)
        inverse[idxperm] = arange(idxperm.size)
        newvals = inverse[self.codes]
        return argsort(newvals)

    def sort(self):
        __doc__ = sort.__doc__
        idxperm = argsort(self.categories)
        inverse = zeros_like(idxperm)
        inverse[idxperm] = arange(idxperm.size)
        newvals = inverse[self.codes]
        return Categorical.from_codes(newvals, self.categories[idxperm])
            
    def merge(self, others):
        """
        Merge this Categorical with other Categoricals, concatenating the arrays and
        synchronizing the categories.

        Parameters
        ----------
        others : list of Categorical
            The Categorical arrays to concatenate and merge with this one

        Notes
        -----
        This operation can be expensive -- slower than concatenating Strings arrays.
        """
        if isinstance(others, Categorical):
            others = [others]
        elif len(others) < 1:
            return self
        samecategories = True
        for c in others:
            if not isinstance(c, Categorical):
                raise TypeError("Categorical: can only merge/concatenate with other Categoricals")
            if (self.categories.size != other.categories.size) or not (self.categories == other.categories).all():
                samecategories = False
        if samecategories:
            newvals = concatenate([self.codes] + [o.codes for o in others])
            return Categorical.from_codes(newvals, self.categories)
        else:
            g = ak.GroupBy(concatenate([self.categories] + [o.categories for o in others]))
            newidx = g.unique_keys
            wherediditgo = zeros(newidx.size, dtype=int64)
            wherediditgo[g.permutation] = arange(newidx.size)
            idxsizes = np.array([self.categories.size] + [o.categories.size for o in others])
            idxoffsets = np.cumsum(idxsizes) - idxsizes
            oldvals = concatenate([c.codes + off for c, off in zip([self.codes] + [o.codes for o in others], idxoffsets)])
            newvals = wherediditgo[oldvals]
            return Categorical.from_codes(newvals, newidx)
    #     msg = "segmentedMerge {} {} {} {} {} {}".format(self.categories.objtype,
    #                                                        self.categories.offsets.name,
    #                                                        self.categories.bytes.name,
    #                                                        other.categories.objtype,
    #                                                        other.categories.offsets.name,
    #                                                        other.categories.bytes.name)
    #     repMsg = generic_msg(msg)
    #     perm_attrib, seg_attrib, val_attrib = repMsg.split('+')
    #     perm = create_pdarray(perm_attrib)
    #     categories = Strings(seg_attrib, val_attrib)
    # #     return create_pdarray(repMsg)
        
    # # def merge(self, other):
    # #     perm = self.argmerge(other)
    # #     categories = concatenate((self.categories, other.categories))[perm]
    #     codes = zeros(self.size + other.size, int64)
    #     codes[:self.size] = perm[self.codes]
    #     codes[self.size:] = perm[other.codes + self.categories.size]
    #     return Categorical(codes, categories)
