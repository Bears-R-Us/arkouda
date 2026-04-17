from __future__ import annotations

import json

from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from arkouda.core.logger import get_arkouda_logger
from arkouda.numpy.dtypes import bool_ as akbool
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import int_scalars, is_supported_int, str_
from arkouda.numpy.dtypes import uint64 as akuint64
from arkouda.numpy.pdarrayclass import (
    RegistrationError,
    create_pdarray,
    is_sorted,
    pdarray,
)
from arkouda.numpy.pdarraycreation import arange, array, ones, zeros
from arkouda.numpy.pdarraysetops import concatenate
from arkouda.numpy.strings import Strings
from arkouda.pandas.groupbyclass import GroupBy, broadcast
from arkouda.pandas.join import gen_ranges


__all__ = [
    "SegArray",
]


SEG_SUFFIX = "_segments"
VAL_SUFFIX = "_values"
LEN_SUFFIX = "_lengths"


def _aggregator(func):
    aggdoc = """
        Aggregate values over each sub-array.

        Parameters
        ----------
        x : pdarray
        The values to aggregate. By default, the values of the sub-arrays
        themselves are used, but the user may supply an array of values
        corresponding to the flattened values of all sub-arrays.

        Returns
        -------
        pdarray
        Array of one aggregated value per sub-array.
        """

    def update_doc():
        func.__doc__ = aggdoc
        return func

    return update_doc


class SegArray:
    objType = "SegArray"

    def __init__(self, segments, values, lengths=None, grouping=None):
        self.logger = get_arkouda_logger(name=__class__.__name__)
        self.registered_name: Optional[str] = None

        # validate inputs
        if not isinstance(segments, pdarray) or segments.dtype != akint64:
            raise TypeError("Segments must be int64 pdarray")
        if not isinstance(values, pdarray) and not isinstance(values, Strings):
            raise TypeError("Values must be a pdarray or Strings.")
        if not is_sorted(segments):
            raise ValueError("Segments must be unique and in sorted order")
        if segments.size > 0:
            if segments[0] != 0:
                raise ValueError("Segments must start at zero.")
        elif values.size > 0:
            raise ValueError("Cannot have non-empty values with empty segments")

        # references to supporting pdarrays
        self.values = values
        self.segments = segments
        self.size = segments.size
        self.valsize = values.size
        self.dtype = values.dtype

        if lengths is None:
            self.lengths = self._get_lengths()
        else:
            self.lengths = lengths

        self._non_empty = self.lengths > 0
        self._non_empty_count = self._non_empty.sum()

        # grouping object computation. (This will need to be moved to the server)
        # GroupBy computation left here because of lack of server obj. May need to move in Future
        if grouping is None:
            if self.size == 0 or self._non_empty_count == 0:
                self._grouping = GroupBy(zeros(0, dtype=akint64))
            else:
                # Treat each sub-array as a group, for grouped aggregations
                self._grouping = GroupBy(
                    broadcast(self.segments[self.non_empty], arange(self._non_empty_count), self.valsize)
                )
        else:
            self._grouping = grouping

    @classmethod
    def from_return_msg(cls, rep_msg) -> SegArray:
        # parse return json
        eles = json.loads(rep_msg)

        # parse the create for the values pdarray
        values = (
            Strings.from_return_msg(eles["values"])
            if eles["values"].split()[2] == "str"
            else create_pdarray(eles["values"])
        )
        segments = create_pdarray(eles["segments"])
        lengths = create_pdarray(eles["lengths"]) if "lengths" in eles else None
        return cls(segments, values, lengths=lengths)

    @classmethod
    def from_multi_array(cls, m):
        """
        Construct a SegArray from a list of columns. This essentially transposes the input,
        resulting in an array of rows.

        Parameters
        ----------
        m : list of pdarray or Strings
            List of columns, the rows of which will form the sub-arrays of the output

        Returns
        -------
        SegArray
            Array of rows of input
        """
        if isinstance(m, pdarray):
            return cls(arange(m.size), m)
        else:
            sizes = np.array([mi.size for mi in m])
            dtypes = {mi.dtype for mi in m}
            if len(dtypes) != 1:
                raise ValueError("All values must have same dtype")
            n = len(m)
            offsets = np.cumsum(sizes) - sizes
            newvals = zeros(sum(sizes), dtype=dtypes.pop())
            for j in range(n):
                newvals[offsets[j] : (offsets[j] + sizes[j])] = m[j]
            return cls(array(offsets), newvals)

    @property
    def non_empty(self):
        from arkouda.core.infoclass import list_symbol_table

        if self._non_empty.name not in list_symbol_table():
            self._non_empty = self.lengths > 0
            self._non_empty_count = self._non_empty.sum()
        return self._non_empty

    @property
    def grouping(self):
        if self._grouping is not None:
            return self._grouping

        if self.size == 0 or self._non_empty_count == 0:
            self._grouping = GroupBy(zeros(0, dtype=akint64))
        else:
            # Treat each sub-array as a group, for grouped aggregations
            self._grouping = GroupBy(
                broadcast(self.segments[self.non_empty], arange(self._non_empty_count), self.valsize)
            )

    @property
    def nbytes(self):
        """
        The size of the segarray in bytes.

        Returns
        -------
        int
            The size of the segarray in bytes.

        """
        return self.values.nbytes

    def _get_lengths(self):
        if self.size == 0:
            return zeros(0, dtype=akint64)
        elif self.size == 1:
            return array([self.valsize])
        else:
            return concatenate((self.segments[1:], array([self.valsize]))) - self.segments

    def __getitem__(self, i):
        if is_supported_int(i):
            start = self.segments[i]
            end = self.segments[i] + self.lengths[i]
            return self.values[start:end]
        elif (isinstance(i, pdarray) and i.dtype in [akint64, akuint64, akbool]) or isinstance(i, slice):
            starts = self.segments[i]
            ends = starts + self.lengths[i]
            newsegs, inds, lengths = gen_ranges(starts, ends, return_lengths=True)
            return SegArray(newsegs, self.values[inds], lengths)
        else:
            raise TypeError(f"Invalid index type: {type(i)}")

    @classmethod
    def concat(cls, x, axis=0, ordered=True):
        """
        Concatenate a sequence of SegArrays.

        Parameters
        ----------
        x : sequence of SegArray
            The SegArrays to concatenate
        axis : 0 or 1
            Select vertical (0) or horizontal (1) concatenation. If axis=1, all
            SegArrays must have same size.
        ordered : bool
            Must be True. This option is present for compatibility only, because unordered
            concatenation is not yet supported.

        Returns
        -------
        SegArray
            The input arrays joined into one SegArray
        """
        from arkouda.numpy import cumsum

        if not ordered:
            raise ValueError("Unordered concatenation not yet supported on SegArray; use ordered=True.")
        if len(x) == 0:
            raise ValueError("Empty sequence passed to concat")
        for xi in x:
            if not isinstance(xi, cls):
                return NotImplemented
        if len({xi.dtype for xi in x}) != 1:
            raise ValueError("SegArrays must all have same dtype to concatenate")
        if axis == 0:
            ctr = 0
            segs = []
            vals = []
            for xi in x:
                # Segment offsets need to be raised by length of previous values
                segs.append(xi.segments + ctr)
                ctr += xi.valsize
                # Values can just be concatenated
                vals.append(xi.values)
            return cls(concatenate(segs), concatenate(vals))
        elif axis == 1:
            sizes = {xi.size for xi in x}
            if len(sizes) != 1:
                raise ValueError("SegArrays must all have same size to concatenate with axis=1")
            if sizes.pop() == 0:
                return x[0]
            dt = list(x)[0].dtype
            newlens = sum(xi.lengths for xi in x)
            newsegs = cumsum(newlens) - newlens
            # Ignore sub-arrays that are empty in all arrays
            nonzero = concatenate((newsegs[:-1] < newsegs[1:], array([True])))
            nzsegs = newsegs[nonzero]
            newvals = zeros(newlens.sum(), dtype=dt)
            for xi in x:
                # Set up fromself for a scan, so that it steps up at the start of a segment
                # from the current array, and steps back down at the end
                fromself = zeros(newvals.size + 1, dtype=akint64)
                fromself[nzsegs] += 1
                nzlens = xi.lengths[nonzero]
                fromself[nzsegs + nzlens] -= 1
                fromself = cumsum(fromself[:-1]) == 1
                newvals[fromself] = xi.values
                nzsegs += nzlens
            return cls(newsegs, newvals)
        else:
            raise ValueError(
                "Supported values for axis are 0 (vertical concat) or 1 (horizontal concat)"
            )

    def copy(self):
        """Return a deep copy."""
        return SegArray(self.segments[:], self.values[:])

    def __eq__(self, other):
        if not isinstance(other, SegArray):
            return NotImplemented
        if self.size != other.size:
            raise ValueError("Segarrays must have same size to compare")
        eq = zeros(self.size, dtype=akbool)
        leneq = self.lengths == other.lengths
        if leneq.sum() > 0:
            selfcmp = self[leneq]
            othercmp = other[leneq]
            intersection = selfcmp.all(selfcmp.values == othercmp.values)
            eq[leneq & (self.lengths != 0)] = intersection
            eq[leneq & (self.lengths == 0)] = True
        return eq

    def __len__(self) -> int:
        return self.size

    def __str__(self):
        if self.size <= 6:
            rows = list(range(self.size))
        else:
            rows = [0, 1, 2, None, self.size - 3, self.size - 2, self.size - 1]
        outlines = ["SegArray(["]
        for r in rows:
            if r is None:
                outlines.append("...")
            else:
                outlines.append(str(self[r]))
        outlines.append("])")
        return "\n".join(outlines)

    def __repr__(self):
        return self.__str__()

    def get_suffixes(self, n, return_origins=True, proper=True):
        """
        Return the n-long suffix of each sub-array, where possible.

        Parameters
        ----------
        n : int
            Length of suffix
        return_origins : bool
            If True, return a logical index indicating which sub-arrays
            were long enough to return an n-suffix
        proper : bool
            If True, only return proper suffixes, i.e. from sub-arrays
            that are at least n+1 long. If False, allow the entire
            sub-array to be returned as a suffix.

        Returns
        -------
        List of pdarray, pdarray|bool
            suffixes : list of pdarray
                An n-long list of pdarrays, essentially a table where each row is an n-suffix.
                The number of rows is the number of True values in the returned mask.
            origin_indices : pdarray, bool
                Boolean array that is True where the sub-array was long enough to return
                an n-suffix, False otherwise.
        """
        if proper:
            longenough = self.lengths > n
        else:
            longenough = self.lengths >= n
        suffixes = []
        for i in range(n):
            ind = (self.segments + self.lengths - (n - i))[longenough]
            suffixes.append(self.values[ind])
        if return_origins:
            return suffixes, longenough
        else:
            return suffixes

    def get_prefixes(self, n, return_origins=True, proper=True):
        """
        Return all sub-array prefixes of length n (for sub-arrays that are at least n+1 long).

        Parameters
        ----------
        n : int
            Length of suffix
        return_origins : bool
            If True, return a logical index indicating which sub-arrays
            were long enough to return an n-prefix
        proper : bool
            If True, only return proper prefixes, i.e. from sub-arrays
            that are at least n+1 long. If False, allow the entire
            sub-array to be returned as a prefix.

        Returns
        -------
        List of pdarray, pdarray|bool
            prefixes : list of pdarray
                An n-long list of pdarrays, essentially a table where each row is an n-prefix.
                The number of rows is the number of True values in the returned mask.
            origin_indices : pdarray, bool
                Boolean array that is True where the sub-array was long enough to return
                an n-suffix, False otherwise.
        """
        if proper:
            longenough = self.lengths > n
        else:
            longenough = self.lengths >= n
        prefixes = []
        for i in range(n):
            ind = (self.segments + i)[longenough]
            prefixes.append(self.values[ind])
        if return_origins:
            return prefixes, longenough
        else:
            return prefixes

    def get_ngrams(self, n, return_origins=True):
        """
        Return all n-grams from all sub-arrays.

        Parameters
        ----------
        n : int
            Length of n-gram
        return_origins : bool
            If True, return an int64 array indicating which sub-array
            each returned n-gram came from.

        Returns
        -------
        pdarray, pdarray|int
            ngrams : list of pdarray
                An n-long list of pdarrays, essentially a table where each row is an n-gram.
            origin_indices : pdarray, int
                The index of the sub-array from which the corresponding n-gram originated
        """
        if n > self.lengths.max():
            raise ValueError("n must be <= the maximum length of the sub-arrays")

        ngrams = []
        notsegstart = ones(self.valsize, dtype=akbool)
        notsegstart[self.segments[self.non_empty]] = False
        valid = ones(self.valsize - n + 1, dtype=akbool)
        for i in range(n):
            end = self.valsize - n + i + 1
            ngrams.append(self.values[i:end])
            if i > 0:
                valid &= notsegstart[i:end]
        ngrams = [char[valid] for char in ngrams]
        if return_origins:
            # set the proper indexes for broadcasting. Needed to alot for empty segments
            seg_idx = arange(self.size)[self.non_empty]
            origin_indices = self.grouping.broadcast(seg_idx, permute=True)[: valid.size][valid]
            return ngrams, origin_indices
        else:
            return ngrams

    def _normalize_index(self, j):
        if not is_supported_int(j):
            raise TypeError(f"index must be integer, not {type(j)}")
        if j >= 0:
            longenough = self.lengths > j
        else:
            j = self.lengths + j
            longenough = j >= 0
        return longenough, j

    def get_jth(self, j, return_origins=True, compressed=False, default=0):
        """
        Select the j-th element of each sub-array, where possible.

        Parameters
        ----------
        j : int
            The index of the value to get from each sub-array. If j is negative,
            it counts backwards from the end of each sub-array.
        return_origins : bool
            If True, return a logical index indicating where j is in bounds
        compressed : bool
            If False, return array is same size as self, with default value
            where j is out of bounds. If True, the return array only contains
            values where j is in bounds.
        default : scalar
            When compressed=False, the value to return when j is out of bounds
            for the sub-array

        Returns
        -------
        pdarray, pdarray|bool
            val : pdarray
                compressed=False: The j-th value of each sub-array where j is in
                bounds and the default value where j is out of bounds.
                compressed=True: The j-th values of only the sub-arrays where j is
                in bounds
            origin_indices : pdarray, bool
                A Boolean array that is True where j is in bounds for the sub-array.

        Notes
        -----
        If values are Strings, only the compressed format is supported.
        """
        longenough, newj = self._normalize_index(j)
        ind = (self.segments + newj)[longenough]
        if compressed or self.dtype == str_:  # Strings not supported by uncompressed version
            res = self.values[ind]
        else:
            res = zeros(self.size, dtype=self.dtype)
            res.fill(default)
            res[longenough] = self.values[ind]
        if return_origins:
            return res, longenough
        else:
            return res

    def set_jth(self, i, j, v):
        """
        Set the j-th element of each sub-array in a subset.

        Parameters
        ----------
        i : pdarray, int
            Indices of sub-arrays to set j-th element
        j : int
            Index of value to set in each sub-array. If j is negative, it counts
            backwards from the end of the sub-array.
        v : pdarray or scalar
            The value(s) to set. If v is a pdarray, it must have same length as i.

        Raises
        ------
        ValueError
            If j is out of bounds in any of the sub-arrays specified by i.
        """
        if self.dtype == str_:
            raise TypeError("String elements are immutable")
        longenough, newj = self._normalize_index(j)
        if not longenough[i].all():
            raise ValueError("Not all (i, j) in bounds")
        ind = (self.segments + newj)[i]
        self.values[ind] = v

    def get_length_n(self, n, return_origins=True):
        """
        Return all sub-arrays of length n, as a list of columns.

        Parameters
        ----------
        n : int
            Length of sub-arrays to select
        return_origins : bool
            Return a logical index indicating which sub-arrays are length n

        Returns
        -------
        List of pdarray, pdarray|bool
            columns : list of pdarray
                An n-long list of pdarray, where each row is one of the n-long
                sub-arrays from the SegArray. The number of rows is the number of
                True values in the returned mask.
            origin_indices : pdarray, bool
                Array of bool for each element of the SegArray, True where sub-array
                has length n.
        """
        mask = self.lengths == n
        elem = []
        for i in range(n):
            ind = (self.segments + self.lengths - (n - i))[mask]
            elem.append(self.values[ind])
        if return_origins:
            return elem, mask
        else:
            return elem

    def append(self, other, axis=0):
        """
        Append other to self, either vertically (axis=0, length of resulting SegArray
        increases), or horizontally (axis=1, each sub-array of other appends to the
        corresponding sub-array of self).

        Parameters
        ----------
        other : SegArray
            Array of sub-arrays to append
        axis : 0 or 1
            Whether to append vertically (0) or horizontally (1). If axis=1, other
            must be same size as self.

        Returns
        -------
        SegArray
            axis=0: New SegArray containing all sub-arrays
            axis=1: New SegArray of same length, with pairs of sub-arrays concatenated
        """
        if not isinstance(other, SegArray):
            return NotImplemented
        if self.dtype != other.dtype:
            raise TypeError("SegArrays must have same value type to append")
        return self.__class__.concat((self, other), axis=axis)

    def append_single(self, x, prepend=False):
        """
        Append a single value to each sub-array.

        Parameters
        ----------
        x : pdarray or scalar
            Single value to append to each sub-array

        Returns
        -------
        SegArray
            Copy of original SegArray with values from x appended to each sub-array
        """
        from arkouda.numpy import cumsum

        if self.dtype == str_:
            raise TypeError("String elements are immutable and cannot accept a single value")
        if hasattr(x, "size"):
            if x.size != self.size:
                raise ValueError("Argument must be scalar or same size as SegArray")
            if not isinstance(x, type(self.values)) or x.dtype != self.dtype:
                raise TypeError("Argument type must match value type of SegArray")
        newlens = self.lengths + 1
        newsegs = cumsum(newlens) - newlens
        newvals = zeros(newlens.sum(), dtype=self.dtype)
        if prepend:
            lastscatter = newsegs
        else:
            lastscatter = newsegs + newlens - 1
        newvals[lastscatter] = x
        origscatter = arange(self.valsize) + self.grouping.broadcast(
            arange(self.size)[self.non_empty], permute=True
        )
        if prepend:
            origscatter += 1
        newvals[origscatter] = self.values
        return SegArray(newsegs, newvals)

    def prepend_single(self, x):
        return self.append_single(x, prepend=True)

    def remove_repeats(self, return_multiplicity=False):
        """
        Condense sequences of repeated values within a sub-array to a single value.

        Parameters
        ----------
        return_multiplicity : bool
            If True, also return the number of times each value was repeated.

        Returns
        -------
        Segarray, Segarray
            norepeats : SegArray
                Sub-arrays with runs of repeated values replaced with single value
            multiplicity : SegArray
                If return_multiplicity=True, this array contains the number of times
                each value in the returned SegArray was repeated in the original SegArray.
        """
        from arkouda.numpy import cumsum

        isrepeat = zeros(self.values.size, dtype=akbool)
        isrepeat[1:] = self.values[:-1] == self.values[1:]
        isrepeat[self.segments[self.non_empty]] = False
        truepaths = self.values[~isrepeat]
        nhops = self.grouping.sum(~isrepeat)[1]
        # Correct segments to properly assign empty lists - prevents dropping empty segments
        lens = self.lengths[:]
        lens[self.non_empty] = nhops
        truesegs = cumsum(lens) - lens

        norepeats = SegArray(truesegs, truepaths)
        if return_multiplicity:
            truehopinds = arange(self.valsize)[~isrepeat]
            multiplicity = zeros(truepaths.size, dtype=akint64)
            multiplicity[:-1] = truehopinds[1:] - truehopinds[:-1]
            multiplicity[-1] = self.valsize - truehopinds[-1]
            return norepeats, SegArray(truesegs, multiplicity)
        else:
            return norepeats

    def to_ndarray(self):
        """
        Convert the array into a numpy.ndarray containing sub-arrays.

        Returns
        -------
        np.ndarray
            A numpy ndarray with the same sub-arrays (also numpy.ndarray) as this array

        See Also
        --------
        array()
        tolist()

        Examples
        --------
        >>> import arkouda as ak
        >>> segarr = ak.SegArray(ak.array([0, 4, 7]), ak.arange(12))
        >>> segarr.to_ndarray()
        array([array([0, 1, 2, 3]), array([4, 5, 6]), array([ 7,  8,  9, 10, 11])],
          dtype=object)
        >>> type(segarr.to_ndarray())
        <class 'numpy.ndarray'>
        """
        ndvals = self.values.to_ndarray()
        ndsegs = self.segments.to_ndarray()
        arr = [ndvals[start:end] for start, end in zip(ndsegs, ndsegs[1:])]
        if self.size > 0:
            arr.append(ndvals[ndsegs[-1] :])
        return np.array(arr, dtype=object)

    def tolist(self):
        """
        Convert the segarray into a list containing sub-arrays.

        Returns
        -------
        list
            A list with the same sub-arrays (also list) as this segarray

        See Also
        --------
        to_ndarray()

        Examples
        --------
        >>> import arkouda as ak
        >>> segarr = ak.SegArray(ak.array([0, 4, 7]), ak.arange(12))
        >>> segarr.tolist()
        [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11]]
        >>> type(segarr.tolist())
        <class 'list'>
        """
        return [arr.tolist() for arr in self.to_ndarray()]

    def sum(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.sum(x)[1]

    def prod(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.prod(x)[1]

    def min(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.min(x)[1]

    def max(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.max(x)[1]

    def argmin(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.argmin(x)[1]

    def argmax(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.argmax(x)[1]

    def any(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.any(x)[1]

    def all(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.all(x)[1]

    def OR(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.OR(x)[1]

    def AND(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.AND(x)[1]

    def XOR(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.XOR(x)[1]

    def nunique(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.nunique(x)[1]

    def mean(self, x=None):
        if x is None:
            x = self.values
        return self.grouping.mean(x)[1]

    def aggregate(self, op, x=None):
        if x is None:
            x = self.values
        return self.grouping.aggregate(x, op)

    def unique(self, x=None) -> SegArray:
        """
        Return sub-arrays of unique values.

        Parameters
        ----------
        x : pdarray
            The values to unique, per group. By default, the values of this
            SegArray's sub-arrays.

        Returns
        -------
        SegArray
            Same number of sub-arrays as original SegArray, but elements in sub-array
            are unique and in sorted order.
        """
        if x is None:
            x = self.values
        keyidx = self.grouping.broadcast(arange(self.size), permute=True)
        ukey, uval = GroupBy([keyidx, x]).unique_keys
        g = GroupBy(ukey, assume_sorted=True)
        _, lengths = g.size()
        return SegArray(g.segments, uval, grouping=g, lengths=lengths)

    def hash(self) -> Tuple[pdarray, pdarray]:
        """
        Compute a 128-bit hash of each segment.

        Returns
        -------
        Tuple[pdarray,pdarray]
            A tuple of two int64 pdarrays. The ith hash value is the concatenation
            of the ith values from each array.
        """
        from arkouda.core.client import generic_msg

        rep_msg = generic_msg(
            cmd="segmentedHash",
            args={
                "objType": self.objType,
                "values": self.values,
                "segments": self.segments,
                "valObjType": self.values.objType,
            },
        )
        h1, h2 = rep_msg.split("+")
        return create_pdarray(h1), create_pdarray(h2)

    def to_hdf(
        self,
        prefix_path,
        dataset: str = "segarray",
        mode: Literal["truncate", "append"] = "truncate",
        file_type: Literal["single", "distribute"] = "distribute",
    ):
        """
        Save the SegArray to HDF5. The result is a collection of HDF5 files, one file
        per locale of the arkouda server, where each filename starts with prefix_path.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files will share
        dataset : str
            Name prefix for saved data within the HDF5 file
        mode : {'truncate', 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', add data as a new column to existing files.
        file_type: {"single", "distribute"}
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.

        Returns
        -------
        None

        See Also
        --------
        load
        """
        from arkouda.core.client import generic_msg
        from arkouda.pandas.io import _file_type_to_int, _mode_str_to_int

        return generic_msg(
            cmd="tohdf",
            args={
                "values": self.values.name,
                "segments": self.segments.name,
                "dset": dataset,
                "write_mode": _mode_str_to_int(mode),
                "filename": prefix_path,
                "dtype": self.dtype,
                "objType": self.objType,
                "file_format": _file_type_to_int(file_type),
            },
        )

    def update_hdf(
        self,
        prefix_path: str,
        dataset: str = "segarray",
        repack: bool = True,
    ):
        """
        Overwrite the dataset with the name provided with this SegArray object. If
        the dataset does not exist it is added.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files
        repack: bool
            Default: True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the SegArray

        Notes
        -----
        - If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        - If the dataset provided does not exist, it will be added
        - Because HDF5 deletes do not release memory, this will create a copy of the
          file with the new data
        """
        from arkouda.core.client import generic_msg
        from arkouda.pandas.io import (
            _file_type_to_int,
            _get_hdf_filetype,
            _mode_str_to_int,
            _repack_hdf,
        )

        if self.dtype == str_:
            # Support will be added by Issue #2443
            raise TypeError("SegArrays with Strings values are not yet supported by HDF5")

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        generic_msg(
            cmd="tohdf",
            args={
                "values": self.values.name,
                "segments": self.segments.name,
                "dset": dataset,
                "write_mode": _mode_str_to_int("append"),
                "filename": prefix_path,
                "dtype": self.dtype,
                "objType": self.objType,
                "file_format": _file_type_to_int(file_type),
                "overwrite": True,
            },
        )

        if repack:
            _repack_hdf(prefix_path)

    def to_parquet(
        self,
        prefix_path,
        dataset="segarray",
        mode: Literal["truncate", "append"] = "truncate",
        compression: Optional[str] = None,
    ):
        """
        Save the SegArray object to Parquet. The result is a collection of files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the object to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : {'truncate', 'append'}
            Deprecated.
            Parameter kept to maintain functionality of other calls. Only Truncate
            supported.
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        compression : str (Optional)
            (None | "snappy" | "gzip" | "brotli" | "zstd" | "lz4")
            Sets the compression type used with Parquet files

        Returns
        -------
        string message indicating result of save operation

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        ValueError
            If write mode is not Truncate.

        Notes
        -----
        - Append mode for Parquet has been deprecated. It was not implemented for SegArray.
        - The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        - Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales`` for `file_type='distribute'`.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        - Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        """
        from arkouda.core.client import generic_msg
        from arkouda.pandas.io import _mode_str_to_int

        if mode.lower() == "append":
            raise ValueError("Append mode is not supported for SegArray.")

        return generic_msg(
            "writeParquet",
            {
                "values": self.values.name,
                "segments": self.segments.name,
                "dset": dataset,
                "mode": _mode_str_to_int(mode),
                "prefix": prefix_path,
                "objType": self.objType,
                "compression": compression,
            },
        )

    @classmethod
    def read_hdf(cls, prefix_path, dataset="segarray"):
        """
        Load a saved SegArray from HDF5. All arguments must match what
        was supplied to SegArray.save().

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix
        dataset : str
            Name prefix for saved data within the HDF5 files

        Returns
        -------
        SegArray
        """
        from arkouda.pandas.io import read_hdf

        return read_hdf(prefix_path, datasets=dataset)

    def intersect(self, other):
        """
        Computes the intersection of 2 SegArrays.

        Parameters
        ----------
        other : SegArray
            SegArray to compute against

        Returns
        -------
        SegArray
            Segments are the 1d intersections of the segments of self and other

        See Also
        --------
        pdarraysetops.intersect1d

        Examples
        --------
        >>> import arkouda as ak
        >>> a = [1, 2, 3, 1, 4]
        >>> b = [3, 1, 4, 5]
        >>> c = [1, 3, 3, 5]
        >>> d = [2, 2, 4]
        >>> seg_a = ak.SegArray(ak.array([0, len(a)]), ak.array(a+b))
        >>> seg_b = ak.SegArray(ak.array([0, len(c)]), ak.array(c+d))
        >>> seg_a.intersect(seg_b)
        SegArray([
        [1 3]
        [4]
        ])
        """
        from arkouda.numpy.pdarraysetops import intersect1d

        a_seg_inds = self.grouping.broadcast(arange(self.size)[self.non_empty])
        b_seg_inds = other.grouping.broadcast(arange(other.size)[other.non_empty])
        (new_seg_inds, new_values) = intersect1d([a_seg_inds, self.values], [b_seg_inds, other.values])
        g = GroupBy(new_seg_inds)
        # This method does not return any empty resulting segments
        # We need to add these if they are missing
        if g.segments.size == self.size:
            return SegArray(g.segments, new_values[g.permutation])
        else:
            segments = zeros(self.size, dtype=akint64)
            truth = ones(self.size, dtype=akbool)
            k, ct = g.size()
            segments[k] = g.segments
            truth[k] = zeros(k.size, dtype=akbool)
            if truth[-1]:
                segments[-1] = g.permutation.size
                truth[-1] = False
            segments[truth] = segments[arange(self.size)[truth] + 1]
            return SegArray(segments, new_values[g.permutation])

    def union(self, other):
        """
        Computes the union of 2 SegArrays.

        Parameters
        ----------
        other : SegArray
            SegArray to compute against

        Returns
        -------
        SegArray
            Segments are the 1d union of the segments of self and other

        See Also
        --------
        pdarraysetops.union1d

        Examples
        --------
        >>> import arkouda as ak
        >>> a = [1, 2, 3, 1, 4]
        >>> b = [3, 1, 4, 5]
        >>> c = [1, 3, 3, 5]
        >>> d = [2, 2, 4]
        >>> seg_a = ak.SegArray(ak.array([0, len(a)]), ak.array(a+b))
        >>> seg_b = ak.SegArray(ak.array([0, len(c)]), ak.array(c+d))
        >>> seg_a.union(seg_b)
        SegArray([
        [1 2 3 4 5]
        [1 2 3 4 5]
        ])
        """
        from arkouda.numpy.pdarraysetops import union1d

        a_seg_inds = self.grouping.broadcast(arange(self.size)[self.non_empty])
        b_seg_inds = other.grouping.broadcast(arange(other.size)[other.non_empty])
        (new_seg_inds, new_values) = union1d([a_seg_inds, self.values], [b_seg_inds, other.values])
        g = GroupBy(new_seg_inds)
        # This method does not return any empty resulting segments
        # We need to add these if they are missing
        if g.segments.size == self.size:
            return SegArray(g.segments, new_values[g.permutation])
        else:
            segments = zeros(self.size, dtype=akint64)
            truth = ones(self.size, dtype=akbool)
            k, ct = g.size()
            segments[k] = g.segments
            truth[k] = zeros(k.size, dtype=akbool)
            if truth[-1]:
                segments[-1] = g.permutation.size
                truth[-1] = False
            segments[truth] = segments[arange(self.size)[truth] + 1]
            return SegArray(segments, new_values[g.permutation])

    def setdiff(self, other):
        """
        Computes the set difference of 2 SegArrays.

        Parameters
        ----------
        other : SegArray
            SegArray to compute against

        Returns
        -------
        SegArray
            Segments are the 1d set difference of the segments of self and other

        See Also
        --------
        pdarraysetops.setdiff1d

        Examples
        --------
        >>> import arkouda as ak
        >>> a = [1, 2, 3, 1, 4]
        >>> b = [3, 1, 4, 5]
        >>> c = [1, 3, 3, 5]
        >>> d = [2, 2, 4]
        >>> seg_a = ak.SegArray(ak.array([0, len(a)]), ak.array(a+b))
        >>> seg_b = ak.SegArray(ak.array([0, len(c)]), ak.array(c+d))
        >>> seg_a.setdiff(seg_b)
        SegArray([
        [2 4]
        [1 3 5]
        ])
        """
        from arkouda.numpy.pdarraysetops import setdiff1d

        a_seg_inds = self.grouping.broadcast(arange(self.size)[self.non_empty])
        b_seg_inds = other.grouping.broadcast(arange(other.size)[other.non_empty])
        (new_seg_inds, new_values) = setdiff1d([a_seg_inds, self.values], [b_seg_inds, other.values])
        g = GroupBy(new_seg_inds)
        # This method does not return any empty resulting segments
        # We need to add these if they are missing
        if g.segments.size == self.size:
            return SegArray(g.segments, new_values[g.permutation])
        else:
            segments = zeros(self.size, dtype=akint64)
            truth = ones(self.size, dtype=akbool)
            k, ct = g.size()
            segments[k] = g.segments
            truth[k] = zeros(k.size, dtype=akbool)
            if truth[-1]:
                segments[-1] = g.permutation.size
                truth[-1] = False
            segments[truth] = segments[arange(self.size)[truth] + 1]
            return SegArray(segments, new_values[g.permutation])

    def setxor(self, other):
        """
        Computes the symmetric difference of 2 SegArrays.

        Parameters
        ----------
        other : SegArray
            SegArray to compute against

        Returns
        -------
        SegArray
            Segments are the 1d symmetric difference of the segments of self and other

        See Also
        --------
        pdarraysetops.setxor1d

        Examples
        --------
        >>> import arkouda as ak
        >>> a = [1, 2, 3, 1, 4]
        >>> b = [3, 1, 4, 5]
        >>> c = [1, 3, 3, 5]
        >>> d = [2, 2, 4]
        >>> seg_a = ak.SegArray(ak.array([0, len(a)]), ak.array(a+b))
        >>> seg_b = ak.SegArray(ak.array([0, len(c)]), ak.array(c+d))
        >>> seg_a.setxor(seg_b)
        SegArray([
        [2 4 5]
        [1 2 3 5]
        ])
        """
        from arkouda.numpy.pdarraysetops import setxor1d

        a_seg_inds = self.grouping.broadcast(arange(self.size)[self.non_empty])
        b_seg_inds = other.grouping.broadcast(arange(other.size)[other.non_empty])
        (new_seg_inds, new_values) = setxor1d([a_seg_inds, self.values], [b_seg_inds, other.values])
        g = GroupBy(new_seg_inds)
        # This method does not return any empty resulting segments
        # We need to add these if they are missing
        if g.segments.size == self.size:
            return SegArray(g.segments, new_values[g.permutation])
        else:
            segments = zeros(self.size, dtype=akint64)
            truth = ones(self.size, dtype=akbool)
            k, ct = g.size()
            segments[k] = g.segments
            truth[k] = zeros(k.size, dtype=akbool)
            if truth[-1]:
                segments[-1] = g.permutation.size
                truth[-1] = False
            segments[truth] = segments[arange(self.size)[truth] + 1]
            return SegArray(segments, new_values[g.permutation])

    def filter(self, filter, discard_empty: bool = False):
        """
        Filter values out of the SegArray object.

        Parameters
        ----------
        filter: pdarray, list, or value
            The value/s to be filtered out of the SegArray
        discard_empty: bool
            Defaults to False. When True, empty segments are removed from
            the return SegArray

        Returns
        -------
        SegArray
        """
        from arkouda.numpy import cumsum
        from arkouda.numpy.pdarraysetops import in1d

        # convert to pdarray if more than 1 element
        if isinstance(filter, Sequence):
            filter = array(filter)

        # create boolean index for values to keep
        keep = (
            in1d(self.values, filter, invert=True)
            if isinstance(filter, pdarray) or isinstance(filter, Strings)
            else self.values != filter
        )

        new_vals = self.values[keep]
        lens = self.lengths[:]
        # recreate the segment boundaries
        seg_cts = self.grouping.sum(keep)[1]
        lens[self.non_empty] = seg_cts
        new_segs = cumsum(lens) - lens

        new_segarray = SegArray(new_segs, new_vals)
        return new_segarray[new_segarray.non_empty] if discard_empty else new_segarray

    def register(self, user_defined_name):
        """
        Register this SegArray object and underlying components with the Arkouda server.

        Parameters
        ----------
        user_defined_name : str
            user defined name which this SegArray object will be registered under

        Returns
        -------
        SegArray
            The same SegArray which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different SegArrays with the same name.

        Raises
        ------
        RegistrationError
            Raised if the server could not register the SegArray object

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        See Also
        --------
        unregister, attach, is_registered

        """
        from arkouda.core.client import generic_msg

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")
        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "segments": self.segments,
                "values": self.values,
                "val_type": self.values.objType,
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this SegArray object in the arkouda server which was previously
        registered using register() and/or attached to using attach().

        Raises
        ------
        RuntimeError
            Raised if the server could not unregister the SegArray object from the Symbol Table

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        See Also
        --------
        register, attach, is_registered

        """
        from arkouda.numpy.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    def is_registered(self) -> bool:
        """
        Check if the name of the SegArray object is registered in the Symbol Table.

        Returns
        -------
        bool
            True if SegArray is registered, false if not

        See Also
        --------
        register, unregister, attach
        """
        from arkouda.numpy.util import is_registered

        if self.registered_name is None:
            # if it is registered as a component of DataFrame
            return is_registered(self.segments.name, as_component=True) and is_registered(
                self.values.name, as_component=True
            )
        else:
            return is_registered(self.registered_name)

    def transfer(self, hostname: str, port: int_scalars):
        """
        Send a Segmented Array to a different Arkouda server.

        Parameters
        ----------
        hostname : str
            The hostname where the Arkouda server intended to
            receive the Segmented Array is running.
        port : int_scalars
            The port to send the array over. This needs to be an
            open port (i.e., not one that the Arkouda server is
            running on). This will open up `numLocales` ports,
            each of which in succession, so will use ports of the
            range {port..(port+numLocales)} (e.g., running an
            Arkouda server of 4 nodes, port 1234 is passed as
            `port`, Arkouda will use ports 1234, 1235, 1236,
            and 1237 to send the array data).
            This port much match the port passed to the call to
            `ak.receive_array()`.

        Returns
        -------
        A message indicating a complete transfer

        Raises
        ------
        ValueError
            Raised if the op is not within the pdarray.BinOps set
        TypeError
            Raised if other is not a pdarray or the pdarray.dtype is not
            a supported dtype
        """
        from arkouda.core.client import generic_msg

        return generic_msg(
            cmd="sendArray",
            args={
                "segments": self.segments,
                "values": self.values,
                "hostname": hostname,
                "port": port,
                "dtype": self.dtype,
                "objType": "segarray",
            },
        )
