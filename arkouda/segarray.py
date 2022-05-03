from __future__ import annotations

from arkouda.pdarrayclass import pdarray, is_sorted, attach_pdarray
from arkouda.numeric import cumsum
from arkouda.dtypes import isSupportedInt
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import str_
from arkouda.pdarraycreation import zeros, ones, array, arange
from arkouda.pdarraysetops import concatenate
from arkouda.groupbyclass import unique, GroupBy, broadcast
from arkouda.pdarrayIO import load


def gen_ranges(starts, ends, stride=1):
    """ Generate a segmented array of variable-length, contiguous ranges between pairs of start- and end-points.

    Parameters
    ----------
    starts : pdarray, int64
    The start value of each range
    ends : pdarray, int64
    The end value (exclusive) of each range
    stride: int
    Difference between successive elements of each range

    Returns
    -------
    segments : pdarray, int64
    The starting index of each range in the resulting array
    ranges : pdarray, int64
    The actual ranges, flattened into a single array
    """
    if starts.size != ends.size:
        raise ValueError("starts and ends must be same length")
    if starts.size == 0:
        return zeros(0, dtype=akint64), zeros(0, dtype=akint64)
    lengths = (ends - starts) // stride
    segs = cumsum(lengths) - lengths
    totlen = lengths.sum()
    slices = ones(totlen, dtype=akint64)
    diffs = concatenate((array([starts[0]]),
                         starts[1:] - starts[:-1] - (lengths[:-1] - 1) * stride))
    slices[segs] = diffs
    return segs, cumsum(slices)


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
    def __init__(self, segments, values, copy=False, lengths=None, grouping=None):
        """
        An array of variable-length arrays, also called a skyline array or ragged array.

        Parameters
        ----------
        segments : pdarray, int64
            Start index of each sub-array in the flattened values array
        values : pdarray
            The flattened values of all sub-arrays
        copy : bool
            If True, make a copy of the input arrays; otherwise, just store a reference.

        Returns
        -------
        SegArray
            Data structure representing an array whose elements are variable-length arrays.

        Notes
        -----
        Keyword args 'lengths' and 'grouping' are not user-facing. They are used by the
        attach method.
        """
        if not isinstance(segments, pdarray) or segments.dtype != akint64:
            raise TypeError("Segments must be int64 pdarray")
        if not is_sorted(segments):
            raise ValueError("Segments must be unique and in sorted order")
        if segments.size > 0:
            if segments.min() != 0:
                raise ValueError("Segments must start at zero and be less than values.size")
        elif values.size > 0:
            raise ValueError("Cannot have non-empty values with empty segments")
        if copy:
            self.segments = segments[:]
            self.values = values[:]
        else:
            self.segments = segments
            self.values = values
        self.size = segments.size
        self.valsize = values.size
        if lengths is None:
            self.lengths = self._get_lengths()
        else:
            self.lengths = lengths
        self.dtype = values.dtype

        self._non_empty = self.lengths > 0
        self._non_empty_count = self._non_empty.sum()

        if grouping is None:
            if self.size == 0 or self._non_empty_count == 0:
                self.grouping = GroupBy(zeros(0, dtype=akint64))
            else:
                # Treat each sub-array as a group, for grouped aggregations
                self.grouping = GroupBy(broadcast(self.segments[self._non_empty], arange(self._non_empty_count), self.valsize))
        else:
            self.grouping = grouping

    @classmethod
    def from_multi_array(cls, m):
        """
        Construct a SegArray from a list of columns. This essentially transposes the input,
        resulting in an array of rows.

        Parameters
        ----------
        m : list of pdarray
            List of columns, the rows of which will form the sub-arrays of the output

        Returns
        -------
        SegArray
            Array of rows of input
        """
        if isinstance(m, pdarray):
            size = m.size
            n = 1
            dtype = m.dtype
        else:
            s = set(mi.size for mi in m)
            if len(s) != 1:
                raise ValueError("All columns must have same length")
            size = s.pop()
            n = len(m)
            d = set(mi.dtype for mi in m)
            if len(d) != 1:
                raise ValueError("All columns must have same dtype")
            dtype = d.pop()
        newsegs = arange(size) * n
        newvals = zeros(size * n, dtype=dtype)
        for j in range(len(m)):
            newvals[j::len(m)] = m[j]
        return cls(newsegs, newvals)

    @classmethod
    def concat(cls, x, axis=0, ordered=True):
        """
        Concatenate a sequence of SegArrays

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
        if not ordered:
            raise ValueError("Unordered concatenation not yet supported on SegArray; use ordered=True.")
        if len(x) == 0:
            raise ValueError("Empty sequence passed to concat")
        for xi in x:
            if not isinstance(xi, cls):
                return NotImplemented
        if len(set(xi.dtype for xi in x)) != 1:
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
            sizes = set(xi.size for xi in x)
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
                fromself = (cumsum(fromself[:-1]) == 1)
                newvals[fromself] = xi.values
                nzsegs += nzlens
            return cls(newsegs, newvals, copy=False)
        else:
            raise ValueError("Supported values for axis are 0 (vertical concat) or 1 (horizontal concat)")

    def copy(self):
        """
        Return a deep copy.
        """
        return SegArray(self.segments, self.values, copy=True)

    def _get_lengths(self):
        if self.size == 0:
            return zeros(0, dtype=akint64)
        elif self.size == 1:
            return array([self.valsize])
        else:
            return concatenate((self.segments[1:], array([self.valsize]))) - self.segments

    def __getitem__(self, i):
        if isSupportedInt(i):
            start = self.segments[i]
            end = self.segments[i] + self.lengths[i]
            return self.values[start:end].to_ndarray()
        elif (isinstance(i, pdarray) and (i.dtype == akint64 or i.dtype == akbool)) or isinstance(i, slice):
            starts = self.segments[i]
            ends = starts + self.lengths[i]
            newsegs, inds = gen_ranges(starts, ends)
            return SegArray(newsegs, self.values[inds], copy=True)
        else:
            raise TypeError(f'Invalid index type: {type(i)}')

    def __eq__(self, other):
        if not isinstance(other, SegArray):
            return NotImplemented
        eq = zeros(self.size, dtype=akbool)
        leneq = self.lengths == other.lengths
        if leneq.sum() > 0:
            selfcmp = self[leneq]
            othercmp = other[leneq]
            intersection = self.all(selfcmp.values == othercmp.values)
            eq[leneq] = intersection
        return eq

    def __str__(self):
        if self.size <= 6:
            rows = list(range(self.size))
        else:
            rows = [0, 1, 2, None, self.size - 3, self.size - 2, self.size - 1]
        outlines = ['SegArray([']
        for r in rows:
            if r is None:
                outlines.append('...')
            else:
                outlines.append(str(self[r]))
        outlines.append('])')
        return '\n'.join(outlines)

    def __repr__(self):
        return self.__str__()

    def get_suffixes(self, n, return_origins=True, proper=True):
        """
        Return the n-long suffix of each sub-array, where possible

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
        Return all sub-array prefixes of length n (for sub-arrays that are at least n+1 long)

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
        ngrams : list of pdarray
            An n-long list of pdarrays, essentially a table where each row is an n-gram.
        origin_indices : pdarray, int
            The index of the sub-array from which the corresponding n-gram originated
        """
        if n > self._get_lengths().max():
            raise ValueError('n must be <= the maximum length of the sub-arrays')

        ngrams = []
        notsegstart = ones(self.valsize, dtype=akbool)
        notsegstart[self.segments[self._non_empty]] = False
        valid = ones(self.valsize - n + 1, dtype=akbool)
        for i in range(n):
            end = self.valsize - n + i + 1
            ngrams.append(self.values[i:end])
            if i > 0:
                valid &= notsegstart[i:end]
        ngrams = [char[valid] for char in ngrams]
        if return_origins:
            # set the proper indexes for broadcasting. Needed to alot for empty segments
            seg_idx = arange(self.size)[self._non_empty]
            origin_indices = self.grouping.broadcast(seg_idx, permute=True)[:valid.size][valid]
            return ngrams, origin_indices
        else:
            return ngrams

    def _normalize_index(self, j):
        if not isSupportedInt(j):
            raise TypeError(f'index must be integer, not {type(j)}')
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
        val : pdarray
            compressed=False: The j-th value of each sub-array where j is in
            bounds and the default value where j is out of bounds.
            compressed=True: The j-th values of only the sub-arrays where j is
            in bounds
        origin_indices : pdarray, bool
            A Boolean array that is True where j is in bounds for the sub-array.
        """
        longenough, newj = self._normalize_index(j)
        ind = (self.segments + newj)[longenough]
        if compressed:
            res = self.values[ind]
        else:
            res = zeros(self.size, dtype=self.dtype) + default
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
        -----
        ValueError
            If j is out of bounds in any of the sub-arrays specified by i.
        """
        if self.dtype == str_:
            raise TypeError("String elements are immutable")
        longenough, newj = self._normalize_index(j)
        if not longenough[i].all():
            raise ValueError(f'Not all (i, j) in bounds')
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
        '''
        Append a single value to each sub-array.

        Parameters
        ----------
        x : pdarray or scalar
            Single value to append to each sub-array

        Returns
        -------
        SegArray
            Copy of original SegArray with values from x appended to each sub-array
        '''
        if hasattr(x, 'size'):
            if x.size != self.size:
                raise ValueError('Argument must be scalar or same size as SegArray')
            if type(x) != type(self.values) or x.dtype != self.dtype:
                raise TypeError('Argument type must match value type of SegArray')
        newlens = self.lengths + 1
        newsegs = cumsum(newlens) - newlens
        newvals = zeros(newlens.sum(), dtype=self.dtype)
        if prepend:
            lastscatter = newsegs
        else:
            lastscatter = newsegs + newlens - 1
        newvals[lastscatter] = x
        origscatter = arange(self.valsize) + self.grouping.broadcast(arange(self._non_empty_count), permute=True)
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
        norepeats : SegArray
            Sub-arrays with runs of repeated values replaced with single value
        multiplicity : SegArray
            If return_multiplicity=True, this array contains the number of times
            each value in the returned SegArray was repeated in the original SegArray.
        """
        isrepeat = zeros(self.values.size, dtype=akbool)
        isrepeat[1:] = self.values[:-1] == self.values[1:]
        isrepeat[self.segments[self._non_empty]] = False
        truepaths = self.values[~isrepeat]
        nhops = self.grouping.sum(~isrepeat)[1]
        truesegs = cumsum(nhops) - nhops
        # Correct segments to properly assign empty lists - prevents dropping empty segments
        if not self._non_empty.all():
            truelens = concatenate((truesegs[1:], array([truepaths.size]))) - truesegs
            len_diff = self.lengths[self._non_empty] - truelens

            x = 0  # tracking which non-empty segment length we need 
            truesegs = zeros(self.size, dtype=akint64)
            for i in range(1, self.size):
                truesegs[i] = (self.segments[i] - len_diff[:x+1].sum())
                if self._non_empty[i]:
                    x += 1

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
        Convert the array into a numpy.ndarray containing sub-arrays

        Returns
        -------
        np.ndarray
            A numpy ndarray with the same sub-arrays (also numpy.ndarray) as this array

        See Also
        --------
        array

        Examples
        --------
        >>> segarr = ak.SegArray(ak.array([0, 4, 7]), ak.arange(12))
        >>> segarr.to_ndarray()
        array([[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12]])
        >>> type(segarr.to_ndarray())
        numpy.ndarray
        """
        ndvals = self.values.to_ndarray()
        ndsegs = self.segments.to_ndarray()
        arr = [ndvals[start:end] for start, end in zip(ndsegs, ndsegs[1:])]
        if self.size > 0:
            arr.append(ndvals[ndsegs[-1]:])
        return arr

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

    def unique(self, x=None):
        '''
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
        '''
        if x is None:
            x = self.values
        keyidx = self.grouping.broadcast(arange(self.size), permute=True)
        ukey, uval = GroupBy([keyidx, x]).unique_keys
        g = GroupBy(ukey, assume_sorted=True)
        _, lengths = g.count()
        return SegArray(g.segments, uval, grouping=g, lengths=lengths)

    def save(self, prefix_path, dataset='segarray', segment_suffix='_segments', value_suffix='_values',
             mode='truncate'):
        '''
        Save the SegArray to HDF5. The result is a collection of HDF5 files, one file
        per locale of the arkouda server, where each filename starts with prefix_path.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files will share
        dataset : str
            Name prefix for saved data within the HDF5 file
        segment_suffix : str
            Suffix to append to dataset name for segments array
        value_suffix : str
            Suffix to append to dataset name for values array
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', add data as a new column to existing files.

        Returns
        -------
        None

        Notes
        -----
        Unlike for ak.Strings, SegArray is saved as two datasets in the top level of
        the HDF5 file, not nested under a group.
        '''
        if segment_suffix == value_suffix:
            raise ValueError("Segment suffix and value suffix must be different")
        self.segments.save(prefix_path, dataset=dataset + segment_suffix, mode=mode)
        self.values.save(prefix_path, dataset=dataset + value_suffix, mode='append')

    @classmethod
    def load(cls, prefix_path, dataset='segarray', segment_suffix='_segments', value_suffix='_values', mode='truncate'):
        '''
        Load a saved SegArray from HDF5. All arguments msut match what
        was supplied to SegArray.save()

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix
        dataset : str
            Name prefix for saved data within the HDF5 files
        segment_suffix : str
            Suffix to append to dataset name for segments array
        value_suffix : str
            Suffix to append to dataset name for values array

        Returns
        -------
        SegArray
        '''
        if segment_suffix == value_suffix:
            raise ValueError("Segment suffix and value suffix must be different")
        segments = load(prefix_path, dataset=dataset + segment_suffix)
        values = load(prefix_path, dataset=dataset + value_suffix)
        return cls(segments, values)

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
        >>> a = [1, 2, 3, 1, 4]
        >>> b = [3, 1, 4, 5]
        >>> c = [1, 3, 3, 5]
        >>> d = [2, 2, 4]
        >>> seg_a = ak.SegArray(ak.array([0, len(a)]), ak.array(a+b))
        >>> seg_b = ak.SegArray(ak.array([0, len(c)]), ak.array(c+d))
        >>> seg_a.intersect(seg_b)
        SegArray([
        [1, 3],
        [4]
        ])
        """
        from arkouda.pdarraysetops import intersect1d

        a_seg_inds = self.grouping.broadcast(arange(self.size)[self._non_empty])
        b_seg_inds = other.grouping.broadcast(arange(other.size)[other._non_empty])
        (new_seg_inds, new_values) = intersect1d([a_seg_inds, self.values], [b_seg_inds, other.values])
        g = GroupBy(new_seg_inds)
        # This method does not return any empty resulting segments. We need to add these if they are missing.
        if g.segments.size == self.size:
            return SegArray(g.segments, new_values[g.permutation])
        else:
            segments = zeros(self.size, dtype=akint64)
            truth = ones(self.size, dtype=akbool)
            k, ct = g.count()
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
        >>> a = [1, 2, 3, 1, 4]
        >>> b = [3, 1, 4, 5]
        >>> c = [1, 3, 3, 5]
        >>> d = [2, 2, 4]
        >>> seg_a = ak.SegArray(ak.array([0, len(a)]), ak.array(a+b))
        >>> seg_b = ak.SegArray(ak.array([0, len(c)]), ak.array(c+d))
        >>> seg_a.union(seg_b)
        SegArray([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
        ])
        """
        from arkouda.pdarraysetops import union1d

        a_seg_inds = self.grouping.broadcast(arange(self.size)[self._non_empty])
        b_seg_inds = other.grouping.broadcast(arange(other.size)[other._non_empty])
        (new_seg_inds, new_values) = union1d([a_seg_inds, self.values], [b_seg_inds, other.values])
        g = GroupBy(new_seg_inds)
        # This method does not return any empty resulting segments. We need to add these if they are missing.
        if g.segments.size == self.size:
            return SegArray(g.segments, new_values[g.permutation])
        else:
            segments = zeros(self.size, dtype=akint64)
            truth = ones(self.size, dtype=akbool)
            k, ct = g.count()
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
        >>> a = [1, 2, 3, 1, 4]
        >>> b = [3, 1, 4, 5]
        >>> c = [1, 3, 3, 5]
        >>> d = [2, 2, 4]
        >>> seg_a = ak.SegArray(ak.array([0, len(a)]), ak.array(a+b))
        >>> seg_b = ak.SegArray(ak.array([0, len(c)]), ak.array(c+d))
        >>> seg_a.setdiff(seg_b)
        SegArray([
        [2, 4],
        [1, 3, 5]
        ])
        """
        from arkouda.pdarraysetops import setdiff1d

        a_seg_inds = self.grouping.broadcast(arange(self.size)[self._non_empty])
        b_seg_inds = other.grouping.broadcast(arange(other.size)[other._non_empty])
        (new_seg_inds, new_values) = setdiff1d([a_seg_inds, self.values], [b_seg_inds, other.values])
        g = GroupBy(new_seg_inds)
        # This method does not return any empty resulting segments. We need to add these if they are missing.
        if g.segments.size == self.size:
            return SegArray(g.segments, new_values[g.permutation])
        else:
            segments = zeros(self.size, dtype=akint64)
            truth = ones(self.size, dtype=akbool)
            k, ct = g.count()
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
        >>> a = [1, 2, 3, 1, 4]
        >>> b = [3, 1, 4, 5]
        >>> c = [1, 3, 3, 5]
        >>> d = [2, 2, 4]
        >>> seg_a = ak.SegArray(ak.array([0, len(a)]), ak.array(a+b))
        >>> seg_b = ak.SegArray(ak.array([0, len(c)]), ak.array(c+d))
        >>> seg_a.setxor(seg_b)
        SegArray([
        [2, 4, 5],
        [1, 3, 5, 2]
        ])
        """
        from arkouda.pdarraysetops import setxor1d

        a_seg_inds = self.grouping.broadcast(arange(self.size)[self._non_empty])
        b_seg_inds = other.grouping.broadcast(arange(other.size)[other._non_empty])
        (new_seg_inds, new_values) = setxor1d([a_seg_inds, self.values], [b_seg_inds, other.values])
        g = GroupBy(new_seg_inds)
        # This method does not return any empty resulting segments. We need to add these if they are missing.
        if g.segments.size == self.size:
            return SegArray(g.segments, new_values[g.permutation])
        else:
            segments = zeros(self.size, dtype=akint64)
            truth = ones(self.size, dtype=akbool)
            k, ct = g.count()
            segments[k] = g.segments
            truth[k] = zeros(k.size, dtype=akbool)
            if truth[-1]:
                segments[-1] = g.permutation.size
                truth[-1] = False
            segments[truth] = segments[arange(self.size)[truth] + 1]
            return SegArray(segments, new_values[g.permutation])

    def register(self, name, segment_suffix='_segments', value_suffix='_values', length_suffix='_lengths', grouping_suffix='_grouping'):
        if len(set((segment_suffix, value_suffix, length_suffix, grouping_suffix))) != 4:
            raise ValueError("Suffixes must all be different")
        self.segments.register(name+segment_suffix)
        self.values.register(name+value_suffix)
        self.lengths.register(name+length_suffix)
        # TODO - groupby does not have register.
        # self.grouping.register(name+grouping_suffix)

    def unregister(self):
        self.segments.unregister()
        self.values.unregister()
        self.lengths.unregister()
        # TODO - groupby does not have unregister.
        #self.grouping.unregister()

    @classmethod
    def attach(cls, name, segment_suffix='_segments', value_suffix='_values', length_suffix='_lengths', grouping_suffix='_grouping'):
        if len(set((segment_suffix, value_suffix, length_suffix, grouping_suffix))) != 4:
            raise ValueError("Suffixes must all be different")
        # TODO - add grouping attaching grouping=ak.GroupBy.attach(name+grouping_suffix)
        return cls(attach_pdarray(name+segment_suffix), attach_pdarray(name+value_suffix),
                   lengths=attach_pdarray(name+length_suffix))
