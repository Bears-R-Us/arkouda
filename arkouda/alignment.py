"""
Alignment and lookup utilities for Arkouda arrays.

This module provides functions to align multiple arrays to a common
0-up index, perform lookups and mappings across sparse identifiers,
and search within defined intervals. It supports single and multi-dimensional
key matching, including hierarchical keys, and interval-based function evaluation.

Functions
---------
- align: Align multiple arrays to a common index.
- left_align: Align two arrays to the index defined by the left array.
- right_align: Align two arrays to the index defined by the right array.
- zero_up: Map sparse identifiers to 0-up indices.
- lookup: Evaluate a function defined by keys and values on input arguments.
- interval_lookup: Evaluate a function defined over intervals.
- search_intervals: Return index of best interval for each query value.
- in1d_intervals: Check membership of values in half-open intervals.
- find: Locate indices of query values in a search space.
- is_cosorted: Determine if a list of arrays are cosorted.
- unsqueeze: Wrap a pdarray in a list if not already a sequence.

Classes
-------
- NonUniqueError: Raised when duplicate values are found in keys.

Examples
--------
>>> import arkouda as ak
>>> a = ak.array([10, 20, 30, 40])
>>> b = ak.array([20, 10, 40, 50])
>>> keep, (a_aligned, b_aligned) = ak.right_align(a, b)
>>> a[keep], a_aligned, b_aligned
(array([10 20 40]), array([0 1 2]), array([1 0 2 3]))

>>> starts = ak.array([0, 5])
>>> ends = ak.array([3, 10])
>>> values = ak.array([100, 200])
>>> x = ak.array([1, 6, 8])
>>> ak.interval_lookup((starts, ends), values, x)
array([100 200 200])

"""

import functools
from typing import Sequence
from warnings import warn

import numpy as np

from arkouda.categorical import Categorical
from arkouda.client import generic_msg
from arkouda.groupbyclass import GroupBy, broadcast, unique
from arkouda.numpy import cumsum, where
from arkouda.numpy.dtypes import bigint
from arkouda.numpy.dtypes import float64 as akfloat64
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import uint64 as akuint64
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import arange, full, ones, zeros
from arkouda.numpy.pdarraysetops import concatenate, in1d
from arkouda.numpy.sorting import argsort, coargsort
from arkouda.numpy.strings import Strings

__all__ = [
    "NonUniqueError",
    "align",
    "find",
    "in1d_intervals",
    "interval_lookup",
    "is_cosorted",
    "left_align",
    "lookup",
    "right_align",
    "search_intervals",
    "unsqueeze",
    "zero_up",
]


def unsqueeze(p):
    """
    Ensure that the input is returned as a list.

    If the input is a single pdarray, Strings, or Categorical object, wrap it in a list.
    Otherwise, return the input unchanged.

    Parameters
    ----------
    p : pdarray, Strings, Categorical, or Sequence
        The input object to be wrapped or returned as-is.

    Returns
    -------
    Sequence
        A list containing the input, or the input itself if it is already a sequence.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([1, 2, 3])
    >>> unsqueeze(a)
    [array([1 2 3])]

    """
    if isinstance(p, pdarray) or isinstance(p, Strings) or isinstance(p, Categorical):
        return [p]
    else:
        return p


def zero_up(vals):
    """
    Map an array of sparse values to 0-up indices.

    Parameters
    ----------
    vals : pdarray
        Array to map to dense index

    Returns
    -------
    aligned : pdarray
        Array with values replaced by 0-up indices

    """
    g = GroupBy(vals)
    unique_size = g.unique_keys.size if not isinstance(vals, Sequence) else g.unique_keys[0].size
    uniqueInds = arange(unique_size)
    idinds = g.broadcast(uniqueInds, permute=True)
    return idinds


def align(*args):
    """
    Map multiple arrays of sparse identifiers to a common 0-up index.

    Parameters
    ----------
    *args : pdarrays or sequences of pdarrays
        Arrays to map to dense index

    Returns
    -------
    aligned : list of pdarrays
        Arrays with values replaced by 0-up indices

    """
    if not any(isinstance(arg, Sequence) for arg in args):
        key = concatenate([full(arg.size, i, akint64) for i, arg in enumerate(args)], ordered=False)
        inds = zero_up(concatenate(args, ordered=False))
    else:
        if not all(isinstance(arg, Sequence) for arg in args):
            raise TypeError("If any of the arguments are a sequence of pdarray, they all have to be")
        key = concatenate([full(arg[0].size, i, akint64) for i, arg in enumerate(args)], ordered=False)
        inds = zero_up([concatenate(x, ordered=False) for x in zip(*args)])
    return [inds[key == i] for i in range(len(args))]


def right_align(left, right):
    """
    Map two arrays of sparse values to the 0-up index.

    Map two arrays of sparse values to the 0-up index set implied by the right array,
    discarding values from left that do not appear in right.

    Parameters
    ----------
    left : pdarray or a sequence of pdarrays
        Left-hand identifiers
    right : pdarray or a sequence of pdarrays
        Right-hand identifiers that define the index

    Returns
    -------
    pdarray, (pdarray, pdarray)
        keep : pdarray, bool
            Logical index of left-hand values that survived
        aligned : (pdarray, pdarray)
            Left and right arrays with values replaced by 0-up indices

    """
    is_sequence = isinstance(left, Sequence) and isinstance(right, Sequence)
    uright = unique(right)
    keep = in1d(left, uright)
    fleft = left[keep] if not is_sequence else [lf[keep] for lf in left]
    return keep, align(fleft, right)


def left_align(left, right):
    """
    Map two arrays of sparse identifiers to the 0-up index.

    Map two arrays of sparse identifiers to the 0-up index set implied by the left array,
    discarding values from right that do not appear in left.
    """
    return right_align(right, left)


class NonUniqueError(ValueError):
    """
    Exception raised when duplicate values are found in a set of keys that are expected to be unique.

    This is typically raised in lookup and alignment operations that assume
    a one-to-one mapping between keys and values.

    Examples
    --------
    >>> from arkouda.alignment import NonUniqueError
    >>> raise NonUniqueError("Duplicate values found in key array.")
    Traceback (most recent call last):
        ...
    arkouda.alignment.NonUniqueError: Duplicate values found in key array.

    """

    pass


def find(query, space, all_occurrences=False, remove_missing=False):
    """
    Return indices of query items in a search list of items.

    Parameters
    ----------
    query : (sequence of) array-like
        The items to search for. If multiple arrays, each "row" is an item.
    space : (sequence of) array-like
        The set of items in which to search. Must have same shape/dtype as query.
    all_occurrences: bool
        When duplicate terms are present in search space, if all_occurrences is True,
        return all occurrences found as a SegArray, otherwise return only the first
        occurrences as a pdarray. Defaults to only finding the first occurrence.
        Finding all occurrences is not yet supported on sequences of arrays
    remove_missing: bool
        If all_occurrences is True, remove_missing is automatically enabled.
        If False, return -1 for any items in query not found in space. If True,
        remove these and only return indices of items that are found.

    Returns
    -------
    indices : pdarray or SegArray
        For each item in query, its index in space. If all_occurrences is False,
        the return will be a pdarray of the first index where each value in the
        query appears in the space. If all_occurrences is True, the return will be
        a SegArray containing every index where each value in the query appears in
        the space. If all_occurrences is True, remove_missing is automatically enabled.
        If remove_missing is True, exclude missing values, otherwise return -1.

    Examples
    --------
    >>> import arkouda as ak
    >>> select_from = ak.arange(10)
    >>> arr1 = select_from[ak.randint(0, select_from.size, 20, seed=10)]
    >>> arr2 = select_from[ak.randint(0, select_from.size, 20, seed=11)]

    Remove some values to ensure we have some values
    which don't appear in the search space

    >>> arr2 = arr2[arr2 != 9]
    >>> arr2 = arr2[arr2 != 3]

    Find with defaults (all_occurrences and remove_missing both False)

    >>> ak.find(arr1, arr2)
    array([-1 -1 -1 0 1 -1 -1 -1 2 -1 5 -1 8 -1 5 -1 -1 11 5 0])

    Set remove_missing to True, only difference from default
    is missing values are excluded

    >>> ak.find(arr1, arr2, remove_missing=True)
    array([0 1 2 5 8 5 11 5 0])

    Set both remove_missing and all_occurrences to True, missing values
    will be empty segments

    >>> ak.find(arr1, arr2, remove_missing=True, all_occurrences=True).to_list()
    [[],
     [],
     [],
     [0, 4],
     [1, 3, 10],
     [],
     [],
     [],
     [2, 6, 12, 13],
     [],
     [5, 7],
     [],
     [8, 9, 14],
     [],
     [5, 7],
     [],
     [],
     [11, 15],
     [5, 7],
     [0, 4]]

    """
    # Concatenate the space and query in fast (block interleaved) mode
    if isinstance(query, (pdarray, Strings, Categorical)):
        if type(query) is not type(space):
            raise TypeError("Arguments must have same type")
        c = concatenate((space, query), ordered=False)
        spacesize = space.size
        querysize = query.size
    else:
        if len(query) != len(space):
            raise TypeError("Multi-array arguments must have same number of arrays")
        spacesize = {s.size for s in space}
        querysize = {q.size for q in query}
        if len(spacesize) != 1 or len(querysize) != 1:
            raise TypeError("Multi-array arguments must be non-empty and have equal-length arrays")
        spacesize = spacesize.pop()
        querysize = querysize.pop()
        atypes = np.array([ai.dtype for ai in query])
        btypes = np.array([bi.dtype for bi in space])
        if not (atypes == btypes).all():
            raise TypeError("Array dtypes of arguments must match")
        c = [concatenate((si, qi), ordered=False) for si, qi in zip(space, query)]
    # Combined index of space and query elements, in block interleaved order
    # All space indices are less than all query indices
    i = concatenate((arange(spacesize), arange(spacesize, spacesize + querysize)), ordered=False)
    # Group on terms
    g = GroupBy(c, dropna=False)
    # For each term, count how many times it appears in the search space

    # since we reuse (i < spacesize)[g.permutation] later, we call sum aggregation manually
    less_than = (i < spacesize)[g.permutation]
    repMsg = generic_msg(
        cmd="segmentedReduction",
        args={
            "values": less_than,
            "segments": g.segments,
            "op": "sum",
            "skip_nan": True,
            "ddof": 1,
        },
    )

    space_multiplicity = create_pdarray(repMsg)
    has_duplicates = (space_multiplicity > 1).any()
    # handle duplicate terms in space
    if has_duplicates:
        if all_occurrences:
            if isinstance(query, Sequence):
                raise TypeError("finding all_occurrences is not yet supported on sequences of arrays")

            from arkouda.numpy.segarray import SegArray

            # create a segarray which contains all the indices from query
            # in our search space, instead of just the min for each segment
            # im not completely convinced there's not a better way to get this given the
            # amount of structure but this is not the bottleneck of the computation anymore
            min_k_vals = i[g.permutation][less_than]
            seg_idx = g.broadcast(arange(g.segments.size))[i >= spacesize]
            min_k_segs = cumsum(space_multiplicity) - space_multiplicity
            sa = SegArray(min_k_segs, min_k_vals)
            return sa[seg_idx]
        else:
            warn(
                "Duplicate terms present in search space. Only first instance of each query term"
                " will be reported. To return all occurrences, set all_occurrences=True."
            )
    # For query terms in the space, the min combined index will be the first index of that
    # term in the space
    uspaceidx = g.min(i)[1]
    # For query terms not in the space, the min combined index will exceed the space size
    # and should be set to -1
    uspaceidx = where(uspaceidx >= spacesize, -1, uspaceidx)
    # Broadcast unique term indices to combined list of space and query terms
    spaceidx = g.broadcast(uspaceidx)
    # Return only the indices of the query terms (remove the search space)
    pda = spaceidx[i >= spacesize]
    return pda[pda != -1] if remove_missing else pda


def lookup(keys, values, arguments, fillvalue=-1):
    """
    Apply the function defined by the mapping keys --> values to arguments.

    Parameters
    ----------
    keys : (sequence of) array-like
        The domain of the function. Entries must be unique (if a sequence of
        arrays is given, each row is treated as a tuple-valued entry).
    values : pdarray
        The range of the function. Must be same length as keys.
    arguments : (sequence of) array-like
        The arguments on which to evaluate the function. Must have same dtype
        (or tuple of dtypes, for a sequence) as keys.
    fillvalue : scalar
        The default value to return for arguments not in keys.

    Returns
    -------
    evaluated : pdarray
        The result of evaluating the function over arguments.

    Notes
    -----
    While the values cannot be Strings (or other complex objects), the same
    result can be achieved by passing an arange as the values, then using
    the return as indices into the desired object.

    Examples
    --------
    >>> import arkouda as ak

    Lookup numbers by two-word name
    >>> keys1 = ak.array(['twenty' for _ in range(5)])
    >>> keys2 = ak.array(['one', 'two', 'three', 'four', 'five'])
    >>> values = ak.array([21, 22, 23, 24, 25])
    >>> args1 = ak.array(['twenty', 'thirty', 'twenty'])
    >>> args2 = ak.array(['four', 'two', 'two'])
    >>> ak.lookup([keys1, keys2], values, [args1, args2])
    array([24 -1 22])

    Other direction requires an intermediate index
    >>> revkeys = values
    >>> revindices = ak.arange(values.size)
    >>> revargs = ak.array([24, 21, 22])
    >>> idx = ak.lookup(revkeys, revindices, revargs)
    >>> keys1[idx], keys2[idx]
    (array(['twenty', 'twenty', 'twenty']),
    array(['four', 'one', 'two']))

    """
    if isinstance(values, Categorical):
        codes = lookup(keys, values.codes, arguments, fillvalue=values._NAcode)
        return Categorical.from_codes(codes, values.categories, NAvalue=values.NAvalue)
    # Find arguments in keys array
    idx = find(arguments, keys)
    # Initialize return values with fillvalue for missing values
    retvals = full(idx.size, fillvalue, dtype=values.dtype)
    # Where arguments were found in keys, put corresponding fuction values
    found = idx >= 0
    retvals[found] = values[idx[found]]
    return retvals


def in1d_intervals(vals, intervals, symmetric=False):
    """
    Test each value for membership in *any* of a set of half-open (pythonic) intervals.

    Parameters
    ----------
    vals : pdarray(int, float)
        Values to test for membership in intervals
    intervals : 2-tuple of pdarrays
        Non-overlapping, half-open intervals, as a tuple of
        (lower_bounds_inclusive, upper_bounds_exclusive)
    symmetric : bool
        If True, also return boolean pdarray indicating which intervals
        contained one or more query values.

    Returns
    -------
    pdarray(bool)
        Array of same length as <vals>, True if corresponding value is
        included in any of the ranges defined by (low[i], high[i]) inclusive.
    pdarray(bool) (if symmetric=True)
        Array of same length as number of intervals, True if corresponding
        interval contains any of the values in <vals>.

    Notes
    -----
    First return array is equivalent to the following:
        ((vals >= intervals[0][0]) & (vals < intervals[1][0])) |
        ((vals >= intervals[0][1]) & (vals < intervals[1][1])) |
        ...
        ((vals >= intervals[0][-1]) & (vals < intervals[1][-1]))
    But much faster when testing many ranges.

    Second (optional) return array is equivalent to:
        ((intervals[0] <= vals[0]) & (intervals[1] > vals[0])) |
        ((intervals[0] <= vals[1]) & (intervals[1] > vals[1])) |
        ...
        ((intervals[0] <= vals[-1]) & (intervals[1] > vals[-1]))
    But much faster when vals is non-trivial size.

    """
    idx = search_intervals(vals, intervals)
    found = idx > -1
    if symmetric:
        containresult = in1d(arange(intervals[0].size), idx)
        return found, containresult
    else:
        return found


def search_intervals(vals, intervals, tiebreak=None, hierarchical=True):
    """
    Return the index of the best interval containing each query value.

    Given an array of query vals and non-overlapping, closed intervals, return
    the index of the best (see tiebreak) interval containing each query value,
    or -1 if not present in any interval.

    Parameters
    ----------
    vals : (sequence of) pdarray(int, uint, float)
        Values to search for in intervals. If multiple arrays, each "row" is an item.
    intervals : 2-tuple of (sequences of) pdarrays
        Non-overlapping, half-open intervals, as a tuple of
        (lower_bounds_inclusive, upper_bounds_exclusive)
        Must have same dtype(s) as vals.
    tiebreak : (optional) pdarray, numeric
        When a value is present in more than one interval, the interval with the
        lowest tiebreak value will be chosen. If no tiebreak is given, the
        first containing interval will be chosen.
    hierarchical: boolean
        When True, sequences of pdarrays will be treated as components specifying
        a single dimension (i.e. hierarchical)
        When False, sequences of pdarrays will be specifying multi-dimensional intervals

    Returns
    -------
    idx : pdarray(int64)
        Index of interval containing each query value, or -1 if not found

    Notes
    -----
    The return idx satisfies the following condition:
        present = idx > -1
        ((intervals[0][idx[present]] <= vals[present]) &
         (intervals[1][idx[present]] >= vals[present])).all()

    Examples
    --------
    >>> import arkouda as ak
    >>> starts = (ak.array([0, 5]), ak.array([0, 11]))
    >>> ends = (ak.array([5, 9]), ak.array([10, 20]))
    >>> vals = (ak.array([0, 0, 2, 5, 5, 6, 6, 9]), ak.array([0, 20, 1, 5, 15, 0, 12, 30]))
    >>> ak.search_intervals(vals, (starts, ends), hierarchical=False)
    array([0 -1 0 0 1 -1 1 -1])
    >>> ak.search_intervals(vals, (starts, ends))
    array([0 0 0 0 1 1 1 -1])
    >>> bi_starts = ak.bigint_from_uint_arrays([ak.cast(a, ak.uint64) for a in starts])
    >>> bi_ends = ak.bigint_from_uint_arrays([ak.cast(a, ak.uint64) for a in ends])
    >>> bi_vals = ak.bigint_from_uint_arrays([ak.cast(a, ak.uint64) for a in vals])
    >>> bi_starts, bi_ends, bi_vals
    (array([0 92233720368547758091]),
    array([92233720368547758090 166020696663385964564]),
    array([0 20 36893488147419103233 92233720368547758085 92233720368547758095
    110680464442257309696 110680464442257309708 166020696663385964574]))
    >>> ak.search_intervals(bi_vals, (bi_starts, bi_ends))
    array([0 0 0 0 1 1 1 -1])

    """
    from arkouda.pandas.join import gen_ranges

    if len(intervals) != 2:
        raise ValueError("intervals must be 2-tuple of (lower_bound_inclusive, upper_bounds_inclusive)")

    low, high = intervals
    singleton = isinstance(vals, pdarray)
    lowsize = low.size if isinstance(low, pdarray) else low[0].size

    if tiebreak is not None and (not isinstance(tiebreak, pdarray) or tiebreak.size != lowsize):
        raise TypeError("Tiebreak must be pdarray of same size as number of intervals")

    if singleton:
        # argument validation for pdarray
        if vals.dtype not in (akint64, akuint64, akfloat64, bigint):
            raise TypeError("arguments must be numeric arrays")

        if not isinstance(low, pdarray) or not isinstance(high, pdarray):
            raise TypeError("intervals must be same objtype as vals")

        # validate the dtypes
        if vals.dtype != low.dtype or vals.dtype != high.dtype:
            raise TypeError(
                f"vals and intervals must all have the same dtype. "
                f"Found {low.dtype}, {high.dtype}, and {vals.dtype}"
            )

        if vals.dtype == bigint:
            return search_intervals(
                vals.bigint_to_uint_arrays(),
                (low.bigint_to_uint_arrays(), high.bigint_to_uint_arrays()),
                tiebreak=tiebreak,
                hierarchical=True,
            )

        # verify lower and upper bounds are same length
        if low.size != high.size:
            raise ValueError("Lower and upper bound arrays must be same size")
        # verify upper bounds are greater than lower bounds
        if not (high >= low).all():
            raise ValueError("Upper bounds must be greater than lower bounds")
        if not low.is_sorted():
            raise ValueError("Intervals must be sorted in ascending order")

        not_overlapping = (low[1:] > high[:-1]).all()
        valsize = vals.size
        perm = argsort(concatenate((low, vals, high)))
    else:
        # argument validation for multi-array
        if not isinstance(low, Sequence) or not isinstance(high, Sequence):
            raise TypeError("intervals must be same objtype as vals")
        if len({len(low), len(high), len(vals)}) != 1:
            raise TypeError("Multi-array arguments must have same number of arrays")
        if (
            any(not isinstance(v, pdarray) for v in vals)
            or any(not isinstance(lo, pdarray) for lo in low)
            or any(not isinstance(hi, pdarray) for hi in high)
        ):
            raise TypeError("All elements of Multi-array arguments must be pdarrays")

        valsize = {v.size for v in vals}
        lowsize = {lo.size for lo in low}
        highsize = {hi.size for hi in high}
        if len(valsize) != 1 or len(lowsize) != 1 or len(highsize) != 1:
            raise TypeError("Multi-array arguments must be non-empty and have equal-length arrays")

        # validate the dtypes
        valtypes = np.array([v.dtype for v in vals])
        lowtypes = np.array([lo.dtype for lo in low])
        hightypes = np.array([hi.dtype for hi in high])
        if (valtypes != lowtypes).any() or (valtypes != hightypes).any():
            raise TypeError("Values and intervals must have matching dtypes")
        for t in valtypes:
            if t not in (akint64, akuint64, akfloat64):
                raise TypeError("arguments must be numeric arrays")

        valsize = valsize.pop()
        lowsize = lowsize.pop()
        highsize = highsize.pop()
        # verify lower and upper bounds are same length
        if lowsize != highsize:
            raise ValueError("Lower and upper bound arrays must be same size")
        # verify upper bounds are greater than lower bounds
        if hierarchical:
            if not is_cosorted(low):
                raise ValueError("Intervals must be sorted in ascending order")
            bounds_okay = True
            checked = zeros(lowsize, dtype=bool)
            needtocheck = ones(lowsize, dtype=bool)
            for lo, hi in zip(low, high):
                if (needtocheck & (hi < lo)).any():
                    bounds_okay = False
                    break
                checked |= needtocheck & (lo < hi)
                if checked.all():
                    bounds_okay = True
                    break
                needtocheck &= lo == hi
            # check non_overlapping
            left = high[0][:-1]
            right = low[0][1:]
            not_overlapping = True
            if (left <= right).any():
                not_overlapping = False
            else:
                boundary = left != right
                for lo, hi in zip(low[1:], high[1:]):
                    left = hi[:-1]
                    right = lo[1:]
                    _ = left <= right
                    if not (_ | boundary).all():
                        not_overlapping = False
                        break
                    boundary = boundary | (left != right)
        else:
            bounds_okay = all((hi >= lo).all() for hi, lo in zip(high, low))

        if not bounds_okay:
            raise ValueError("Upper bounds must be greater than lower bounds")

        perm = coargsort([concatenate((lo, va, hi)) for lo, va, hi in zip(low, vals, high)])

    if singleton or (isinstance(vals, Sequence) and hierarchical):
        # Index of interval containing each unique value (initialized to -1: not found)
        containing_interval = -ones(valsize, dtype=akint64)
        # iperm is the indices of the original values in the sorted array
        iperm = argsort(perm)  # aku.invert_permutation(perm)
        boundary = valsize + lowsize
        # indices of the lower bounds in the sorted array
        starts = iperm[:lowsize]
        # indices of the upper bounds in the sorted array
        ends = iperm[boundary:]
        # which lower/upper bound pairs have any indices between them?
        valid = ends > starts + 1
        if valid.sum() > 0:
            # pranges is all the indices in sorted array that fall between a lower and an uppper bound
            segs, pranges = gen_ranges(starts[valid] + 1, ends[valid])
            # matches are the indices of those items in the original array
            matches = perm[pranges]
            # integer indices of each interval containing a hit
            hit_idx = arange(valid.size)[valid]
            # broadcast interval index out to matches
            match_interval_idx = broadcast(segs, hit_idx, matches.size)
            # make sure not to include any of the bounds themselves
            valid_match = (matches >= lowsize) & (matches < boundary)
            match_interval_idx = match_interval_idx[valid_match]
            # indices of unique values found (translated from concat keys)
            uval_idx = matches[valid_match] - lowsize
            if not_overlapping:
                # set index of containing interval for uvals that were found
                containing_interval[uval_idx] = match_interval_idx[valid_match]
            else:
                # break ties for values contained in more than one interval
                by_item = GroupBy(uval_idx)
                if tiebreak is None:
                    best_match_idx = by_item.permutation[by_item.segments]
                else:
                    _, best_match_idx = by_item.argmin(tiebreak[match_interval_idx])
                # set index of containing interval for uvals that were found
                containing_interval[by_item.unique_keys] = match_interval_idx[best_match_idx]
        return containing_interval
    elif isinstance(vals, Sequence):
        # Index of interval containing each unique value (initialized to -1: not found)
        containing_interval = -ones(valsize, dtype=akint64)
        perm = [argsort(concatenate((lo, v, hi))) for lo, v, hi in zip(low, vals, high)]
        # iperm is the indices of the original values in the sorted array
        iperm = [argsort(p) for p in perm]  # aku.invert_permutation(perm)
        boundary = valsize + lowsize
        # indices of the lower bounds in the sorted array
        starts = [ip[:lowsize] for ip in iperm]
        # indices of the upper bounds in the sorted array
        ends = [ip[boundary:] for ip in iperm]
        # which lower/upper bound pairs have any indices between them?
        # take the logical AND of all elements in list
        valid = functools.reduce(lambda x, y: x & y, [e > s + 1 for e, s in zip(ends, starts)])
        if valid.sum() > 0:
            # pranges is all the indices in sorted array that fall between a lower and
            # an uppper bound for each dimension
            segs, pranges = zip(*[gen_ranges(s[valid] + 1, e[valid]) for s, e in zip(starts, ends)])
            # matches are the indices of those items in the original array
            matches = [pm[pr] for pm, pr in zip(perm, pranges)]
            # integer indices of each one-dimensional interval containing a hit
            hit_idx = arange(valid.size)[valid]
            # broadcast 1-d interval index out to matches
            match_interval_idx = [broadcast(s, hit_idx, m.size) for s, m in zip(segs, matches)]
            # make sure not to include any of the bounds themselves
            valid_match = [(m >= lowsize) & (m < boundary) for m in matches]
            # indices of values found (translated from concat keys) in 1-d intervals
            uval_idx = [m[vm] - lowsize for m, vm in zip(matches, valid_match)]
            # now go from 1-d to full dimensionality
            # for each interval, intersect the hits from its 1-d projections
            # do this by concatenating all the projections and grouping on the id of the hit
            # and the interval and looking for hits that cover all dimensions
            all_uval_idx = concatenate(uval_idx, ordered=False)
            all_match_interval_idx = concatenate(
                [mi[vm] for mi, vm in zip(match_interval_idx, valid_match)], ordered=False
            )
            by_val_interval = GroupBy([all_uval_idx, all_match_interval_idx])
            # a true hit happens when a value is contained in all of an interval's 1-d projections
            is_a_hit = by_val_interval.size()[1] == len(low)
            # indices of the true hits and their containing intervals
            val_hits, interval_hits = [x[is_a_hit] for x in by_val_interval.unique_keys]
            # a value might be found in more than one interval, so we need to break ties
            by_val = GroupBy(val_hits)
            if tiebreak is None:
                best_match_idx = by_val.permutation[by_val.segments]
            else:
                _, best_match_idx = by_val.argmin(tiebreak[interval_hits])
            # set index of best containing interval for values that were found
            containing_interval[by_val.unique_keys] = interval_hits[best_match_idx]
            return containing_interval
    else:
        raise TypeError("arguments must be numeric pdarrays or a sequence of numeric pdarrays")


def is_cosorted(arrays):
    """
    Return True iff the arrays are cosorted.

    Return True iff the arrays are cosorted, i.e., if the arrays were columns in a table
    then the rows are sorted.

    Parameters
    ----------
    arrays : list-like of pdarrays
        Arrays to check for cosortedness

    Returns
    -------
    bool
        True iff arrays are cosorted.

    Raises
    ------
    ValueError
        Raised if arrays are not the same length
    TypeError
        Raised if arrays is not a list-like of pdarrays

    """
    if not isinstance(arrays, Sequence) or not all(isinstance(array, pdarray) for array in arrays):
        raise TypeError("Input must be a list-like of pdarrays")

    # check for equal length
    if len({array.size for array in arrays}) > 1:
        raise ValueError("Arrays must all be same length")

    # fail fast if the first array isn't sorted
    if not arrays[0].is_sorted():
        return False

    # initialize the array to track boundary
    boundary = arrays[0][:-1] != arrays[0][1:]
    for array in arrays[1:]:
        left = array[:-1]
        right = array[1:]
        _ = left <= right
        if not (_ | boundary).all():
            return False
        boundary = boundary | (left != right)
    return True


def interval_lookup(keys, values, arguments, fillvalue=-1, tiebreak=None, hierarchical=False):
    """
    Apply a function defined over intervals to an array of arguments.

    Parameters
    ----------
    keys : 2-tuple of (sequences of) pdarrays
        Tuple of closed intervals expressed as (lower_bounds_inclusive, upper_bounds_inclusive).
        Must have same dtype(s) as vals.
    values : pdarray
        Function value to return for each entry in keys.
    arguments : (sequences of) pdarray
        Values to search for in intervals. If multiple arrays, each "row" is an item.
    fillvalue : scalar
        Default value to return when argument is not in any interval.
    tiebreak : (optional) pdarray, numeric
        When an argument is present in more than one key interval, the interval with the
        lowest tiebreak value will be chosen. If no tiebreak is given, the
        first valid key interval will be chosen.

    Returns
    -------
    pdarray
        Value of function corresponding to the keys interval
        containing each argument, or fillvalue if argument not
        in any interval.

    """
    if isinstance(values, Categorical):
        codes = interval_lookup(keys, values.codes, arguments, fillvalue=values._NAcode)
        return Categorical.from_codes(codes, values.categories, NAvalue=values.NAvalue)
    idx = search_intervals(arguments, keys, tiebreak=tiebreak, hierarchical=hierarchical)
    arguments_size = arguments.size if isinstance(arguments, pdarray) else arguments[0].size
    res = zeros(arguments_size, dtype=values.dtype)
    if fillvalue is not None:
        res.fill(fillvalue)
    found = idx > -1
    res[found] = values[idx[found]]
    return res
