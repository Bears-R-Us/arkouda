import arkouda as ak
import numpy as np
from functools import reduce
from operator import or_

def unsqueeze(p):
    if isinstance(p, ak.pdarray) or isinstance(p, ak.Strings) or isinstance(p, ak.Categorical):
        return [p]
    else:
        return p

def zero_up(vals):
    """ Map an array of sparse values to 0-up indices.
    Parameters
    ----------
    vals : pdarray
        Array to map to dense index

    Returns
    -------
    aligned : pdarray
        Array with values replaced by 0-up indices
    """
    g = ak.GroupBy(vals)
    uniqueInds = ak.arange(g.unique_keys.size)
    idinds = g.broadcast(uniqueInds, permute=True)
    return idinds

def align(*args):
    """ Map multiple arrays of sparse identifiers to a common 0-up index.

    Parameters
    ----------
    *args : pdarrays
        Arrays to map to dense index

    Returns
    -------
    aligned : list of pdarrays
        Arrays with values replaced by 0-up indices
    """
    c = ak.concatenate(args)
    inds = zero_up(c)
    pos = 0
    ret = []
    for arg in args:
        ret.append(inds[pos:pos+arg.size])
        pos += arg.size
    return ret

def right_align(left, right):
    """ Map two arrays of sparse values to the 0-up index set implied by the right array, discarding values from left that do not appear in right.

    Parameters
    ----------
    left : pdarray
        Left-hand identifiers
    right : pdarray
        Right-hand identifiers that define the index

    Returns
    -------
    keep : pdarray, bool
        Logical index of left-hand values that survived
    aligned : (pdarray, pdarray)
        Left and right arrays with values replaced by 0-up indices
    """
    uright = ak.unique(right)
    keep = ak.in1d(left, uright)
    fleft = left[keep]
    return keep, align(fleft, right)

def left_align(left, right):
    """ Map two arrays of sparse identifiers to the 0-up index set implied by the left array, discarding values from right that do not appear in left.
    """
    return right_align(right, left)

class NonUniqueError(ValueError):
    pass

def in1dmulti(a, b, assume_unique=False, symmetric=False):
    """ The multi-level analog of ak.in1d -- test membership of rows of a in the set of rows of b.

    Parameters
    ----------
    a : list of pdarrays
        Rows are elements for which to test membership in b
    b : list of pdarrays
        Rows are elements of the set in which to test membership
    assume_unique : bool
        If true, assume rows of a and b are each unique and sorted. By default, sort and unique them explicitly.

    Returns
    -------
    pdarray, bool
        True for each row in a that is contained in b

    Notes:
        Only works for pdarrays of int64 dtype, Strings, or Categorical
    """
    if isinstance(a, (ak.pdarray, ak.Strings, ak.Categorical)):
        if type(a) != type(b):
            raise TypeError("Arguments must have same type")
        if symmetric:
            return ak.in1d(a, b), ak.in1d(b, a)
        else:
            return ak.in1d(a, b)
    atypes = np.array([ai.dtype for ai in a])
    btypes = np.array([bi.dtype for bi in b])
    if not (atypes == btypes).all():
        raise TypeError("Array dtypes of arguments must match")
    if not assume_unique:
        ag = ak.GroupBy(a)
        ua = ag.unique_keys
        bg = ak.GroupBy(b)
        ub = bg.unique_keys
    else:
        ua = a
        ub = b
    # Key for deinterleaving result
    isa = ak.concatenate((ak.ones(ua[0].size, dtype=ak.bool), ak.zeros(ub[0].size, dtype=ak.bool)), ordered=False)
    c = [ak.concatenate(x, ordered=False) for x in zip(ua, ub)]
    g = ak.GroupBy(c)
    k, ct = g.count()
    if assume_unique:
        # need to verify uniqueness, otherwise answer will be wrong
        if (g.sum(isa)[1] > 1).any():
            raise NonUniqueError("Called with assume_unique=True, but first argument is not unique")
        if (g.sum(~isa)[1] > 1).any():
            raise NonUniqueError("Called with assume_unique=True, but second argument is not unique")
    # Where value appears twice, it is present in both a and b
    # truth = answer in c domain
    truth = g.broadcast(ct == 2, permute=True)
    if assume_unique:
        # Deinterleave truth into a and b domains
        if symmetric:
            return truth[isa], truth[~isa]
        else:
            return truth[isa]
    else:
        # If didn't start unique, first need to deinterleave into ua domain,
        # then broadcast to a domain
        atruth = ag.broadcast(truth[isa], permute=True)
        if symmetric:
            btruth = bg.broadcast(truth[~isa], permute=True)
            return atruth, btruth
        else:
            return atruth


def lookup(keys, values, arguments, fillvalue=-1):
    '''
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
    # Lookup numbers by two-word name 
    >>> keys1 = ak.array(['twenty' for _ in range(5)])
    >>> keys2 = ak.array(['one', 'two', 'three', 'four', 'five'])
    >>> values = ak.array([21, 22, 23, 24, 25])
    >>> args1 = ak.array(['twenty', 'thirty', 'twenty'])
    >>> args2 = ak.array(['four', 'two', 'two'])
    >>> aku.lookup([keys1, keys2], values, [args1, args2])
    array([24, -1, 22])

    # Other direction requires an intermediate index 
    >>> revkeys = values 
    >>> revindices = ak.arange(values.size)
    >>> revargs = ak.array([24, 21, 22])
    >>> idx = aku.lookup(revkeys, revindices, revargs)
    >>> keys1[idx], keys2[idx]
    (array(['twenty', 'twenty', 'twenty']),
    array(['four', 'one', 'two']))
    '''
    # Condense down to unique arguments to query
    g = ak.GroupBy(arguments)
    # scattermask = Args that exist in table
    # gathermask = Elements of keys that are being asked for
    try:
        scattermask, gathermask = in1dmulti(g.unique_keys, keys, assume_unique=True, symmetric=True)
    except NonUniqueError as e:
        raise NonUniqueError("Function keys must be unique.")
    # uvals = Retrieved values corresponding to unique args being queried
    if g.nkeys == 1:
        uvals = ak.zeros(g.unique_keys.size, dtype=values.dtype)
    else:
        uvals = ak.zeros(g.unique_keys[0].size, dtype=values.dtype)
    # Use default value for arguments not present in keys
    if fillvalue is not None:
        uvals.fill(fillvalue)
    # Set values for arguments present in keys
    # Set them to the values corresponding to elements queried
    uvals[scattermask] = values[gathermask]
    # Broadcast return values back to non-unique arguments
    return g.broadcast(uvals, permute=True)

def gen_ranges(starts, ends):
    """ Generate a segmented array of variable-length, contiguous 
    ranges between pairs of start- and end-points.

    Parameters
    ----------
    starts : pdarray, int64
        The start value of each range
    ends : pdarray, int64
        The end value (exclusive) of each range

    Returns
    -------
    segments : pdarray, int64
        The starting index of each range in the resulting array
    ranges : pdarray, int64
        The actual ranges, flattened into a single array
    """
    if starts.size != ends.size:
        raise ValueError("starts and ends must be same size")
    if not ((ends - starts) > 0).all():
        raise ValueError("all ends must be greater than starts")
    lengths = ends - starts
    segs = ak.cumsum(lengths) - lengths
    totlen = lengths.sum()
    slices = ak.ones(totlen, dtype=ak.int64)
    diffs = ak.concatenate((ak.array([starts[0]]), 
                            starts[1:] - starts[:-1] - lengths[:-1] + 1))
    slices[segs] = diffs
    return segs, ak.cumsum(slices)


def in1d_intervals(vals, intervals, symmetric=False, assume_unique=False):
    """
    Test each value for membership in *any* of a set of half-open (pythonic) 
    intervals.

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
    idx = search_intervals(vals, intervals, assume_unique=assume_unique)
    found = idx > -1
    if symmetric:
        containresult = ak.in1d(ak.arange(intervals[0].size), idx)
        return found, containresult
    else:
        return found

def search_intervals(vals, intervals, assume_unique=False):
    """
    Given an array of query vals and non-overlapping, half-open (pythonic) 
    intervals, return the index of the interval containing each query value, 
    or -1 if not present in any interval.

    Parameters
    ----------
    vals : pdarray(int, float)
        Values to search for in intervals
    intervals : 2-tuple of pdarrays
        Non-overlapping, half-open intervals, as a tuple of 
        (lower_bounds_inclusive, upper_bounds_exclusive)
    assume_unique : bool
        If True, assume query vals are unique. Default: False.
        
    Returns
    -------
    idx : pdarray(int64)
        Index of interval containing each query value, or -1 if not found 

    Notes
    -----
    The return idx satisfies the following condition:
        present = idx > -1
        ((intervals[0][idx[present]] <= vals[present]) & (intervals[1][idx[present]] > vals[present])).all()
    """
    if len(intervals) != 2:
        raise ValueError("intervals must be 2-tuple of (lower_bound_inclusive, upper_bounds_exclusive)")
    def check_numeric(x):
        if not (isinstance(x, ak.pdarray) and x.dtype in (ak.int64, ak.float64)):
            raise TypeError("arguments must be numeric arrays")
    check_numeric(vals)
    check_numeric(intervals[0])
    check_numeric(intervals[1])
    low = intervals[0]
    # Convert to closed (inclusive) intervals
    high = intervals[1] - 1
    if low.size != high.size:
        raise ValueError("Lower and upper bound arrays must be same size")
    if not (high >= low).all(): 
        raise ValueError("Upper bounds must be greater than lower bounds")
    if not ak.is_sorted(low):
        raise ValueError("Intervals must be sorted in ascending order")
    if not (low[1:] > high[:-1]).all():
        raise ValueError("Intervals must be non-overlapping")
    if assume_unique:
        uvals = vals
    else:
        g = ak.GroupBy(vals)
        uvals = g.unique_keys
    # Index of interval containing each unique value (initialized to -1: not found)
    containinginterval = -ak.ones(uvals.size, dtype=ak.int64)
    concat = ak.concatenate((low, uvals, high))
    perm = ak.argsort(concat)
    # iperm is the indices of the original values in the sorted array
    iperm = ak.argsort(perm) # aku.invert_permutation(perm)
    boundary = uvals.size+low.size
    # indices of the lower bounds in the sorted array
    starts = iperm[:low.size]
    # indices of the upper bounds in the sorted array
    ends = iperm[boundary:]
    # which lower/upper bound pairs have any indices between them?
    valid = (ends > starts + 1)
    if valid.sum() > 0:
        # pranges is all the indices in sorted array that fall between a lower and an uppper bound
        segs, pranges = gen_ranges(starts[valid]+1, ends[valid])
        # matches are the indices of those items in the original array
        matches = perm[pranges]
        # integer indices of each interval containing a hit
        hitidx = ak.arange(valid.size)[valid]
        # broadcast interval index out to matches
        matchintervalidx = ak.broadcast(segs, hitidx, matches.size)
        # make sure not to include any of the bounds themselves
        validmatch = (matches >= low.size) & (matches < boundary)
        # indices of unique values found (translated from concat keys)
        uvalidx = matches[validmatch] - low.size
        # set index of containing interval for uvals that were found
        containinginterval[uvalidx] = matchintervalidx[validmatch]
    if assume_unique:
        res = containinginterval
    else:
        res = g.broadcast(containinginterval, permute=True)
    return res

def interval_lookup(keys, values, arguments, fillvalue=-1):
    '''
    Apply a function defined over non-overlapping intervals to
    an array of arguments.
    
    Parameters
    ----------
    keys : 2-tuple of pdarray
        Tuple of non-overlapping, half-open intervals expressed
        as (lower_bounds_inclusive, upper_bounds_exclusive)
    values : pdarray
        Function value to return for each entry in keys.
    arguments : pdarray
        Arguments to the function
    fillvalue : scalar
        Default value to return when argument is not in any interval.

    Returns
    -------
    pdarray
        Value of function corresponding to the keys interval
        containing each argument, or fillvalue if argument not
        in any interval.
    '''
    idx = search_intervals(arguments, keys, assume_unique=True)
    res = ak.zeros(arguments.size, dtype=values.dtype)
    if fillvalue is not None:
        res.fill(fillvalue)
    found = idx > -1
    res[found] = values[idx[found]]
    return res
