import arkouda as ak

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
    idinds = ak.zeros_like(vals)
    idinds[g.permutation] = g.broadcast(uniqueInds)
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

def in1dmulti(a, b, assume_unique=False):
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
    if not assume_unique:
        ag = ak.GroupBy(a)
        ua = ag.unique_keys
        bg = ak.GroupBy(b)
        ub = bg.unique_keys
    else:
        ua = a
        ub = b
    c = [ak.concatenate(x) for x in zip(ua, ub)]
    g = ak.GroupBy(c)
    k, ct = g.count()
    truth = ak.zeros(c[0].size, dtype=ak.bool)
    truth[g.permutation] = (g.broadcast(1*(ct == 2)) == 1)
    if assume_unique:
        return truth[:a[0].size]
    else:
        truth2 = ak.zeros(a[0].size, dtype=ak.bool)
        truth2[ag.permutation] = (ag.broadcast(1*truth[:ua[0].size]) == 1)
        return truth2
