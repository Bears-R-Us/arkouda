from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.pdarraycreation import zeros

__all__ = ["argsort", "coargsort", "local_argsort"]

def argsort(pda):
    """
    Return the permutation that sorts the array.
    
    Parameters
    ----------
    pda : pdarray
        The array to sort (int64 or float64)

    Returns
    -------
    pdarray, int64
        The indices such that ``pda[indices]`` is sorted

    See Also
    --------
    coargsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive.

    Examples
    --------
    >>> a = ak.randint(0, 10, 10)
    >>> perm = ak.argsort(a)
    >>> a[perm]
    array([0, 1, 1, 3, 4, 5, 7, 8, 8, 9])
    """
    if isinstance(pda, pdarray):
        if pda.size == 0:
            return zeros(0, dtype=int64)
        repMsg = generic_msg("argsort {}".format(pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def coargsort(arrays):
    """
    Return the permutation that sorts the rows (left-to-right), if the
    input arrays are treated as columns.
    
    Parameters
    ----------
    arrays : iterable of pdarray
        The columns (int64 or float64) to sort by row

    Returns
    -------
    pdarray, int64
        The indices that permute the rows to sorted order

    See Also
    --------
    argsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive. Starts with the
    last array and moves forward.

    Examples
    --------
    >>> a = ak.array([0, 1, 0, 1])
    >>> b = ak.array([1, 1, 0, 0])
    >>> perm = ak.coargsort([a, b])
    >>> perm
    array([2, 0, 3, 1])
    >>> a[perm]
    array([0, 0, 1, 1])
    >>> b[perm]
    array([0, 1, 0, 1])
    """
    size = -1
    for a in arrays:
        if not isinstance(a, pdarray):
            raise ValueError("Argument must be an iterable of pdarrays")
        if size == -1:
            size = a.size
        elif size != a.size:
            raise ValueError("All pdarrays must have same size")
    if size == 0:
        return zeros(0, dtype=int64)
    repMsg = generic_msg("coargsort {} {}".format(len(arrays), ' '.join([a.name for a in arrays])))
    return create_pdarray(repMsg)

def local_argsort(pda):
    if isinstance(pda, pdarray):
        if pda.size == 0:
            return zeros(0, dtype=int64)
        repMsg = generic_msg("localArgsort {}".format(pda.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
