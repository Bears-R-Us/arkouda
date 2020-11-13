import akutil as aku
import arkouda as ak

def expand(size, segs, vals):
    """ Expand an array with values placed into the indicated segments.

    Parameters
    ----------
    size : ak.pdarray
        The size of the array to be expanded
    segs : ak.pdarray
        The indices where the values should be placed
    vals : ak.pdarray
        The values to be placed in each segment

    Returns
    -------
    pdarray
        The expanded array.

    """
    temp = ak.zeros(size, vals.dtype)
    diffs = ak.concatenate((ak.array([vals[0]]), vals[1:]-vals[:-1]))
    temp[segs] = diffs
    return ak.cumsum(temp)

def invert_permutation(perm):
    """ Find the inverse of a permutation array.

    Parameters
    ----------
    perm : ak.pdarray
        The permutation array.

    Returns
    -------
    ak.array
        The inverse of the permutation array.

    """
    # I think this suffers from overflow errors on large arrays.
    #if perm.sum() != (perm.size * (perm.size -1)) / 2:
    #    raise ValueError("The indicated permutation is invalid.")
    if ak.unique(perm).size != perm.size:
        raise ValueError("The array is not a permutation.")
    return ak.coargsort([perm, ak.arange(0, perm.size)])
