from arkouda.client import generic_msg, verbose
from arkouda.pdarrayclass import pdarray, create_pdarray

global verbose

__all__ = ["mink"]
global verbose

def mink(pda, k):
    """
    Find the `k` minimum values of an array.

    Returns the smallest k values of an array, sorted

    Parameters
    ----------
    pda : pdarray
        Input array.
    k : integer
        The desired count of minimum values to be returned by the output.

    Returns
    -------
    pdarray, int
        The minimum `k` values from pda

    Notes
    -----
    Currently only works on integers, could be exended to also work for floats.

    Examples
    --------
    >>> A = ak.array([10,5,1,3,7,2,9,0])
    >>> ak.mink(A, 3)
    array([0, 1, 2])
    """
    if isinstance(pda, pdarray):
        if k == 0:
            return []
        if pda.dtype != int or pda.size == 0:
            raise TypeError("must be a non-empty pdarray {} of type int".format(pda))
        repMsg = generic_msg("mink {} {}".format(pda.name, k))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
