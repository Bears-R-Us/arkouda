import numpy as np

from arkouda.strings import Strings
from arkouda.pdarrayclass import pdarray
from arkouda.pdarraycreation import arange
from arkouda.pdarraysetops import concatenate
from arkouda.categorical import Categorical
from arkouda.groupbyclass import GroupBy

def unsqueeze(p):
    if isinstance(p, pdarray) or isinstance(p, Strings) or isinstance(p, Categorical):
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
    g = GroupBy(vals)
    uniqueInds = arange(g.unique_keys.size)
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
    c = concatenate(args)
    inds = zero_up(c)
    pos = 0
    ret = []
    for arg in args:
        ret.append(inds[pos:pos+arg.size])
        pos += arg.size
    return ret