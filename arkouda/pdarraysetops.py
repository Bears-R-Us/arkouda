from arkouda.client import generic_msg, verbose
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.pdarraycreation import zeros, array
from arkouda.sorting import argsort
from arkouda.strings import Strings

global verbose

__all__ = ["newsetdiff1d", "newSetxor1d","newUnion1d", "unique", "in1d", "concatenate", "union1d", "intersect1d",
           "setdiff1d", "setxor1d"]
global verbose

def unique(pda, return_counts=False):
    """
    Find the unique elements of an array.

    Returns the unique elements of an array, sorted if the values are integers. 
    There is an optional output in addition to the unique elements: the number 
    of times each unique value comes up in the input array.

    Parameters
    ----------
    pda : pdarray or Strings or Categorical
        Input array.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `pda`.

    Returns
    -------
    unique : pdarray or Strings
        The unique values. If input dtype is int64, return values will be sorted.
    unique_counts : pdarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    Notes
    -----
    For integer arrays, this function checks to see whether `pda` is sorted and, if so,
    whether it is already unique. This step can save considerable computation.
    Otherwise, this function will sort `pda`. For 

    Examples
    --------
    >>> A = ak.array([3, 2, 1, 1, 2, 3])
    >>> ak.unique(A)
    array([1, 2, 3])
    """
    if hasattr(pda, 'unique'):
        return pda.unique()
    elif isinstance(pda, pdarray):
        repMsg = generic_msg("unique {} {} {}".format(pda.objtype, pda.name, return_counts))
        if return_counts:
            vc = repMsg.split("+")
            if verbose: print(vc)
            return create_pdarray(vc[0]), create_pdarray(vc[1])
        else:
            return create_pdarray(repMsg)
    elif isinstance(pda, Strings):
        name = '{}+{}'.format(pda.offsets.name, pda.bytes.name)
        repMsg = generic_msg("unique {} {} {}".format(pda.objtype, name, return_counts))
        vc = repMsg.split('+')
        if verbose: print(vc)
        if return_counts:
            return Strings(vc[0], vc[1]), create_pdarray(vc[2])
        else:
            return Strings(vc[0], vc[1])
    else:
        raise TypeError("must be pdarray or Strings {}".format(pda))

def in1d(pda1, pda2, invert=False):
    """
    Test whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as `pda1` that is True
    where an element of `pda1` is in `pda2` and False otherwise.

    Parameters
    ----------
    pda1 : pdarray or Strings or Categorical
        Input array.
    pda2 : pdarray or Strings
        The values against which to test each value of `pda1`. Must be the 
        same type as `pda1`.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of `pda1` is in `pda2` and True otherwise).
        Default is False. ``ak.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``~ak.in1d(a, b)``.

    Returns
    -------
    pdarray, bool
        The values `pda1[in1d]` are in `pda2`.

    See Also
    --------
    unique, intersect1d, union1d

    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is logically
    equivalent to ``ak.array([item in b for item in a])``, but is much
    faster and scales to arbitrarily large ``a``.
    """
    if hasattr(pda1, 'in1d'):
        return pda1.in1d(pda2)
    elif isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        repMsg = generic_msg("in1d {} {} {}".format(pda1.name, pda2.name, invert))
        return create_pdarray(repMsg)
    elif isinstance(pda1, Strings) and isinstance(pda2, Strings):
        repMsg = generic_msg("segmentedIn1d {} {} {} {} {} {} {}".format(pda1.objtype,
                                                                         pda1.offsets.name,
                                                                         pda1.bytes.name,
                                                                         pda2.objtype,
                                                                         pda2.offsets.name,
                                                                         pda2.bytes.name,
                                                                         invert))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

def concatenate(arrays):
    """
    Concatenate an iterable of ``pdarray`` objects into one ``pdarray``.

    Parameters
    ----------
    arrays : iterable of ``pdarray`` or Strings or Categorical
        The arrays to concatenate. Must all have same dtype.

    Returns
    -------
    pdarray
        Single array containing all values, in original order

    Examples
    --------
    >>> ak.concatenate([ak.array([1, 2, 3]), ak.array([4, 5, 6])])
    array([1, 2, 3, 4, 5, 6])
    """
    size = 0
    objtype = None
    dtype = None
    names = []
    if len(arrays) < 1:
        raise ValueError("concatenate called on empty iterable")
    if len(arrays) == 1:
        return arrays[0]
    if hasattr(arrays[0], 'concatenate'):
        return arrays[0].concatenate(arrays[1:])
    for a in arrays:
        if not isinstance(a, pdarray) and not isinstance(a, Strings):
            raise ValueError("Argument must be an iterable of pdarrays or Strings")
        if objtype == None:
            objtype = a.objtype
        if objtype == "pdarray":
            if dtype == None:
                dtype = a.dtype
            elif dtype != a.dtype:
                raise ValueError("All pdarrays must have same dtype")
            names.append(a.name)
        elif objtype == "str":
            names.append('{}+{}'.format(a.offsets.name, a.bytes.name))
        else:
            raise NotImplementedError("concatenate not implemented for object type {}".format(objtype))
        size += a.size
    if size == 0:
        if objtype == "pdarray":
            return zeros_like(arrays[0])
        else:
            return arrays[0]
    repMsg = generic_msg("concatenate {} {} {}".format(len(arrays), objtype, ' '.join(names)))
    if objtype == "pdarray":
        return create_pdarray(repMsg)
    elif objtype == "str":
        return Strings(*(repMsg.split('+')))

def newUnion1d(pda1, pda2):
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda2 # union is pda2
        if pda2.size == 0:
            return pda1 # union is pda1
        
        repMsg = generic_msg("newUnion1d {} {}".format(pda1.name, pda2.name))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))
    
# (A1 | A2) Set Union: elements are in one or the other or both
def union1d(pda1, pda2):
    """
    Find the union of two arrays.

    Return the unique, sorted array of values that are in either of the two
    input arrays.

    Parameters
    ----------
    pda1 : pdarray
        Input array
    pda2 : pdarray
        Input array

    Returns
    -------
    pdarray
        Unique, sorted union of the input arrays.

    See Also
    --------
    intersect1d, unique

    Examples
    --------
    >>> ak.union1d([-1, 0, 1], [-2, 0, 2])
    array([-2, -1,  0,  1,  2])
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda2 # union is pda2
        if pda2.size == 0:
            return pda1 # union is pda1
        return unique(concatenate((unique(pda1), unique(pda2))))
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

# (A1 & A2) Set Intersection: elements have to be in both arrays
def intersect1d(pda1, pda2, assume_unique=False):
    """
    Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    pda1 : pdarray
        Input array
    pda2 : pdarray
        Input array
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray
        Sorted 1D array of common and unique elements.

    See Also
    --------
    unique, union1d

    Examples
    --------
    >>> ak.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda1 # nothing in the intersection
        if pda2.size == 0:
            return pda2 # nothing in the intersection
        if not assume_unique:
            pda1 = unique(pda1)
            pda2 = unique(pda2)
        aux = concatenate((pda1, pda2))
        aux_sort_indices = argsort(aux)
        aux = aux[aux_sort_indices]
        mask = aux[1:] == aux[:-1]
        int1d = aux[:-1][mask]
        return int1d
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

# (A1 - A2) Set Difference: elements have to be in first array but not second
def setdiff1d(pda1, pda2, assume_unique=False):
    """
    Find the set difference of two arrays.

    Return the sorted, unique values in `pda1` that are not in `pda2`.

    Parameters
    ----------
    pda1 : pdarray
        Input array.
    pda2 : pdarray
        Input comparison array.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray
        Sorted 1D array of values in `pda1` that are not in `pda2`.

    See Also
    --------
    unique, setxor1d

    Examples
    --------
    >>> a = ak.array([1, 2, 3, 2, 4, 1])
    >>> b = ak.array([3, 4, 5, 6])
    >>> ak.setdiff1d(a, b)
    array([1, 2])
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda1 # return a zero length pdarray
        if pda2.size == 0:
            return pda1 # subtracting nothing return orig pdarray
        if not assume_unique:
            pda1 = unique(pda1)
            pda2 = unique(pda2)
        return pda1[in1d(pda1, pda2, invert=True)]
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

def newsetdiff1d(pda1, pda2, assume_unique=False):
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda1 # return a zero length pdarray
        if pda2.size == 0:
            return pda1 # subtracting nothing return orig pdarray
        repMsg = generic_msg("newSetdiff1d {} {} {}".format(pda1.name, pda2.name, assume_unique))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

# (A1 ^ A2) Set Symmetric Difference: elements are not in the intersection
def setxor1d(pda1, pda2, assume_unique=False):
    """
    Find the set exclusive-or (symmetric difference) of two arrays.

    Return the sorted, unique values that are in only one (not both) of the
    input arrays.

    Parameters
    ----------
    pda1 : pdarray
        Input array.
    pda2 : pdarray
        Input array.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray
        Sorted 1D array of unique values that are in only one of the input
        arrays.

    Examples
    --------
    >>> a = ak.array([1, 2, 3, 2, 4])
    >>> b = ak.array([2, 3, 5, 7, 5])
    >>> ak.setxor1d(a,b)
    array([1, 4, 5, 7])
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda2 # return other pdarray if pda1 is empty
        if pda2.size == 0:
            return pda1 # return other pdarray if pda2 is empty
        if not assume_unique:
            pda1 = unique(pda1)
            pda2 = unique(pda2)
        aux = concatenate((pda1, pda2))
        aux_sort_indices = argsort(aux)
        aux = aux[aux_sort_indices]
        flag = concatenate((array([True]), aux[1:] != aux[:-1], array([True])))
        return aux[flag[1:] & flag[:-1]]
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))

def newSetxor1d(pda1, pda2, assume_unique=False):
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda2 # return other pdarray if pda1 is empty
        if pda2.size == 0:
            return pda1 # return other pdarray if pda2 is empty
        
        repMsg = generic_msg("newSetxor1d {} {} {}".format(pda1.name, pda2.name, assume_unique))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {} or {}".format(pda1,pda2))
