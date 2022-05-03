from __future__ import annotations
from typing import cast, Optional, Sequence, Tuple, Union, ForwardRef, List
from typeguard import typechecked
from arkouda.client import generic_msg, get_config
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.pdarraycreation import zeros, zeros_like, array, ones
from arkouda.sorting import argsort
from arkouda.strings import Strings
from arkouda.logger import getArkoudaLogger
from arkouda.dtypes import uint64 as akuint64
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import bool as akbool
from arkouda.groupbyclass import unique, GroupBy, groupable, groupable_element_type

Categorical = ForwardRef('Categorical')

__all__ = ["in1d", "concatenate", "union1d", "intersect1d",
           "setdiff1d", "setxor1d"]

logger = getArkoudaLogger(name='pdarraysetops')

def in1d(pda1: Union[pdarray, Strings, 'Categorical'], pda2: Union[pdarray, Strings, 'Categorical'],  # type: ignore
         invert: bool = False) -> pdarray:  # type: ignore
    """
    Test whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as `pda1` that is True
    where an element of `pda1` is in `pda2` and False otherwise.

    Parameters
    ----------
    pda1 : pdarray or Strings or Categorical
        Input array.
    pda2 : pdarray or Strings or Categorical
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
        
    Raises
    ------
    TypeError
        Raised if either pda1 or pda2 is not a pdarray, Strings, or 
        Categorical object or if invert is not a bool
    RuntimeError
        Raised if the dtype of either array is not supported

    See Also
    --------
    unique, intersect1d, union1d

    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is logically
    equivalent to ``ak.array([item in b for item in a])``, but is much
    faster and scales to arbitrarily large ``a``.
    
    ak.in1d is not supported for bool or float64 pdarrays

    Examples
    --------
    >>> ak.in1d(ak.array([-1, 0, 1]), ak.array([-2, 0, 2]))
    array([False, True, False])    
    
    >>> ak.in1d(ak.array(['one','two']),ak.array(['two', 'three','four','five']))
    array([False, True])
    """
    from arkouda.categorical import Categorical as Categorical_
    from arkouda.dtypes import bool as ak_bool
    if isinstance(pda1, pdarray) or isinstance(pda1, Strings) or isinstance(pda1, Categorical_):
        # While isinstance(thing, type) can be called on a tuple of types, this causes an issue with mypy for unknown reasons.
        if pda1.size == 0:
            return zeros(0, dtype=ak_bool)
    if isinstance(pda2, pdarray) or isinstance(pda2, Strings) or isinstance(pda2, Categorical_):
        if pda2.size == 0:
            return zeros(pda1.size, dtype=ak_bool)
    if hasattr(pda1, 'categories'):
        return cast(Categorical_, pda1).in1d(pda2)
    elif isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        repMsg = generic_msg(cmd="in1d", args="{} {} {}". \
                             format(pda1.name, pda2.name, invert))
        return create_pdarray(repMsg)
    elif isinstance(pda1, Strings) and isinstance(pda2, Strings):
        repMsg = generic_msg(cmd="segmentedIn1d", args="{} {} {} {} {}". \
                             format(pda1.objtype,
                                    pda1.entry.name,
                                    pda2.objtype,
                                    pda2.entry.name,
                                    invert))
        return create_pdarray(cast(str, repMsg))
    else:
        raise TypeError('Both pda1 and pda2 must be pdarray, Strings, or Categorical')


@typechecked
def concatenate(arrays: Sequence[Union[pdarray, Strings, 'Categorical', ]],  # type: ignore
                ordered: bool = True) -> Union[pdarray, Strings, 'Categorical']:  # type: ignore
    """
    Concatenate a list or tuple of ``pdarray`` or ``Strings`` objects into 
    one ``pdarray`` or ``Strings`` object, respectively.

    Parameters
    ----------
    arrays : Sequence[Union[pdarray,Strings,Categorical]]
        The arrays to concatenate. Must all have same dtype.
    ordered : bool
        If True (default), the arrays will be appended in the
        order given. If False, array data may be interleaved
        in blocks, which can greatly improve performance but
        results in non-deterministic ordering of elements.

    Returns
    -------
    Union[pdarray,Strings,Categorical]
        Single pdarray or Strings object containing all values, returned in
        the original order
        
    Raises
    ------
    ValueError
        Raised if arrays is empty or if 1..n pdarrays have
        differing dtypes
    TypeError
        Raised if arrays is not a pdarrays or Strings python Sequence such as a 
        list or tuple
    RuntimeError
        Raised if 1..n array elements are dtypes for which
        concatenate has not been implemented.

    Examples
    --------
    >>> ak.concatenate([ak.array([1, 2, 3]), ak.array([4, 5, 6])])
    array([1, 2, 3, 4, 5, 6])
    
    >>> ak.concatenate([ak.array([True,False,True]),ak.array([False,True,True])])
    array([True, False, True, False, True, True])
    
    >>> ak.concatenate([ak.array(['one','two']),ak.array(['three','four','five'])])
    array(['one', 'two', 'three', 'four', 'five'])

    """
    from arkouda.categorical import Categorical as Categorical_
    from arkouda.dtypes import int_scalars
    size: int_scalars = 0
    objtype = None
    dtype = None
    names = []
    if ordered:
        mode = 'append'
    else:
        mode = 'interleave'
    if len(arrays) < 1:
        raise ValueError("concatenate called on empty iterable")
    if len(arrays) == 1:
        return cast(groupable_element_type, arrays[0])

    if hasattr(arrays[0], 'concatenate'):
        return cast(Sequence[Categorical_],
                    cast(Categorical_,
                         arrays[0]).concatenate(cast(Sequence[Categorical_],
                                                     arrays[1:]), ordered=ordered))
    for a in arrays:
        if not isinstance(a, pdarray) and not isinstance(a, Strings):
            raise TypeError(("arrays must be an iterable of pdarrays"
                             " or Strings"))
        if objtype == None:
            objtype = a.objtype
        if objtype == "pdarray":
            if dtype == None:
                dtype = a.dtype
            elif dtype != a.dtype:
                raise ValueError("All pdarrays must have same dtype")
            names.append(cast(pdarray, a).name)
        elif objtype == "str":
            names.append(cast(Strings, a).entry.name)
        else:
            raise NotImplementedError(("concatenate not implemented " +
                                       "for object type {}".format(objtype)))
        size += a.size
    if size == 0:
        if objtype == "pdarray":
            return zeros_like(cast(pdarray, arrays[0]))
        else:
            return arrays[0]

    repMsg = generic_msg(cmd="concatenate", args="{} {} {} {}". \
                         format(len(arrays), objtype, mode, ' '.join(names)))
    if objtype == "pdarray":
        return create_pdarray(cast(str, repMsg))
    elif objtype == "str":
        # ConcatenateMsg returns created attrib(name)+created nbytes=123
        return Strings.from_return_msg(cast(str, repMsg))
    else:
        raise TypeError('arrays must be an array of pdarray or Strings objects')


def multiarray_setop_validation(pda1: Sequence[groupable_element_type],
                                pda2: Sequence[groupable_element_type]):
    from arkouda.categorical import Categorical as Categorical_
    if len(pda1) != len(pda2):
        raise ValueError("multi-array setops require same number of arrays in arguments.")
    size1 = set([x.size for x in pda1])
    if len(size1) > 1:
        raise ValueError("multi-array setops require arrays in pda1 be the same size.")
    size2 = set([x.size for x in pda2])
    if len(size2) > 1:
        raise ValueError("multi-array setops require arrays in pda2 be the same size.")
    atypes = [akint64 if isinstance(x, Categorical_) else x.dtype for x in pda1]
    btypes = [akint64 if isinstance(x, Categorical_) else x.dtype for x in pda2]
    if not atypes == btypes:
        raise TypeError("Array dtypes of arguments must match")

# (A1 | A2) Set Union: elements are in one or the other or both
@typechecked
def union1d(pda1: Union[pdarray, Sequence[groupable_element_type]],
            pda2: Union[pdarray, Sequence[groupable_element_type]]) -> \
        Union[pdarray, groupable]:
    """
    Find the union of two arrays/List of Arrays.

    Return the unique, sorted array of values that are in either 
    of the two input arrays.

    Parameters
    ----------
    pda1 : pdarray/Sequence[pdarray, Strings, Categorical]
        Input array/Sequence of groupable objects
    pda2 : pdarray/List
        Input array/sequence of groupable objects

    Returns
    -------
    pdarray/groupable
        Unique, sorted union of the input arrays.
        
    Raises
    ------
    TypeError
        Raised if either pda1 or pda2 is not a pdarray
    RuntimeError
        Raised if the dtype of either array is not supported

    See Also
    --------
    intersect1d, unique

    Notes
    -----
    ak.union1d is not supported for bool or float64 pdarrays

    Examples
    --------
    # 1D Example
    >>> ak.union1d(ak.array([-1, 0, 1]), ak.array([-2, 0, 2]))
    array([-2, -1, 0, 1, 2])

    #Multi-Array Example
    >>> a = ak.arange(1, 6)
    >>> b = ak.array([1, 5, 3, 4, 2])
    >>> c = ak.array([1, 4, 3, 2, 5])
    >>> d = ak.array([1, 2, 3, 5, 4])
    >>> multia = [a, a, a]
    >>> multib = [b, c, d]
    >>> ak.union1d(multia, multib)
    [array[1, 2, 2, 3, 4, 4, 5, 5], array[1, 2, 5, 3, 2, 4, 4, 5], array[1, 2, 4, 3, 5, 4, 2, 5]]
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda2  # union is pda2
        if pda2.size == 0:
            return pda1  # union is pda1
        if pda1.dtype == int and pda2.dtype == int or \
                (pda1.dtype == akuint64 and pda2.dtype == akuint64):
            repMsg = generic_msg(cmd="union1d", args="{} {}".
                                 format(pda1.name, pda2.name))
            return cast(pdarray, create_pdarray(repMsg))
        return cast(pdarray,
                    unique(cast(pdarray,
                                concatenate((unique(pda1), unique(pda2)), ordered=False))))  # type: ignore
    elif isinstance(pda1, Sequence) and isinstance(pda2, Sequence):
        multiarray_setop_validation(pda1, pda2)
        ag = GroupBy(pda1)
        ua = ag.unique_keys
        bg = GroupBy(pda2)
        ub = bg.unique_keys

        c = [concatenate(x, ordered=False) for x in zip(ua, ub)]
        g = GroupBy(c)
        k, ct = g.count()
        return k
    else:
        raise TypeError(f'Both pda1 and pda2 must be pdarray, List, or Tuple. Received {type(pda1)} and {type(pda2)}')


# (A1 & A2) Set Intersection: elements have to be in both arrays
@typechecked
def intersect1d(pda1: groupable,
                pda2: groupable,
                assume_unique: bool = False) -> Union[pdarray, groupable]:
    """
    Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    pda1 : pdarray/Sequence[pdarray, Strings, Categorical]
        Input array/Sequence of groupable objects
    pda2 : pdarray/List
        Input array/sequence of groupable objects
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray/groupable
        Sorted 1D array/List of sorted pdarrays of common and unique elements.

    Raises
    ------
    TypeError
        Raised if either pda1 or pda2 is not a pdarray
    RuntimeError
        Raised if the dtype of either pdarray is not supported

    See Also
    --------
    unique, union1d

    Notes
    -----
    ak.intersect1d is not supported for bool or float64 pdarrays

    Examples
    --------
    # 1D Example
    >>> ak.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])

    # Multi-Array Example
    >>> a = ak.arange(5)
    >>> b = ak.array([1, 5, 3, 4, 2])
    >>> c = ak.array([1, 4, 3, 2, 5])
    >>> d = ak.array([1, 2, 3, 5, 4])
    >>> multia = [a, a, a]
    >>> multib = [b, c, d]
    >>> ak.intersect1d(multia, multib)
    [array([1, 3]), array([1, 3]), array([1, 3])]
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda1  # nothing in the intersection
        if pda2.size == 0:
            return pda2  # nothing in the intersection
        if (pda1.dtype == int and pda2.dtype == int) or \
                (pda1.dtype == akuint64 and pda2.dtype == akuint64):
            repMsg = generic_msg(cmd="intersect1d", args="{} {} {}". \
                                 format(pda1.name, pda2.name, assume_unique))
            return create_pdarray(cast(str, repMsg))
        if not assume_unique:
            pda1 = cast(pdarray, unique(pda1))
            pda2 = cast(pdarray, unique(pda2))
        aux = concatenate((pda1, pda2), ordered=False)
        aux_sort_indices = argsort(aux)
        aux = aux[aux_sort_indices]
        mask = aux[1:] == aux[:-1]
        int1d = aux[:-1][mask]
        return int1d
    elif (isinstance(pda1, list) or isinstance(pda1, tuple)) and (isinstance(pda2, list) or isinstance(pda2, tuple)):
        multiarray_setop_validation(pda1, pda2)

        if not assume_unique:
            ag = GroupBy(pda1)
            ua = ag.unique_keys
            bg = GroupBy(pda2)
            ub = bg.unique_keys
        else:
            ua = pda1
            ub = pda2

        # Key for deinterleaving result
        isa = concatenate((ones(ua[0].size, dtype=akbool), zeros(ub[0].size, dtype=akbool)), ordered=False)
        c = [concatenate(x, ordered=False) for x in zip(ua, ub)]
        g = GroupBy(c)
        if assume_unique:
            # need to verify uniqueness, otherwise answer will be wrong
            if (g.sum(isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but first argument is not unique")
            if (g.sum(~isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but second argument is not unique")
        k, ct = g.count()
        in_union = ct == 2
        return [x[in_union] for x in k]
    else:
        raise TypeError(f'Both pda1 and pda2 must be pdarray, List, or Tuple. Received {type(pda1)} and {type(pda2)}')

# (A1 - A2) Set Difference: elements have to be in first array but not second
@typechecked
def setdiff1d(pda1: groupable,
              pda2: groupable,
              assume_unique: bool = False) -> Union[pdarray, groupable]:
    """
    Find the set difference of two arrays.

    Return the sorted, unique values in `pda1` that are not in `pda2`.

    Parameters
    ----------
    pda1 : pdarray/Sequence[pdarray, Strings, Categorical]
        Input array/Sequence of groupable objects
    pda2 : pdarray/List
        Input array/sequence of groupable objects
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray/groupable
        Sorted 1D array/List of sorted pdarrays of values in `pda1` that are not in `pda2`.

    Raises
    ------
    TypeError
        Raised if either pda1 or pda2 is not a pdarray
    RuntimeError
        Raised if the dtype of either pdarray is not supported

    See Also
    --------
    unique, setxor1d

    Notes
    -----
    ak.setdiff1d is not supported for bool or float64 pdarrays

    Examples
    --------
    >>> a = ak.array([1, 2, 3, 2, 4, 1])
    >>> b = ak.array([3, 4, 5, 6])
    >>> ak.setdiff1d(a, b)
    array([1, 2])

    #Multi-Array Example
    >>> a = ak.arange(1, 6)
    >>> b = ak.array([1, 5, 3, 4, 2])
    >>> c = ak.array([1, 4, 3, 2, 5])
    >>> d = ak.array([1, 2, 3, 5, 4])
    >>> multia = [a, a, a]
    >>> multib = [b, c, d]
    >>> ak.setdiff1d(multia, multib)
    [array([2, 4, 5]), array([2, 4, 5]), array([2, 4, 5])]
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda1  # return a zero length pdarray
        if pda2.size == 0:
            return pda1  # subtracting nothing return orig pdarray
        if (pda1.dtype == int and pda2.dtype == int) or \
                (pda1.dtype == akuint64 and pda2.dtype == akuint64):
            repMsg = generic_msg(cmd="setdiff1d", args="{} {} {}". \
                                 format(pda1.name, pda2.name, assume_unique))
            return create_pdarray(cast(str, repMsg))
        if not assume_unique:
            pda1 = cast(pdarray, unique(pda1))
            pda2 = cast(pdarray, unique(pda2))
        return pda1[in1d(pda1, pda2, invert=True)]
    elif (isinstance(pda1, list) or isinstance(pda1, tuple)) and (isinstance(pda2, list) or isinstance(pda2, tuple)):
        multiarray_setop_validation(pda1, pda2)

        if not assume_unique:
            ag = GroupBy(pda1)
            ua = ag.unique_keys
            bg = GroupBy(pda2)
            ub = bg.unique_keys
        else:
            ua = pda1
            ub = pda2

        # Key for deinterleaving result
        isa = concatenate((ones(ua[0].size, dtype=akbool), zeros(ub[0].size, dtype=akbool)), ordered=False)
        c = [concatenate(x, ordered=False) for x in zip(ua, ub)]
        g = GroupBy(c)
        if assume_unique:
            # need to verify uniqueness, otherwise answer will be wrong
            if (g.sum(isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but first argument is not unique")
            if (g.sum(~isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but second argument is not unique")
        k, ct = g.count()
        truth = g.broadcast(ct == 1, permute=True)
        atruth = truth[isa]
        return [x[atruth] for x in ua]
    else:
        raise TypeError(f'Both pda1 and pda2 must be pdarray, List, or Tuple. Received {type(pda1)} and {type(pda2)}')


# (A1 ^ A2) Set Symmetric Difference: elements are not in the intersection
@typechecked
def setxor1d(pda1: groupable,
             pda2: groupable,
             assume_unique: bool = False) -> Union[pdarray, groupable]:
    """
    Find the set exclusive-or (symmetric difference) of two arrays.

    Return the sorted, unique values that are in only one (not both) of the
    input arrays.

    Parameters
    ----------
    pda1 : pdarray/Sequence[pdarray, Strings, Categorical]
        Input array/Sequence of groupable objects
    pda2 : pdarray/List
        Input array/sequence of groupable objects
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray/groupable
        Sorted 1D array/List of sorted pdarrays of unique values that are in only one of the input
        arrays.

    Raises
    ------
    TypeError
        Raised if either pda1 or pda2 is not a pdarray
    RuntimeError
        Raised if the dtype of either pdarray is not supported

    Notes
    -----
    ak.setxor1d is not supported for bool or float64 pdarrays

    Examples
    --------
    >>> a = ak.array([1, 2, 3, 2, 4])
    >>> b = ak.array([2, 3, 5, 7, 5])
    >>> ak.setxor1d(a,b)
    array([1, 4, 5, 7])

    #Multi-Array Example
    >>> a = ak.arange(1, 6)
    >>> b = ak.array([1, 5, 3, 4, 2])
    >>> c = ak.array([1, 4, 3, 2, 5])
    >>> d = ak.array([1, 2, 3, 5, 4])
    >>> multia = [a, a, a]
    >>> multib = [b, c, d]
    >>> ak.setxor1d(multia, multib)
    [array([2, 2, 4, 4, 5, 5]), array([2, 5, 2, 4, 4, 5]), array([2, 4, 5, 4, 2, 5])]
    """
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size == 0:
            return pda2  # return other pdarray if pda1 is empty
        if pda2.size == 0:
            return pda1  # return other pdarray if pda2 is empty
        if (pda1.dtype == int and pda2.dtype == int) or \
                (pda1.dtype == akuint64 and pda2.dtype == akuint64):
            repMsg = generic_msg(cmd="setxor1d", args="{} {} {}". \
                                 format(pda1.name, pda2.name, assume_unique))
            return create_pdarray(cast(str, repMsg))
        if not assume_unique:
            pda1 = cast(pdarray, unique(pda1))
            pda2 = cast(pdarray, unique(pda2))
        aux = concatenate((pda1, pda2), ordered=False)
        aux_sort_indices = argsort(aux)
        aux = aux[aux_sort_indices]
        flag = concatenate((array([True]), aux[1:] != aux[:-1], array([True])))
        return aux[flag[1:] & flag[:-1]]
    elif (isinstance(pda1, list) or isinstance(pda1, tuple)) and (isinstance(pda2, list) or isinstance(pda2, tuple)):
        multiarray_setop_validation(pda1, pda2)

        if not assume_unique:
            ag = GroupBy(pda1)
            ua = ag.unique_keys
            bg = GroupBy(pda2)
            ub = bg.unique_keys
        else:
            ua = pda1
            ub = pda2

        # Key for deinterleaving result
        isa = concatenate((ones(ua[0].size, dtype=akbool), zeros(ub[0].size, dtype=akbool)), ordered=False)
        c = [concatenate(x, ordered=False) for x in zip(ua, ub)]
        g = GroupBy(c)
        if assume_unique:
            # need to verify uniqueness, otherwise answer will be wrong
            if (g.sum(isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but first argument is not unique")
            if (g.sum(~isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but second argument is not unique")
        k, ct = g.count()
        single = ct == 1
        return [x[single] for x in k]
    else:
        raise TypeError(f'Both pda1 and pda2 must be pdarray, List, or Tuple. Received {type(pda1)} and {type(pda2)}')
