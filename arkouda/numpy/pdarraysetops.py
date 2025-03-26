from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar, Union, cast

import numpy as np
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.client_dtypes import BitVector
from arkouda.groupbyclass import GroupBy, groupable, groupable_element_type, unique
from arkouda.logger import getArkoudaLogger
from arkouda.numpy.dtypes import bigint
from arkouda.numpy.dtypes import bool_ as akbool
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import uint64 as akuint64
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import array, ones, zeros, zeros_like
from arkouda.numpy.sorting import argsort
from arkouda.numpy.strings import Strings

if TYPE_CHECKING:
    from arkouda.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")

__all__ = ["in1d", "concatenate", "union1d", "intersect1d", "setdiff1d", "setxor1d", "indexof1d"]

logger = getArkoudaLogger(name="pdarraysetops")

# TODO: combine in1d and _in1d_single into one function


def _in1d_single(
    pda1: Union[pdarray, Strings, "Categorical"],
    pda2: Union[pdarray, Strings, "Categorical"],
    invert: bool = False,
) -> pdarray:
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
        if both are pdarrays and either has rank > 1, or if either is a
        Categorical object or if invert is not a bool
    RuntimeError
        Raised if the dtype of either array is not supported

    See Also
    --------
    arkouda.groupbyclass.unique, intersect1d, union1d

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
    array([False True False])

    >>> ak.in1d(ak.array(['one','two']),ak.array(['two', 'three','four','five']))
    array([False True])
    """
    from arkouda.categorical import Categorical as Categorical_

    if isinstance(pda1, pdarray) or isinstance(pda1, Strings) or isinstance(pda1, Categorical_):
        # While isinstance(thing, type) can be called on a tuple of types,
        # this causes an issue with mypy for unknown reasons.
        if pda1.size == 0:
            return zeros(0, dtype=akbool)
    if isinstance(pda2, pdarray) or isinstance(pda2, Strings) or isinstance(pda2, Categorical_):
        if pda2.size == 0:
            return zeros(pda1.size, dtype=akbool)
    if hasattr(pda1, "categories"):
        x = cast(Categorical_, pda1).in1d(pda2)
        return x if not invert else ~x
    elif isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.ndim > 1 or pda2.ndim > 1 :
            raise TypeError("in1d does not support multi-dim inputs")
        if pda1.dtype == bigint and pda2.dtype == bigint:
            return in1d(pda1.bigint_to_uint_arrays(), pda2.bigint_to_uint_arrays(), invert=invert)
        repMsg = generic_msg(
            cmd="in1d",
            args={
                "pda1": pda1,
                "pda2": pda2,
                "invert": invert,
            },
        )
        return create_pdarray(repMsg)
    elif isinstance(pda1, Strings) and isinstance(pda2, Strings):
        repMsg = generic_msg(
            cmd="segmentedIn1d",
            args={
                "objType": pda1.objType,
                "obj": pda1.entry,
                "otherType": pda2.objType,
                "other": pda2.entry,
                "invert": invert,
            },
        )
        return create_pdarray(cast(str, repMsg))
    else:
        raise TypeError("Both pda1 and pda2 must be pdarray, Strings, or Categorical")


@typechecked
def in1d(
    A: groupable,
    B: groupable,
    assume_unique: bool = False,
    symmetric: bool = False,
    invert: bool = False,
) -> groupable:
    """
    Test whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as `A` that is True
    where an element of `A` is in `B` and False otherwise.

    Supports multi-level, i.e. test if rows of a are in the set of rows of b.
    But note that multi-dimensional pdarrays are not supported.

    Parameters
    ----------
    A : list of pdarrays, pdarray, Strings, or Categorical
        Entries will be tested for membership in B
    B : list of pdarrays, pdarray, Strings, or Categorical
        The set of elements in which to test membership
    assume_unique : bool, optional, defaults to False
        If true, assume rows of a and b are each unique and sorted.
        By default, sort and unique them explicitly.
    symmetric: bool, optional, defaults to False
        Return in1d(A, B), in1d(B, A) when A and B are single items.
    invert : bool, optional, defaults to False
        If True, the values in the returned array are inverted (that is,
        False where an element of `A` is in `B` and True otherwise).
        Default is False. ``ak.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``~ak.in1d(a, b)``.

    Returns
    -------
    pdarray, bool
        True for each row in a that is contained in b

    Raises
    ------
    TypeError
        Raised if either A or B is not a pdarray, Strings, or Categorical
        object, or if both are pdarrays and either has rank > 1,
        or if invert is not a bool
    RuntimeError
        Raised if the dtype of either array is not supported

    Examples
    --------
    >>> ak.in1d(ak.array([-1, 0, 1]), ak.array([-2, 0, 2]))
    array([False True False])

    >>> ak.in1d(ak.array(['one','two']),ak.array(['two', 'three','four','five']))
    array([False True])

    See Also
    --------
    arkouda.groupbyclass.unique, intersect1d, union1d

    Notes
    ------
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is logically
    equivalent to ``ak.array([item in b for item in a])``, but is much
    faster and scales to arbitrarily large ``a``.

    ak.in1d is not supported for bool or float64 pdarrays
    """
    from arkouda.alignment import NonUniqueError
    from arkouda.categorical import Categorical as Categorical_

    if isinstance(A, (pdarray, Strings, Categorical_)):
        if isinstance(A, (Strings, Categorical_)) and not isinstance(B, (Strings, Categorical_)):
            raise TypeError("Arguments must have compatible types, Strings/Categorical")
        elif isinstance(A, pdarray) and not isinstance(B, pdarray):
            raise TypeError("If A is pdarray, B must also be pdarray")
        elif isinstance(B, (pdarray, Strings, Categorical_)):
            if symmetric:
                return _in1d_single(A, B), _in1d_single(B, A, invert)
            return _in1d_single(A, B, invert)
        else:
            raise TypeError(
                "Inputs should both be Union[pdarray, Strings, Categorical] or both be "
                "Sequence[pdarray, Strings, Categorical]."
                "  (Do not mix and match.)"
            )
    atypes = np.array([ai.dtype for ai in A])
    btypes = np.array([bi.dtype for bi in B])
    if not (atypes == btypes).all():
        raise TypeError("Array dtypes of arguments must match")
    if not assume_unique:
        ag = GroupBy(A)
        ua = ag.unique_keys
        bg = GroupBy(B)
        ub = bg.unique_keys
    else:
        ua = A
        ub = B
    # Key for deinterleaving result
    isa = concatenate((ones(ua[0].size, dtype=akbool), zeros(ub[0].size, dtype=akbool)), ordered=False)
    c = [concatenate(x, ordered=False) for x in zip(ua, ub)]
    g = GroupBy(c)
    k, ct = g.size()
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
            return truth[isa], truth[~isa] if not invert else ~truth[isa], ~truth[~isa]
        else:
            return truth[isa] if not invert else ~truth[isa]
    else:
        # If didn't start unique, first need to deinterleave into ua domain,
        # then broadcast to a domain
        atruth = ag.broadcast(truth[isa], permute=True)
        if symmetric:
            btruth = bg.broadcast(truth[~isa], permute=True)
            return atruth, btruth if not invert else ~atruth, ~btruth
        else:
            return atruth if not invert else ~atruth


def in1dmulti(a, b, assume_unique=False, symmetric=False):
    """
    Alias for in1d to maintain backwards compatibility.
    Calls in1d.
    """
    return in1d(a, b, assume_unique=assume_unique, symmetric=symmetric)


def indexof1d(query: groupable, space: groupable) -> pdarray:
    """
    Return indices of query items in a search list of items. Items not found will be excluded.
    When duplicate terms are present in search space return indices of all occurrences.

    Parameters
    ----------
    query : (sequence of) pdarray or Strings or Categorical
        The items to search for. If multiple arrays, each "row" is an item.
    space : (sequence of) pdarray or Strings or Categorical
        The set of items in which to search. Must have same shape/dtype as query.

    Returns
    -------
    indices : pdarray, int64
        For each item in query that is found in space, its index in space.

    Notes
    -----
    This is an alias of
    `ak.find(query, space, all_occurrences=True, remove_missing=True).values`

    Examples
    --------
    >>> select_from = ak.arange(10)
    >>> query = select_from[ak.randint(0, select_from.size, 20, seed=10)]
    >>> space = select_from[ak.randint(0, select_from.size, 20, seed=11)]

    remove some values to ensure that query has entries
    which don't appear in space

    >>> space = space[arr2 != 9]
    >>> space = space[arr2 != 3]

    >>> ak.indexof1d(query, space)
    array([0 4 1 3 10 2 6 12 13 5 7 8 9 14 5 7 11 15 5 7 0 4])

    Raises
    ------
    TypeError
        Raised if either `query` or `space` is not a pdarray, Strings, or
        Categorical object
    RuntimeError
        Raised if the dtype of either array is not supported
    """
    from arkouda.categorical import Categorical as Categorical_

    if isinstance(query, (pdarray, Strings, Categorical_)):
        if isinstance(query, (Strings, Categorical_)) and not isinstance(space, (Strings, Categorical_)):
            raise TypeError("Arguments must have compatible types, Strings/Categorical")
        elif isinstance(query, pdarray) and not isinstance(space, pdarray):
            raise TypeError("If query is pdarray, space must also be pdarray")

    from arkouda.alignment import find as akfind

    found = akfind(query, space, all_occurrences=True, remove_missing=True)
    return found if isinstance(found, pdarray) else found.values


# fmt: off
@typechecked
def concatenate(
    arrays: Sequence[Union[pdarray, Strings, "Categorical", ]],
    ordered: bool = True,
) -> Union[pdarray, Strings, Categorical, Sequence[Categorical]]:
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
        Raised if arrays is empty or if pdarrays have differing dtypes
    TypeError
        Raised if arrays is not a pdarrays or Strings python Sequence such as a
        list or tuple
    RuntimeError
        Raised if any array elements are dtypes for which
        concatenate has not been implemented.

    Examples
    --------
    >>> ak.concatenate([ak.array([1, 2, 3]), ak.array([4, 5, 6])])
    array([1 2 3 4 5 6])

    >>> ak.concatenate([ak.array([True,False,True]),ak.array([False,True,True])])
    array([True False True False True True])

    >>> ak.concatenate([ak.array(['one','two']),ak.array(['three','four','five'])])
    array(['one', 'two', 'three', 'four', 'five'])

    """
    from arkouda.categorical import Categorical as Categorical_
    from arkouda.numpy.dtypes import int_scalars
    from arkouda.numpy.util import get_callback

    size: int_scalars = 0
    objtype = None
    dtype = None
    names = []
    if ordered:
        mode = "append"
    else:
        mode = "interleave"
    if len(arrays) < 1:
        raise ValueError("concatenate called on empty iterable")
    callback = get_callback(list(arrays)[0])
    if len(arrays) == 1:
        # return object as it's original type
        return callback(arrays[0])

    types = {type(x) for x in arrays}
    if len(types) != 1:
        raise TypeError(f"Items must all have same type: {types}")

    if isinstance(arrays[0], BitVector):
        # everything should be a BitVector because all have the same type, but do isinstance for mypy
        widths = {x.width for x in arrays if isinstance(x, BitVector)}
        revs = {x.reverse for x in arrays if isinstance(x, BitVector)}
        if len(widths) != 1 or len(revs) != 1:
            raise TypeError("BitVectors must all have same width and direction")

    if hasattr(arrays[0], "concatenate"):
        return cast(
            Sequence[Categorical_],
            cast(Categorical_, arrays[0]).concatenate(
                cast(Sequence[Categorical_], arrays[1:]), ordered=ordered
            ),
        )
    for a in arrays:
        if not isinstance(a, pdarray) and not isinstance(a, Strings):
            raise TypeError("arrays must be an iterable of pdarrays or Strings")
        if objtype is None:
            objtype = a.objType
        if objtype == pdarray.objType:
            if dtype is None:
                dtype = a.dtype
            elif dtype != a.dtype:
                raise ValueError("All pdarrays must have same dtype")
            names.append(cast(pdarray, a).name)
        elif objtype == Strings.objType:
            names.append(cast(Strings, a).entry.name)
        else:
            raise NotImplementedError(f"concatenate not implemented for object type {objtype}")
        size += a.size
    if size == 0:
        if objtype == "pdarray":
            return callback(zeros_like(cast(pdarray, arrays[0])))
        else:
            return arrays[0]

    repMsg = generic_msg(
        cmd="concatenate",
        args={
            "nstr": len(arrays),
            "objType": objtype,
            "mode": mode,
            "names": names,
        })
    if objtype == pdarray.objType:
        return callback(create_pdarray(cast(str, repMsg)))
    elif objtype == Strings.objType:
        # ConcatenateMsg returns created attrib(name)+created nbytes=123
        return Strings.from_return_msg(cast(str, repMsg))
    else:
        raise TypeError("arrays must be an array of pdarray or Strings objects")
# fmt:on


def multiarray_setop_validation(
    pda1: Sequence[groupable_element_type], pda2: Sequence[groupable_element_type]
):
    from arkouda.categorical import Categorical as Categorical_

    if len(pda1) != len(pda2):
        raise ValueError("multi-array setops require same number of arrays in arguments.")
    size1 = {x.size for x in pda1}
    if len(size1) > 1:
        raise ValueError("multi-array setops require arrays in pda1 be the same size.")
    size2 = {x.size for x in pda2}
    if len(size2) > 1:
        raise ValueError("multi-array setops require arrays in pda2 be the same size.")
    atypes = [akint64 if isinstance(x, Categorical_) else x.dtype for x in pda1]
    btypes = [akint64 if isinstance(x, Categorical_) else x.dtype for x in pda2]
    if not atypes == btypes:
        raise TypeError("Array dtypes of arguments must match")


# (A | B) Set Union: elements are in one or the other or both
@typechecked
def union1d(
    A: groupable,
    B: groupable,
) -> groupable:
    """
    Find the union of two arrays/List of Arrays.

    Return the unique, sorted array of values that are in either
    of the two input arrays.

    Parameters
    ----------
    A : list of pdarrays, pdarray, Strings, or Categorical
    B : list of pdarrays, pdarray, Strings, or Categorical

    Returns
    -------
    pdarray/groupable
        Unique, sorted union of the input arrays.

    Raises
    ------
    TypeError
        Raised if either A or B is not a groupable
    RuntimeError
        Raised if the dtype of either input is not supported

    See Also
    --------
    intersect1d, arkouda.groupbyclass.unique

    Examples
    --------

    1D Example

    >>> ak.union1d(ak.array([-1, 0, 1]), ak.array([-2, 0, 2]))
    array([-2 -1 0 1 2])

    Multi-Array Example

    >>> a = ak.arange(1, 6)
    >>> b = ak.array([1, 5, 3, 4, 2])
    >>> c = ak.array([1, 4, 3, 2, 5])
    >>> d = ak.array([1, 2, 3, 5, 4])
    >>> multia = [a, a, a]
    >>> multib = [b, c, d]
    >>> ak.union1d(multia, multib)
    [array([1 2 2 3 4 4 5 5]), array([1 2 5 3 2 4 4 5]), array([1 2 4 3 5 4 2 5])]
    """
    from arkouda.categorical import Categorical as Categorical_

    if (
        isinstance(A, (pdarray, Strings, Categorical_))
        and isinstance(B, (pdarray, Strings, Categorical_))
        and type(A) is type(B)
    ):
        if A.size == 0:
            return B  # union is B
        if B.size == 0:
            return A  # union is A
        if (
            A.dtype == int
            and B.dtype == int
            or (A.dtype == akuint64 and B.dtype == akuint64)
        ):
            repMsg = generic_msg(cmd="union1d", args={"arg1": A, "arg2": B})
            return cast(pdarray, create_pdarray(repMsg))
        x = cast(
            pdarray, unique(cast(pdarray, concatenate((unique(A), unique(B)), ordered=False)))
        )
        return x[argsort(x)]
    elif isinstance(A, Sequence) and isinstance(B, Sequence):
        multiarray_setop_validation(A, B)
        ag = GroupBy(A)
        ua = ag.unique_keys
        bg = GroupBy(B)
        ub = bg.unique_keys

        c = [concatenate(x, ordered=False) for x in zip(ua, ub)]
        g = GroupBy(c)
        k, ct = g.size()
        return list(k)
    else:
        raise TypeError(
            f"Both A and B must be pdarray, List, or Tuple. Received {type(A)} and {type(B)}"
        )


# (A & B) Set Intersection: elements have to be in both arrays
@typechecked
def intersect1d(
    A: groupable, B: groupable, assume_unique: bool = False
) -> Union[pdarray, groupable]:
    """
    Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    A : list of pdarrays, pdarray, Strings, or Categorical
    B : list of pdarrays, pdarray, Strings, or Categorical
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
        Raised if either A or B is not a groupable
    RuntimeError
        Raised if the dtype of either pdarray is not supported

    See Also
    --------
    arkouda.groupbyclass.unique, union1d

    Examples
    --------

    1D Example

    >>> ak.intersect1d(ak.array([1, 3, 4, 3]), ak.array([3, 1, 2, 1]))
    array([1 3])

    Multi-Array Example

    >>> a = ak.arange(5)
    >>> b = ak.array([1, 5, 3, 4, 2])
    >>> c = ak.array([1, 4, 3, 2, 5])
    >>> d = ak.array([1, 2, 3, 5, 4])
    >>> multia = [a, a, a]
    >>> multib = [b, c, d]
    >>> ak.intersect1d(multia, multib)
    [array([1 3]), array([1 3]), array([1 3])]
    """
    from arkouda.categorical import Categorical as Categorical_

    if (
        isinstance(A, (pdarray, Strings, Categorical_))
        and isinstance(B, (pdarray, Strings, Categorical_))
        and type(A) is type(B)
    ):
        if A.size == 0:
            return A  # nothing in the intersection
        if B.size == 0:
            return B  # nothing in the intersection
        if (A.dtype == int and B.dtype == int) or (
            A.dtype == akuint64 and B.dtype == akuint64
        ):
            repMsg = generic_msg(
                cmd="intersect1d", args={"arg1": A, "arg2": B, "assume_unique": assume_unique}
            )
            return create_pdarray(cast(str, repMsg))
        if not assume_unique:
            A = cast(pdarray, unique(A))
            B = cast(pdarray, unique(B))
        aux = concatenate((A, B), ordered=False)
        aux_sort_indices = argsort(aux)
        aux = aux[aux_sort_indices]
        mask = aux[1:] == aux[:-1]
        int1d = aux[:-1][mask]
        return int1d
    elif (isinstance(A, list) or isinstance(A, tuple)) and (
        isinstance(B, list) or isinstance(B, tuple)
    ):
        multiarray_setop_validation(A, B)

        if not assume_unique:
            ag = GroupBy(A)
            ua = ag.unique_keys
            bg = GroupBy(B)
            ub = bg.unique_keys
        else:
            ua = A
            ub = B

        # Key for deinterleaving result
        isa = concatenate(
            (ones(ua[0].size, dtype=akbool), zeros(ub[0].size, dtype=akbool)), ordered=False
        )
        c = [concatenate(x, ordered=False) for x in zip(ua, ub)]
        g = GroupBy(c)
        if assume_unique:
            # need to verify uniqueness, otherwise answer will be wrong
            if (g.sum(isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but first argument is not unique")
            if (g.sum(~isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but second argument is not unique")
        k, ct = g.size()
        in_union = ct == 2
        return [x[in_union] for x in k]
    else:
        raise TypeError(
            f"Both A and B must be pdarray, List, or Tuple. Received {type(A)} and {type(B)}"
        )


# (A - B) Set Difference: elements have to be in first array but not second
@typechecked
def setdiff1d(
    A: groupable, B: groupable, assume_unique: bool = False
) -> Union[pdarray, groupable]:
    """
    Find the set difference of two arrays.

    Return the sorted, unique values in `A` that are not in `B`.

    Parameters
    ----------
    A : list of pdarrays, pdarray, Strings, or Categorical
    B : list of pdarrays, pdarray, Strings, or Categorical
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.

    Returns
    -------
    pdarray/groupable
        Sorted 1D array/List of sorted pdarrays of values in `A` that are not in `B`.

    Raises
    ------
    TypeError
        Raised if either A or B is not a pdarray
    RuntimeError
        Raised if the dtype of either pdarray is not supported

    See Also
    --------
    arkouda.groupbyclass.unique, setxor1d

    Notes
    -----
    ak.setdiff1d is not supported for bool pdarrays

    Examples
    --------
    >>> a = ak.array([1, 2, 3, 2, 4, 1])
    >>> b = ak.array([3, 4, 5, 6])
    >>> ak.setdiff1d(a, b)
    array([1 2])

    Multi-Array Example

    >>> a = ak.arange(1, 6)
    >>> b = ak.array([1, 5, 3, 4, 2])
    >>> c = ak.array([1, 4, 3, 2, 5])
    >>> d = ak.array([1, 2, 3, 5, 4])
    >>> multia = [a, a, a]
    >>> multib = [b, c, d]
    >>> ak.setdiff1d(multia, multib)
    [array([2 4 5]), array([2 4 5]), array([2 4 5])]
    """
    from arkouda.categorical import Categorical as Categorical_

    if (
        isinstance(A, (pdarray, Strings, Categorical_))
        and isinstance(B, (pdarray, Strings, Categorical_))
        and type(A) is type(B)
    ):
        if A.size == 0:
            return A  # return a zero length pdarray
        if B.size == 0:
            return A  # subtracting nothing return orig pdarray
        if (A.dtype == int and B.dtype == int) or (
            A.dtype == akuint64 and B.dtype == akuint64
        ):
            repMsg = generic_msg(
                cmd="setdiff1d", args={"arg1": A, "arg2": B, "assume_unique": assume_unique}
            )
            return create_pdarray(cast(str, repMsg))
        if not assume_unique:
            A = cast(pdarray, unique(A))
            B = cast(pdarray, unique(B))
        x = A[in1d(A, B, invert=True)]
        return x[argsort(x)]
    elif (isinstance(A, list) or isinstance(A, tuple)) and (
        isinstance(B, list) or isinstance(B, tuple)
    ):
        multiarray_setop_validation(A, B)

        if not assume_unique:
            ag = GroupBy(A)
            ua = ag.unique_keys
            bg = GroupBy(B)
            ub = bg.unique_keys
        else:
            ua = A
            ub = B

        # Key for deinterleaving result
        isa = concatenate(
            (ones(ua[0].size, dtype=akbool), zeros(ub[0].size, dtype=akbool)), ordered=False
        )
        c = [concatenate(x, ordered=False) for x in zip(ua, ub)]
        g = GroupBy(c)
        if assume_unique:
            # need to verify uniqueness, otherwise answer will be wrong
            if (g.sum(isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but first argument is not unique")
            if (g.sum(~isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but second argument is not unique")
        k, ct = g.size()
        truth = g.broadcast(ct == 1, permute=True)
        atruth = truth[isa]
        return [x[atruth] for x in ua]
    else:
        raise TypeError(
            f"Both A and B must be pdarray, List, or Tuple. Received {type(A)} and {type(B)}"
        )


# (A1 ^ A2) Set Symmetric Difference: elements are not in the intersection
@typechecked
def setxor1d(A: groupable, B: groupable, assume_unique: bool = False) -> Union[pdarray, groupable]:
    """
    Find the set exclusive-or (symmetric difference) of two arrays.

    Return the sorted, unique values that are in only one (not both) of the
    input arrays.

    Parameters
    ----------
    A : list of pdarrays, pdarray, Strings, or Categorical
    B : list of pdarrays, pdarray, Strings, or Categorical
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
        Raised if either A or B is not a groupable
    RuntimeError
        Raised if the dtype of either pdarray is not supported

    Examples
    --------
    >>> a = ak.array([1, 2, 3, 2, 4])
    >>> b = ak.array([2, 3, 5, 7, 5])
    >>> ak.setxor1d(a,b)
    array([1 4 5 7])

    Multi-Array Example

    >>> a = ak.arange(1, 6)
    >>> b = ak.array([1, 5, 3, 4, 2])
    >>> c = ak.array([1, 4, 3, 2, 5])
    >>> d = ak.array([1, 2, 3, 5, 4])
    >>> multia = [a, a, a]
    >>> multib = [b, c, d]
    >>> ak.setxor1d(multia, multib)
    [array([2 2 4 4 5 5]), array([2 5 2 4 4 5]), array([2 4 5 4 2 5])]
    """
    from arkouda.categorical import Categorical as Categorical_

    if (
        isinstance(A, (pdarray, Strings, Categorical_))
        and isinstance(B, (pdarray, Strings, Categorical_))
        and type(A) is type(B)
    ):
        if A.size == 0:
            return B  # return other pdarray if A is empty
        if B.size == 0:
            return A  # return other pdarray if B is empty
        if (A.dtype == int and B.dtype == int) or (
            A.dtype == akuint64 and B.dtype == akuint64
        ):
            repMsg = generic_msg(
                cmd="setxor1d", args={"arg1": A, "arg2": B, "assume_unique": assume_unique}
            )
            return create_pdarray(cast(str, repMsg))
        if not assume_unique:
            A = cast(pdarray, unique(A))
            B = cast(pdarray, unique(B))
        aux = concatenate((A, B), ordered=False)
        aux_sort_indices = argsort(aux)
        aux = aux[aux_sort_indices]
        flag = concatenate((array([True]), aux[1:] != aux[:-1], array([True])))
        return aux[flag[1:] & flag[:-1]]
    elif (isinstance(A, list) or isinstance(A, tuple)) and (
        isinstance(B, list) or isinstance(B, tuple)
    ):
        multiarray_setop_validation(A, B)

        if not assume_unique:
            ag = GroupBy(A)
            ua = ag.unique_keys
            bg = GroupBy(B)
            ub = bg.unique_keys
        else:
            ua = A
            ub = B

        # Key for deinterleaving result
        isa = concatenate(
            (ones(ua[0].size, dtype=akbool), zeros(ub[0].size, dtype=akbool)), ordered=False
        )
        c = [concatenate(x, ordered=False) for x in zip(ua, ub)]
        g = GroupBy(c)
        if assume_unique:
            # need to verify uniqueness, otherwise answer will be wrong
            if (g.sum(isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but first argument is not unique")
            if (g.sum(~isa)[1] > 1).any():
                raise ValueError("Called with assume_unique=True, but second argument is not unique")
        k, ct = g.size()
        single = ct == 1
        return [x[single] for x in k]
    else:
        raise TypeError(
            f"Both A and B must be pdarray, List, or Tuple. Received {type(A)} and {type(B)}"
        )
