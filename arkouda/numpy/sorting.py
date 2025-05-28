from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal, Sequence, TypeVar, Union, cast

import numpy as np
from typeguard import check_type, typechecked

from arkouda.client import generic_msg
from arkouda.numpy.dtypes import (
    bigint,
    bool_,
    dtype,
    float64,
    int64,
    int_scalars,
    uint64,
)
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import array, zeros
from arkouda.numpy.strings import Strings

numeric_dtypes = {dtype(int64), dtype(uint64), dtype(float64)}

__all__ = ["argsort", "coargsort", "sort", "SortingAlgorithm", "searchsorted"]

SortingAlgorithm = Enum("SortingAlgorithm", ["RadixSortLSD", "TwoArrayRadixSort"])

if TYPE_CHECKING:
    from arkouda.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")


def argsort(
    pda: Union[pdarray, Strings, Categorical],
    algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
    axis: int_scalars = 0,
) -> pdarray:
    """
    Return the permutation that sorts the array.

    Parameters
    ----------
    pda : pdarray, Strings, or Categorical
        The array to sort (int64, uint64, or float64)
    algorithm : SortingAlgorithm, default=SortingAlgorithm.RadixSortLSD
        The algorithm to be used for sorting the array.
    axis : int_scalars, default=0
        The axis to sort over.

    Returns
    -------
    pdarray
        The indices such that ``pda[indices]`` is sorted

    Raises
    ------
    TypeError
        Raised if the parameter is other than a pdarray, Strings or Categorical

    See Also
    --------
    coargsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and
    resilient to non-uniformity in data but communication intensive.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.randint(0, 10, 10, seed=1)
    >>> a
    array([7 9 5 1 4 1 8 5 5 0])

    >>> perm = ak.argsort(a)
    >>> a[perm]
    array([0 1 1 4 5 5 5 7 8 9])

    >>> ak.argsort(a, ak.sorting.SortingAlgorithm["RadixSortLSD"])
    array([9 3 5 4 2 7 8 0 6 1])

    >>> ak.argsort(a, ak.sorting.SortingAlgorithm["TwoArrayRadixSort"])
    array([9 3 5 4 2 7 8 0 6 1])
    """
    from arkouda.categorical import Categorical

    ndim = cast(Union[int, np.integer], getattr(pda, "ndim"))

    if axis < -1 or axis > int(ndim):
        raise ValueError(f"Axis must be between -1 and the PD Array's rank ({int(ndim)})")
    if axis == -1:
        axis = int(ndim) - 1

    check_type(argname="argsort", value=pda, expected_type=Union[pdarray, Strings, Categorical])

    if hasattr(pda, "argsort"):
        return cast(Categorical, pda).argsort()
    if pda.size == 0 and hasattr(pda, "dtype"):
        return zeros(0, dtype=pda.dtype)
    if isinstance(pda, pdarray) and pda.dtype == bigint:
        return coargsort(pda.bigint_to_uint_arrays(), algorithm)

    if isinstance(pda, Strings):
        repMsg = generic_msg(
            cmd="argsortStrings",
            args={
                "name": pda.entry.name,
                "algoName": algorithm.name,
            },
        )
    else:
        repMsg = generic_msg(
            cmd=f"argsort<{pda.dtype.name},{pda.ndim}>",
            args={
                "name": pda.name,
                "algoName": algorithm.name,
                "objType": pda.objType,
                "axis": axis,
            },
        )

    return create_pdarray(cast(str, repMsg))


def coargsort(
    arrays: Sequence[Union[Strings, pdarray, Categorical]],
    algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
) -> pdarray:
    """
    Return the permutation that groups the rows (left-to-right), if the
    input arrays are treated as columns. The permutation sorts numeric
    columns, but not strings/Categoricals -- strings/Categoricals are grouped, but not ordered.

    Parameters
    ----------
    arrays : Sequence of Strings, pdarray, or Categorical
        The columns (int64, uint64, float64, Strings, or Categorical) to sort by row
    algorithm : SortingAlgorithm, default=SortingAlgorithm.RadixSortLSD
        The algorithm to be used for sorting the arrays.

    Returns
    -------
    pdarray
        The indices that permute the rows to grouped order

    Raises
    ------
    ValueError
        Raised if the pdarrays are not of the same size or if the parameter
        is not an Iterable containing pdarrays, Strings, or Categoricals

    See Also
    --------
    argsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive. Starts with the
    last array and moves forward. This sort operates directly on numeric types,
    but for Strings, it operates on a hash. Thus, while grouping of equivalent
    strings is guaranteed, lexicographic ordering of the groups is not. For Categoricals,
    coargsort sorts based on Categorical.codes which guarantees grouping of equivalent categories
    but not lexicographic ordering of those groups.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([0, 1, 0, 1])
    >>> b = ak.array([1, 1, 0, 0])
    >>> perm = ak.coargsort([a, b])
    >>> perm
    array([2 0 3 1])
    >>> a[perm]
    array([0 0 1 1])
    >>> b[perm]
    array([0 1 0 1])
    """
    from arkouda.categorical import Categorical
    from arkouda.numpy import cast as akcast

    check_type(
        argname="coargsort", value=arrays, expected_type=Sequence[Union[pdarray, Strings, Categorical]]
    )
    size: int_scalars = -1
    anames = []
    atypes = []
    expanded_arrays = []
    for a in arrays:
        if not isinstance(a, pdarray) or a.dtype not in [bigint, bool_]:
            expanded_arrays.append(a)
        elif a.dtype == bigint:
            expanded_arrays.extend(a.bigint_to_uint_arrays())
        else:
            # cast bool arrays to int
            expanded_arrays.append(akcast(a, "int"))

    for a in expanded_arrays:
        if isinstance(a, pdarray):
            anames.append(a.name)
            atypes.append(a.objType)
        elif isinstance(a, Categorical):
            anames.append(a.codes.name)
            atypes.append(a.objType)
        elif isinstance(a, Strings):
            atypes.append(a.objType)
            anames.append(a.entry.name)
        else:
            raise ValueError("Argument must be an iterable of pdarrays, Strings, or Categoricals")
        if size == -1:
            size = a.size
        elif size != a.size:
            raise ValueError("All pdarrays, Strings, or Categoricals must be of the same size")

    if size == 0:
        return zeros(0, dtype=int if isinstance(arrays[0], (Strings, Categorical)) else arrays[0].dtype)

    repMsg = generic_msg(
        cmd="coargsort",
        args={
            "algoName": algorithm.name,
            "nstr": len(expanded_arrays),
            "arr_names": anames,
            "arr_types": atypes,
        },
    )
    return create_pdarray(cast(str, repMsg))


@typechecked
def sort(
    pda: pdarray, algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD, axis: int_scalars = -1
) -> pdarray:
    """
    Return a sorted copy of the array. Only sorts numeric arrays;
    for Strings, use argsort.

    Parameters
    ----------
    pda : pdarray
        The array to sort (int64, uint64, or float64)
    algorithm : SortingAlgorithm, default=SortingAlgorithm.RadixSortLSD
        The algorithm to be used for sorting the arrays.
    axis : int_scalars, default=-1
        The axis to sort over. Setting to -1 means that it will sort over axis = ndim - 1.

    Returns
    -------
    pdarray
        The sorted copy of pda

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    ValueError
        Raised if sort attempted on a pdarray with an unsupported dtype
        such as bool

    See Also
    --------
    argsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.randint(0, 10, 10, seed=1)
    >>> a
    array([7 9 5 1 4 1 8 5 5 0])
    >>> sorted = ak.sort(a)
    >>> sorted
    array([0 1 1 4 5 5 5 7 8 9])
    """
    if pda.dtype == bigint:
        return pda[coargsort(pda.bigint_to_uint_arrays(), algorithm)]
    if pda.dtype not in numeric_dtypes:
        raise ValueError(f"ak.sort supports int64, uint64, or float64, not {pda.dtype}")
    if pda.size == 0:
        return zeros(0, dtype=pda.dtype)
    repMsg = generic_msg(
        cmd=f"sort<{pda.dtype.name},{pda.ndim}>",
        args={"alg": algorithm.name, "array": pda, "axis": axis},
    )
    return create_pdarray(cast(str, repMsg))


@typechecked
def searchsorted(
    a: pdarray, v: Union[int_scalars, float64, bigint, pdarray], side: Literal["left", "right"] = "left"
) -> Union[int, pdarray]:
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the corresponding
    elements in `v` were inserted before the indices, the order of `a` would be preserved.

    Parameters
    ----------
    a : pdarray
        1-D input array. Must be sorted in ascending order. `sorter` is not currently supported.
    v : int_scalars, float64, bigint, or pdarray
        Values to insert into `a`. Can be a scalar or array-like.
    side : {'left', 'right'}, default='left'
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.

    Returns
    -------
    indices : int or pdarray
        If `v` is an array, returns an array of insertion points with the same shape.
        If `v` is a scalar, returns a single integer index.

    Raises
    ------
    ValueError
        If `a` has more than one dimension.
    TypeError
        If `a` has an unsupported dtype (i.e., not int64, uint64, bigint, or float64).
        If the dtype of `a` and `v` does not match


    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([11, 12, 13, 14, 15])
    >>> ak.searchsorted(a, 13)
    2
    >>> ak.searchsorted(a, 13, side='right')
    3
    >>> v = ak.array([-10, 20, 12, 13])
    >>> ak.searchsorted(a, v)
    array([0 5 1 2])
    """

    if a.ndim > 1:
        raise ValueError(f"a must be one dimensional, but has {a.ndim} dimensions.")
    if a.dtype not in numeric_dtypes and a.dtype != bigint:
        raise TypeError(f"ak.searchsorted supports int64, uint64, bigint, or float64, not {a.dtype}")

    # Normalize v to array
    scalar_input = False
    v_: pdarray
    if isinstance(v, pdarray):
        v_ = v
    else:
        scalar_input = True
        v_ = cast(pdarray, array([v]))

    if a.dtype != v_.dtype:
        raise TypeError(f"The dtype of a ({a.dtype}) and v ({v_.dtype}) must match.")

    repMsg = generic_msg(
        cmd=f"searchSorted<{a.dtype},{a.ndim},{v_.ndim}>",
        args={
            "x1": a,
            "x2": v_,
            "side": side,
        },
    )

    out = create_pdarray(cast(str, repMsg))

    if scalar_input:
        return int(out[0])
    return out
