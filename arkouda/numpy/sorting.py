from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal, Sequence, TypeVar, Union
from typing import cast
from typing import cast as type_cast

from typeguard import check_type, typechecked

from arkouda.numpy.dtypes import bigint, dtype, float64, int64, int_scalars, uint64
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import array, zeros
from arkouda.numpy.strings import Strings


if TYPE_CHECKING:
    from arkouda.client import generic_msg
    from arkouda.pandas.categorical import Categorical
else:
    generic_msg = TypeVar("generic_msg")
    Categorical = TypeVar("Categorical")


numeric_dtypes = {dtype(int64), dtype(uint64), dtype(float64)}

__all__ = ["argsort", "coargsort", "sort", "SortingAlgorithm", "searchsorted"]

SortingAlgorithm = Enum("SortingAlgorithm", ["RadixSortLSD", "TwoArrayRadixSort"])


def argsort(
    pda: Union[pdarray, Strings, Categorical],
    algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
    axis: int_scalars = 0,
    ascending: bool = True,
) -> pdarray:
    """
    Return the permutation (indices) that sorts the array.

    Parameters
    ----------
    pda : pdarray, Strings, or Categorical
        The array to sort (supported: int64, uint64, float64 for pdarray).
    algorithm : SortingAlgorithm, default SortingAlgorithm.RadixSortLSD
        The algorithm to use for sorting.
    axis : int, default 0
        Axis to sort along. Negative values are normalized against the array rank.
        For 1D types (Strings, Categorical), must be 0.
    ascending : bool, default True
        Sort order.

    Returns
    -------
    pdarray
        Indices such that ``pda[indices]`` is sorted.

    Raises
    ------
    TypeError
        If `pda` is not a pdarray, Strings, or Categorical.
    ValueError
        If `axis` is out of bounds.

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
    >>> ak.argsort(a, ascending=False)
    array([1 6 0 8 7 2 4 5 3 9])

    """
    from arkouda.numpy.dtypes import int64
    from arkouda.numpy.util import _integer_axis_validation
    from arkouda.pandas.categorical import Categorical

    check_type("pda", value=pda, expected_type=Union[pdarray, Strings, Categorical])

    ndim = pda.ndim
    valid, axis_ = _integer_axis_validation(axis, ndim)
    if not valid:
        raise IndexError(f"{axis} is not a valid axis for array of rank {ndim}")

    size = pda.size
    if size == 0:
        return zeros(0, dtype=int64)
    check_type("pda", value=pda, expected_type=Union[pdarray, Strings, Categorical])

    # Categorical / Strings (always 1D; axis must be 0)
    if isinstance(pda, Categorical):
        if axis != 0:
            raise ValueError("Categorical argsort only supports axis=0")
        return cast(Categorical, pda).argsort(algorithm=algorithm, ascending=ascending)

    if isinstance(pda, Strings):
        if axis != 0:
            raise ValueError("Strings argsort only supports axis=0")
        return cast(Strings, pda).argsort(algorithm=algorithm, ascending=ascending)

    # pdarray
    if isinstance(pda, pdarray):
        perm = cast(pdarray, pda).argsort(algorithm=algorithm, axis=axis, ascending=ascending)
        return perm

    raise TypeError(f"ak.argsort only supports pdarray, Strings, and Categorical, not {type(pda)}")


def coargsort(
    arrays: Sequence[Union[Strings, pdarray, Categorical]],
    algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
    ascending: bool = True,
) -> pdarray:
    """
    Return the permutation that groups the rows (left-to-right), if the
    input arrays are treated as columns. The permutation sorts numeric
    columns, but not Strings or Categoricals — those are grouped, not ordered.

    Parameters
    ----------
    arrays : Sequence of Strings, pdarray, or Categorical
        The columns (int64, uint64, float64, Strings, or Categorical) to sort by row.
    algorithm : SortingAlgorithm, default=SortingAlgorithm.RadixSortLSD
        The algorithm to be used for sorting the arrays.
    ascending : bool, default=True
        Whether to sort in ascending order. Ignored when arrays have ndim > 1.

    Returns
    -------
    pdarray
        The indices that permute the rows into grouped order.

    Raises
    ------
    ValueError
        If the inputs are not all the same size or not valid array types.

    See Also
    --------
    argsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive. Starts with the
    last array and moves forward.

    For Strings, sorting is based on a hash. This ensures grouping of identical strings,
    but not lexicographic order. For Categoricals, sorting is based on the internal codes.

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
    from arkouda.client import generic_msg
    from arkouda.numpy import cast as akcast
    from arkouda.pandas.categorical import Categorical

    check_type("arrays", value=arrays, expected_type=Sequence[Union[pdarray, Strings, Categorical]])

    size: int_scalars = -1
    anames, atypes, expanded_arrays = [], [], []
    max_dim = 1

    for a in arrays:
        if hasattr(a, "ndim"):
            from numpy import maximum

            max_dim = maximum(a.ndim, max_dim)

        if isinstance(a, pdarray):
            if a.dtype == bigint:
                expanded_arrays.extend(a.bigint_to_uint_arrays())
            elif a.dtype == bool:
                expanded_arrays.append(type_cast(pdarray, akcast(a, "int")))
            else:
                expanded_arrays.append(a)
        else:
            expanded_arrays.append(type_cast(pdarray, a))

    for a in expanded_arrays:
        if isinstance(a, pdarray):
            anames.append(a.name)
            atypes.append(a.objType)
        elif isinstance(a, Categorical):
            anames.append(a.codes.name)
            atypes.append(a.objType)
        elif isinstance(a, Strings):
            anames.append(a.entry.name)
            atypes.append(a.objType)
        else:
            raise ValueError("Each array must be a pdarray, Strings, or Categorical")

        if size == -1:
            size = a.size
        elif size != a.size:
            raise ValueError("All arrays must have the same size")

    if size == 0:
        dtype = int if isinstance(arrays[0], (Strings, Categorical)) else arrays[0].dtype
        return zeros(0, dtype=dtype)

    repMsg = generic_msg(
        cmd="coargsort",
        args={
            "algoName": algorithm.name,
            "nstr": len(expanded_arrays),
            "arr_names": anames,
            "arr_types": atypes,
        },
    )

    sorted_array = create_pdarray(cast(str, repMsg))

    if ascending or max_dim > 1:
        return sorted_array
    else:
        from arkouda.numpy.manipulation_functions import flip

        return flip(sorted_array)


@typechecked
def sort(
    pda: pdarray,
    algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
    axis: int_scalars = -1,
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
    from arkouda.client import generic_msg
    from arkouda.numpy.util import _integer_axis_validation

    valid, axis_ = _integer_axis_validation(axis, pda.ndim)
    if not valid:
        raise IndexError(f"{axis} is not a valid axis for array of rank {pda.ndim}")

    if pda.dtype == bigint:
        return pda[coargsort(pda.bigint_to_uint_arrays(), algorithm)]
    if pda.dtype not in numeric_dtypes:
        raise ValueError(f"ak.sort supports int64, uint64, or float64, not {pda.dtype}")
    if pda.size == 0:
        return zeros(0, dtype=pda.dtype)
    repMsg = generic_msg(
        cmd=f"sort<{pda.dtype.name},{pda.ndim}>",
        args={"alg": algorithm.name, "array": pda, "axis": axis_},
    )
    return create_pdarray(cast(str, repMsg))


@typechecked
def searchsorted(
    a: pdarray,
    v: Union[int_scalars, float64, bigint, pdarray],
    side: Literal["left", "right"] = "left",
    x2_sorted: bool = False,
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
    x2_sorted : bool, default=False
        If True, assumes that `v` (x2) is already sorted in ascending order. This can improve performance
        for large, sorted search arrays. If False, no assumption is made about the order of `v`.

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
    >>> v_sorted = ak.array([-10, 12, 13, 20])
    >>> ak.searchsorted(a, v_sorted, x2_sorted=True)
    array([0 1 2 5])
    """
    from arkouda.client import generic_msg

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
            "x2Sorted": x2_sorted,
        },
    )

    out = create_pdarray(cast(str, repMsg))

    if scalar_input:
        return int(out[0])
    return out
