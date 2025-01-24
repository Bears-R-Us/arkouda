from __future__ import annotations

from enum import Enum
from typing import Sequence, Union, cast

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
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.pdarraycreation import zeros
from arkouda.strings import Strings

numeric_dtypes = {dtype(int64), dtype(uint64), dtype(float64)}

__all__ = ["argsort", "coargsort", "sort", "SortingAlgorithm"]

SortingAlgorithm = Enum("SortingAlgorithm", ["RadixSortLSD", "TwoArrayRadixSort"])


def argsort(
    pda: Union[pdarray, Strings, "Categorical"],  # type: ignore # noqa
    algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
    axis: int_scalars = 0,
) -> pdarray:
    """
    Return the permutation that sorts the array.

    Parameters
    ----------
    pda : pdarray or Strings or Categorical
        The array to sort (int64, uint64, or float64)
    algorithm : SortingAlgorithm
        The algorithm to be used for sorting the array.
    axis : int_scalars
        The axis to sort over.

    Returns
    -------
    pdarray, int64
        The indices such that ``pda[indices]`` is sorted

    Raises
    ------
    TypeError
        Raised if the parameter is other than a pdarray or Strings

    See Also
    --------
    coargsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and
    resilient to non-uniformity in data but communication intensive.

    Examples
    --------
    >>> a = ak.randint(0, 10, 10)
    >>> perm = ak.argsort(a)
    >>> a[perm]
    array([0, 1, 1, 3, 4, 5, 7, 8, 8, 9])

    >>> ak.argsort(a, ak.sorting.SortingAlgorithm["RadixSortLSD"])
    array([0 2 9 6 8 1 3 5 7 4])

    >>> ak.argsort(a, ak.sorting.SortingAlgorithm["TwoArrayRadixSort"])
    array([0 2 9 6 8 1 3 5 7 4])
    """
    from arkouda.categorical import Categorical

    if axis < -1 or axis > pda.ndim:
        raise ValueError(f"Axis must be between -1 and the PD Array's rank ({pda.ndim})")
    if axis == -1:
        axis = pda.ndim - 1

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
    arrays: Sequence[Union[Strings, pdarray, "Categorical"]],  # type: ignore # noqa
    algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
) -> pdarray:
    """
    Return the permutation that groups the rows (left-to-right), if the
    input arrays are treated as columns. The permutation sorts numeric
    columns, but not strings/Categoricals -- strings/Categoricals are grouped, but not ordered.

    Parameters
    ----------
    arrays : Sequence[Union[Strings, pdarray, Categorical]]
        The columns (int64, uint64, float64, Strings, or Categorical) to sort by row

    Returns
    -------
    pdarray, int64
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
def sort(pda: pdarray, algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD, axis=-1) -> pdarray:
    """
    Return a sorted copy of the array. Only sorts numeric arrays;
    for Strings, use argsort.

    Parameters
    ----------
    pda : pdarray or Categorical
        The array to sort (int64, uint64, or float64)

    Returns
    -------
    pdarray, int64, uint64, or float64
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
    >>> a = ak.randint(0, 10, 10)
    >>> sorted = ak.sort(a)
    >>> a
    array([0, 1, 1, 3, 4, 5, 7, 8, 8, 9])
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
