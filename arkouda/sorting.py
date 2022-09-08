from __future__ import annotations

from enum import Enum
from typing import Sequence, Union, cast

from typeguard import check_type, typechecked

from arkouda.client import generic_msg
from arkouda.dtypes import float64, int64, int_scalars, uint64
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.pdarraycreation import zeros
from arkouda.strings import Strings

numeric_dtypes = {int64, uint64, float64}

__all__ = ["argsort", "coargsort", "sort", "SortingAlgorithm"]

SortingAlgorithm = Enum("SortingAlgorithm", ["RadixSortLSD", "TwoArrayRadixSort"])


def argsort(
    pda: Union[pdarray, Strings, "Categorical"],  # type: ignore # noqa
    algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
) -> pdarray:  # type: ignore
    """
    Return the permutation that sorts the array.

    Parameters
    ----------
    pda : pdarray or Strings or Categorical
        The array to sort (int64, uint64, or float64)

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
    """
    from arkouda.categorical import Categorical

    check_type(argname="argsort", value=pda, expected_type=Union[pdarray, Strings, Categorical])
    if hasattr(pda, "argsort"):
        return cast(Categorical, pda).argsort()
    if pda.size == 0 and hasattr(pda, "dtype"):
        return zeros(0, dtype=pda.dtype)
    repMsg = generic_msg(cmd="argsort", args={
        "name": pda.entry.name if isinstance(pda, Strings) else pda.name,
        "algoName": algorithm.name,
        "objType": pda.objtype,
    })
    return create_pdarray(cast(str, repMsg))


def coargsort(
    arrays: Sequence[Union[Strings, pdarray, "Categorical"]],  # type: ignore # noqa
    algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
) -> pdarray:  # type: ignore
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

    check_type(
        argname="coargsort", value=arrays, expected_type=Sequence[Union[pdarray, Strings, Categorical]]
    )
    size: int_scalars = -1
    anames = []
    atypes = []
    for a in arrays:
        if isinstance(a, pdarray):
            anames.append("+".join(a._list_component_names()))
            atypes.append(a.objtype)
        elif isinstance(a, Categorical):
            anames.append(a.codes.name)
            atypes.append(a.objtype)
        elif isinstance(a, Strings):
            atypes.append(a.objtype)
            anames.append(a.entry.name)
        else:
            raise ValueError("Argument must be an iterable of pdarrays, Strings, or Categoricals")
        if size == -1:
            size = a.size
        elif size != a.size:
            raise ValueError("All pdarrays, Strings, or Categoricals must be of the same size")
    if size == 0:
        return zeros(0, dtype=arrays[0].dtype)

    repMsg = generic_msg(
        cmd="coargsort",
        args={
            "algoName": algorithm.name,
            "nstr": len(arrays),
            "arr_names": anames,
            "arr_types": atypes,
        },
    )
    return create_pdarray(cast(str, repMsg))


@typechecked
def sort(pda: pdarray, algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD) -> pdarray:
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
    if pda.dtype not in numeric_dtypes:
        raise ValueError(f"ak.sort supports int64, uint64, or float64, not {pda.dtype}")
    if pda.size == 0:
        return zeros(0, dtype=pda.dtype)
    repMsg = generic_msg(cmd="sort", args={
        "alg": algorithm.name,
        "array": pda
    })
    return create_pdarray(cast(str, repMsg))
