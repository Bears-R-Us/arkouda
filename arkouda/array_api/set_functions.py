from __future__ import annotations

from typing import NamedTuple, cast

from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray, create_pdarrays

from .array_object import Array


class UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array


class UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array


class UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array


def unique_all(x: Array, /) -> UniqueAllResult:
    """
    Return a tuple of arrays containing:
    - the unique values in `x`
    - the indices of the first occurrence of each unique value
    - the inverse indices that reconstruct `x` from the unique values
    - the counts of each unique value
    """
    arrays = create_pdarrays(
        cast(
            str,
            generic_msg(
                cmd=f"uniqueAll<{x.dtype},{x.ndim}>",
                args={"name": x._array},
            ),
        )
    )

    return UniqueAllResult(
        values=Array._new(arrays[0]),
        indices=Array._new(arrays[1]),
        inverse_indices=Array._new(arrays[2]),
        counts=Array._new(arrays[3]),
    )


def unique_counts(x: Array, /) -> UniqueCountsResult:
    """
    Return a tuple of arrays containing:
    - the unique values in `x`
    - the counts of each unique value
    """
    arrays = create_pdarrays(
        cast(
            str,
            generic_msg(
                cmd=f"uniqueCounts<{x.dtype},{x.ndim}>",
                args={"name": x._array},
            ),
        )
    )

    return UniqueCountsResult(
        values=Array._new(arrays[0]),
        counts=Array._new(arrays[1]),
    )


def unique_inverse(x: Array, /) -> UniqueInverseResult:
    """
    Return a tuple of arrays containing:
    - the unique values in `x`
    - the inverse indices that reconstruct `x` from the unique values
    """
    arrays = create_pdarrays(
        cast(
            str,
            generic_msg(
                cmd=f"uniqueInverse<{x.dtype},{x.ndim}>",
                args={"name": x._array},
            ),
        )
    )

    return UniqueInverseResult(
        values=Array._new(arrays[0]),
        inverse_indices=Array._new(arrays[1]),
    )


def unique_values(x: Array, /) -> Array:
    """
    Return an array containing the unique values from `x`.
    """
    return Array._new(
        create_pdarray(
            cast(
                str,
                generic_msg(
                    cmd=f"uniqueValues<{x.dtype},{x.ndim}>",
                    args={"name": x._array},
                ),
            )
        )
    )
