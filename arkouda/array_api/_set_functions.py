from __future__ import annotations

from ._array_object import Array

from typing import NamedTuple, cast

from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray


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
    resp = cast(
                str,
                generic_msg(
                    cmd=f"uniqueAll{x.ndim}D",
                    args={"name": x._array},
                ),
            )

    arrays = [Array._new(create_pdarray(r)) for r in resp.split('+')]

    return UniqueAllResult(
        values=arrays[0],
        indices=arrays[1],
        inverse_indices=arrays[2],
        counts=arrays[3],
    )


def unique_counts(x: Array, /) -> UniqueCountsResult:
    resp = cast(
                str,
                generic_msg(
                    cmd=f"uniqueCounts{x.ndim}D",
                    args={"name": x._array},
                ),
            )

    arrays = [Array._new(create_pdarray(r)) for r in resp.split('+')]

    return UniqueCountsResult(
        values=arrays[0],
        counts=arrays[1],
    )


def unique_inverse(x: Array, /) -> UniqueInverseResult:
    resp = cast(
                str,
                generic_msg(
                    cmd=f"uniqueInverse{x.ndim}D",
                    args={"name": x._array},
                ),
            )

    arrays = [Array._new(create_pdarray(r)) for r in resp.split('+')]

    return UniqueInverseResult(
        values=arrays[0],
        inverse_indices=arrays[1],
    )


def unique_values(x: Array, /) -> Array:
    return Array._new(
        create_pdarray(
            cast(
                str,
                generic_msg(
                    cmd=f"uniqueValues{x.ndim}D",
                    args={"name": x._array},
                ),
            )
        )
    )
