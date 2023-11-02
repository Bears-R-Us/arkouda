from __future__ import annotations

from ._array_object import Array

from typing import NamedTuple

import arkouda as ak

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
    raise ValueError("unique_all not implemented")


def unique_counts(x: Array, /) -> UniqueCountsResult:
    raise ValueError("unique_counts not implemented")


def unique_inverse(x: Array, /) -> UniqueInverseResult:
    raise ValueError("unique_inverse not implemented")


def unique_values(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    res = ak.unique(
        x._array
    )
    return Array._new(res)
