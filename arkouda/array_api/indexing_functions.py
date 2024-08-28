from __future__ import annotations

from .array_object import Array
from typing import Optional

from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray


def take(x: Array, indices: Array, /, *, axis: Optional[int] = None) -> Array:
    """
    Get the specified elements of an array along an axis.

    Parameters
    ----------

    x : Array
        The array from which to take elements
    indices : Array
        A 1D integer array of indices to take from `x`
    axis : int, optional
        The axis along which to take elements. If None, `x` must be 1D.
    """

    if axis is None and x.ndim != 1:
        raise ValueError("axis must be specified for multidimensional arrays")

    if indices.ndim != 1:
        raise ValueError("indices must be 1D")

    if axis is None:
        axis = 0

    repMsg = generic_msg(
        cmd=f"takeAlongAxis<{x.dtype},{indices.dtype},{x.ndim}>",
        args={
            "x": x._array,
            "indices": indices._array,
            "axis": axis,
        },
    )

    return Array._new(create_pdarray(repMsg))
