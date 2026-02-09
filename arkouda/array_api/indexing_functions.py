from __future__ import annotations

from typing import Optional

from .array_object import Array


__all__ = [
    "take",
]


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
    from arkouda.numpy.numeric import take

    if axis is None and x.ndim != 1:
        raise ValueError("axis must be specified for multidimensional arrays")

    return Array._new(take(x._array, indices._array, axis))
