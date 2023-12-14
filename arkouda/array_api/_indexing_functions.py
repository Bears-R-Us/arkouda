from __future__ import annotations

from ._array_object import Array
from typing import Optional

from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray

def take(x: Array, indices: Array, /, *, axis: Optional[int] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.take <numpy.take>`.

    See its docstring for more information.
    """

    if axis is None and x.ndim != 1:
        raise ValueError("axis must be specified for multidimensional arrays")

    if indices.ndim != 1:
        raise ValueError("indices must be 1D")

    if axis is None:
        axis = 0

    repMsg = generic_msg(
        cmd=f"takeAlongAxis{x.ndim}D",
        args={
            "x": x._array,
            "indices": indices._array,
            "axis": axis,
        },
    )

    return Array._new(create_pdarray(repMsg))
