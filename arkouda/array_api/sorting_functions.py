from __future__ import annotations

import arkouda as ak

from ._dtypes import _real_numeric_dtypes
from .array_object import Array
from .manipulation_functions import flip

__all__ = [
    "argsort",
    "sort",
]


def argsort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    """
    Return the indices that sort an array along a specified axis.

    Parameters
    ----------
    x : Array
        The array to sort
    axis : int, optional
        The axis along which to sort.
    descending : bool, optional
        Whether to sort in descending order.
    stable : bool, optional
        Whether to use a stable sorting algorithm. Note: arkouda's sorting algorithm is always stable so
        this argument is ignored.

    """
    a = Array._new(ak.argsort(x._array, axis=axis))

    # TODO: pass a 'flip' argument to the server to avoid this extra step
    if descending:
        return flip(a, axis=axis)

    return a


def sort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    """
    Return a sorted copy of an array along a specified axis.

    Parameters
    ----------
    x : Array
        The array to sort
    axis : int, optional
        The axis along which to sort.
    descending : bool, optional
        Whether to sort in descending order.
    stable : bool, optional
        Whether to use a stable sorting algorithm. Note: arkouda's sorting algorithm is always stable so
        this argument is ignored.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in sort")

    a = Array._new(ak.sort(x._array, axis=axis))

    # TODO: pass a 'flip' argument to the server to avoid this extra step
    if descending:
        flip(a, axis=axis)

    return a
