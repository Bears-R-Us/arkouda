from __future__ import annotations

from ._array_object import Array
from ._dtypes import _real_numeric_dtypes
from ._manipulation_functions import flip

import arkouda as ak


def argsort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    if axis == -1:
        axis = x.ndim - 1

    a = Array._new(ak.argsort(x._array, axis=axis))

    # TODO: pass a 'flip' argument to the server to avoid this extra step
    if descending:
        flip(a, axis=axis)

    return a


def sort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    if axis == -1:
        axis = x.ndim - 1

    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in sort")

    a = Array._new(ak.sort(x._array, axis=axis))

    # TODO: pass a 'flip' argument to the server to avoid this extra step
    if descending:
        flip(a, axis=axis)

    return a
