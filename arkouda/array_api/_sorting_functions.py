from __future__ import annotations

from ._array_object import Array
from ._dtypes import _real_numeric_dtypes
from ._manipulation_functions import flip

import arkouda as ak


# Note: the descending keyword argument is new in this function
def argsort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argsort <numpy.argsort>`.

    See its docstring for more information.
    """

    if axis == -1:
        axis = x.ndim - 1

    a = Array._new(ak.argsort(x._array, axis=axis))

    if descending:
        flip(a, axis=axis)

    return a


# Note: the descending keyword argument is new in this function
def sort(x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sort <numpy.sort>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in sort")
    res = ak.sort(x._array)
    return Array._new(res)
