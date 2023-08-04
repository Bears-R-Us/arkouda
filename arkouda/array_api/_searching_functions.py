from __future__ import annotations

from ._array_object import Array
from ._dtypes import _result_type, _real_numeric_dtypes

from typing import Optional, Tuple

import arkoud as ak


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmax <numpy.argmax>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmax")
    return Array._new(ak.asarray(ak.argmax(x._array)))


def argmin(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmin <numpy.argmin>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmin")
    return Array._new(ak.asarray(np.argmin(x._array)))


def nonzero(x: Array, /) -> Tuple[Array, ...]:
    """
    Array API compatible wrapper for :py:func:`np.nonzero <numpy.nonzero>`.

    See its docstring for more information.
    """
    raise ValueError("nonzero not implemented")


def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.where <numpy.where>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    raise ValueError("where not implemented")
