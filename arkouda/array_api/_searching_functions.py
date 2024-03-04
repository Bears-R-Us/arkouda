from __future__ import annotations

from ._array_object import Array
from ._dtypes import _real_numeric_dtypes

from typing import Optional, Tuple

from ._manipulation_functions import squeeze, reshape

from arkouda.client import generic_msg
from arkouda.pdarrayclass import parse_single_value, create_pdarray
from arkouda.pdarraycreation import scalar_array


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmax <numpy.argmax>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmax")

    if x.ndim > 1 and axis is None:
        # must flatten ND arrays to 1D without an axis argument
        x_op = reshape(x, shape=(-1,))
    else:
        x_op = x

    resp = generic_msg(
        cmd=f"reduce->idx{x_op.ndim}D",
        args={
            "x": x_op._array,
            "op": "argmax",
            "hasAxis": axis is not None,
            "axis": axis if axis is not None else 0,
        },
    )

    if axis is None:
        return Array._new(scalar_array(parse_single_value(resp)))
    else:
        arr = Array._new(create_pdarray(resp))

        if keepdims:
            return arr
        else:
            return squeeze(arr, axis)


def argmin(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.argmin <numpy.argmin>`.

    See its docstring for more information.
    """
    if x.dtype not in _real_numeric_dtypes:
        raise TypeError("Only real numeric dtypes are allowed in argmax")

    if x.ndim > 1 and axis is None:
        # must flatten ND arrays to 1D without an axis argument
        x_op = reshape(x, shape=(-1,))
    else:
        x_op = x

    resp = generic_msg(
        cmd=f"reduce->idx{x_op.ndim}D",
        args={
            "x": x_op._array,
            "op": "argmin",
            "hasAxis": axis is not None,
            "axis": axis if axis is not None else 0,
        },
    )

    if axis is None:
        return Array._new(scalar_array(parse_single_value(resp)))
    else:
        arr = Array._new(create_pdarray(resp))

        if keepdims:
            return arr
        else:
            return squeeze(arr, axis)


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
