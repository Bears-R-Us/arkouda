from __future__ import annotations

from ._array_object import Array
from ._dtypes import _real_numeric_dtypes, _real_floating_dtypes

from typing import Optional, Tuple, Literal, cast

from ._manipulation_functions import squeeze, reshape, broadcast_arrays

from arkouda.client import generic_msg
from arkouda.pdarrayclass import parse_single_value, create_pdarray
from arkouda.pdarraycreation import scalar_array
from arkouda.numeric import cast as akcast
import arkouda as ak


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
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
    resp = cast(
        str,
        generic_msg(
            cmd=f"nonzero{x.ndim}D",
            args={"x": x._array},
        )
    )

    return tuple([Array._new(create_pdarray(a)) for a in resp.split("+")])


def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    broadcasted = broadcast_arrays(condition, x1, x2)

    return Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"efunc3vv{broadcasted[0].ndim}D",
                args={
                    "func": "where",
                    "condition": akcast(broadcasted[0]._array, ak.dtypes.bool),
                    "a": broadcasted[1]._array,
                    "b": broadcasted[2]._array,
                },
            )
        )
    )


def searchsorted(
    x1: Array, x2: Array, /, *, side: Literal['left', 'right'] = 'left', sorter: Optional[Array] = None
) -> Array:
    if x1.dtype not in _real_floating_dtypes or x2.dtype not in _real_floating_dtypes:
        raise TypeError("Only real dtypes are allowed in searchsorted")

    if x1.ndim > 1:
        raise ValueError("searchsorted only supports 1D arrays for x1")

    if sorter is not None:
        _x1 = x1[sorter]
    else:
        _x1 = x1

    resp = generic_msg(
        cmd=f"searchSorted{x2.ndim}D",
        args={
            "x1": _x1._array,
            "x2": x2._array,
            "side": side,
        },
    )

    return Array._new(create_pdarray(resp))
