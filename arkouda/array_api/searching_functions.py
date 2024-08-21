from __future__ import annotations

from .array_object import Array
from ._dtypes import _real_numeric_dtypes, _real_floating_dtypes

from typing import Optional, Tuple, Literal, cast

from .manipulation_functions import squeeze, reshape, broadcast_arrays

from arkouda.client import generic_msg
from arkouda.pdarrayclass import parse_single_value, create_pdarray, create_pdarrays
from arkouda.pdarraycreation import scalar_array
from arkouda.numeric import cast as akcast
import arkouda as ak


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Returns an array with the indices of the maximum values along a given axis.

    Parameters
    ----------
    x : Array
        The array to search for maximum values
    axis : int, optional
        The axis along which to search for maximum values. If None, the array is flattened before
        searching.
    keepdims : bool, optional
        Whether to keep the singleton dimension along `axis` in the result.

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
    Returns an array with the indices of the minimum values along a given axis.

    Parameters
    ----------
    x : Array
        The array to search for minimum values
    axis : int, optional
        The axis along which to search for minimum values. If None, the array is flattened before
        searching.
    keepdims : bool, optional
        Whether to keep the singleton dimension along `axis` in the result.
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
    Returns a tuple of arrays containing the indices of the non-zero elements of the input array.
    """
    resp = cast(
        str,
        generic_msg(
            cmd=f"nonzero<{x.dtype},{x.ndim}>",
            args={"x": x._array},
        ),
    )

    return tuple([Array._new(a) for a in create_pdarrays(resp)])


def where(condition: Array, x1: Array, x2: Array, /) -> Array:
    """
    Return elements, either from `x1` or `x2`, depending on `condition`.

    Parameters
    ----------
    condition : Array
        When condition[i] is True, store x1[i] in the output array, otherwise store x2[i].
    x1 : Array
        Values selected at indices where `condition` is True.
    x2 : Array
        Values selected at indices where `condition` is False.
    """
    broadcasted = broadcast_arrays(condition, x1, x2)

    return Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"efunc3vv{broadcasted[0].ndim}D",
                args={
                    "func": "where",
                    "condition": akcast(broadcasted[0]._array, ak.dtypes.bool_),
                    "a": broadcasted[1]._array,
                    "b": broadcasted[2]._array,
                },
            )
        )
    )


def searchsorted(
    x1: Array,
    x2: Array,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Optional[Array] = None,
) -> Array:
    """
    Given a sorted array `x1`, find the indices to insert elements from another array `x2` such that
    the sorted order is maintained.

    Parameters
    ----------
    x1 : Array
        The sorted array to search in.
    x2 : Array
        The values to search for in `x1`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given. If 'right', return the
        last such index. Default is 'left'.
    sorter : Array, optional
        The indices that would sort `x1` in ascending order. If None, `x1` is assumed to be sorted.

    """
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
