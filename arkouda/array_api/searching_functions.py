from __future__ import annotations

from typing import Literal, Optional, Tuple, cast

import arkouda as ak
from arkouda.numpy import cast as akcast
from arkouda.numpy.pdarrayclass import create_pdarray, create_pdarrays

from ._dtypes import _real_floating_dtypes, _real_numeric_dtypes
from .array_object import Array
from .manipulation_functions import broadcast_arrays

__all__ = [
    "argmax",
    "argmin",
    "nonzero",
    "searchsorted",
    "where",
]


def argmax(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Return an array with the indices of the maximum values along a given axis.

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

    return Array._new(ak.argmax(x._array, axis=axis, keepdims=keepdims))


def argmin(x: Array, /, *, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """
    Return an array with the indices of the minimum values along a given axis.

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
    return Array._new(ak.argmin(x._array, axis=axis, keepdims=keepdims))


def nonzero(x: Array, /) -> Tuple[Array, ...]:
    """
    Return a tuple of arrays containing the indices of the non-zero elements of the input array.
    """
    from arkouda.client import generic_msg

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
    from arkouda.client import generic_msg

    broadcasted = broadcast_arrays(condition, x1, x2)
    assert isinstance(broadcasted, list) and all(isinstance(arg, Array) for arg in broadcasted)

    a = broadcasted[1]._array
    b = broadcasted[2]._array
    c = akcast(broadcasted[0]._array, ak.bool_)

    return Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"wherevv<{c.ndim},{a.dtype},{b.dtype}>",
                args={
                    "condition": c,
                    "a": a,
                    "b": b,
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
    x2_sorted: bool = False,
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
    x2_sorted : bool, default=False
        If True, assumes that `x2` is already sorted in ascending order. This can improve performance
        for large, sorted search arrays. If False, no assumption is made about the order of `x2`.
    """
    from arkouda.client import generic_msg

    if x1.dtype not in _real_floating_dtypes or x2.dtype not in _real_floating_dtypes:
        raise TypeError("Only real dtypes are allowed in searchsorted")

    if x1.ndim > 1:
        raise ValueError("searchsorted only supports 1D arrays for x1")

    if sorter is not None:
        _x1 = x1[sorter]
    else:
        _x1 = x1

    resp = generic_msg(
        cmd=f"searchSorted<{x1.dtype},1,{x2.ndim}>",
        args={
            "x1": _x1._array,
            "x2": x2._array,
            "side": side,
            "x2Sorted": x2_sorted,
        },
    )

    return Array._new(create_pdarray(resp))
