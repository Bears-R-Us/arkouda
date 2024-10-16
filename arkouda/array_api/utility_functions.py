from __future__ import annotations

from .array_object import Array
from .manipulation_functions import concat

from typing import Optional, Tuple, Union

from arkouda.pdarraycreation import scalar_array
from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray
import arkouda as ak


def all(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Check whether all elements of an array evaluate to True along a given axis.

    Parameters
    ----------
    x : Array
        The array to check for all True values
    axis : int or Tuple[int], optional
        The axis or axes along which to check for all True values. If None, check all elements.
    keepdims : bool, optional
        Whether to keep the singleton dimensions along `axis` in the result.
    """
    return Array._new(scalar_array(ak.all(x._array)))


def any(
    x: Array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Array:
    """
    Check whether any elements of an array evaluate to True along a given axis.

    Parameters
    ----------
    x : Array
        The array to check for any True values
    axis : int or Tuple[int], optional
        The axis or axes along which to check for any True values. If None, check all elements.
    keepdims : bool, optional
        Whether to keep the singleton dimensions along `axis` in the result.
    """
    return Array._new(scalar_array(ak.any(x._array)))


def clip(a: Array, a_min, a_max, /) -> Array:
    """
    Clip (limit) the values in an array to a given range.

    Parameters
    ----------
    a : Array
        The array to clip
    a_min : scalar
        The minimum value
    a_max : scalar
        The maximum value
    """
    if a.dtype == ak.bigint or a.dtype == ak.bool_:
        raise RuntimeError(f"Error executing command: clip does not support dtype {a.dtype}")

    return Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"clip<{a.dtype},{a.ndim}>",
                args={
                    "x": a._array,
                    "min": a_min,
                    "max": a_max,
                },
            ),
        )
    )


def diff(a: Array, /, n: int = 1, axis: int = -1, prepend=None, append=None) -> Array:
    """
    Calculate the n-th discrete difference along the given axis.

    Parameters
    ----------
    a : Array
        The array to calculate the difference
    n : int, optional
        The order of the finite difference. Default is 1.
    axis : int, optional
        The axis along which to calculate the difference. Default is the last axis.
    prepend : Array, optional
        Array to prepend to `a` along `axis` before calculating the difference.
    append : Array, optional
        Array to append to `a` along `axis` before calculating the difference.
    """
    if a.dtype == ak.bigint or a.dtype == ak.bool_:
        raise RuntimeError(f"Error executing command: diff does not support dtype {a.dtype}")

    if prepend is not None and append is not None:
        a_ = concat((prepend, a, append), axis=axis)
    elif prepend is not None:
        a_ = concat((prepend, a), axis=axis)
    elif append is not None:
        a_ = concat((a, append), axis=axis)
    else:
        a_ = a

    return Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"diff<{a.dtype},{a.ndim}>",
                args={
                    "x": a_._array,
                    "n": n,
                    "axis": axis,
                },
            ),
        )
    )


def pad(
    array: Array,
    pad_width,  # Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]]
    mode="constant",
    **kwargs,
) -> Array:
    """
    Pad an array.

    Parameters
    ----------
    array : Array
        The array to pad
    pad_width : int or Tuple[int, int] or Tuple[Tuple[int, int], ...]
        Number of values padded to the edges of each axis. If a single int, the same value is used for
        all axes. If a tuple of two ints, those values are used for all axes. If a tuple of tuples, each
        inner tuple specifies the number of values padded to the beginning and end of each axis.
    mode : str, optional
        Padding mode. Only 'constant' is currently supported. Use the `constant_values` keyword argument
        to specify the padding value or values (in the same format as `pad_width`).
    """
    if mode != "constant":
        raise NotImplementedError(f"pad mode '{mode}' is not supported")

    if array.dtype == ak.bigint:
        raise RuntimeError("Error executing command: pad does not support dtype bigint")

    if "constant_values" not in kwargs:
        cvals = 0
    else:
        cvals = kwargs["constant_values"]

    if isinstance(pad_width, int):
        pad_widths_b = [pad_width] * array.ndim
        pad_widths_a = [pad_width] * array.ndim
    elif isinstance(pad_width, tuple):
        if isinstance(pad_width[0], int):
            pad_widths_b = [pad_width[0]] * array.ndim
            pad_widths_a = [pad_width[1]] * array.ndim
        elif isinstance(pad_width[0], tuple):
            pad_widths_b = [pw[0] for pw in pad_width]
            pad_widths_a = [pw[1] for pw in pad_width]

    if isinstance(cvals, int):
        pad_vals_b = [cvals] * array.ndim
        pad_vals_a = [cvals] * array.ndim
    elif isinstance(cvals, tuple):
        if isinstance(cvals[0], int):
            pad_vals_b = [cvals[0]] * array.ndim
            pad_vals_a = [cvals[1]] * array.ndim
        else:
            pad_vals_b = [cv[0] for cv in cvals]
            pad_vals_a = [cv[1] for cv in cvals]

    return Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"pad<{array.dtype},{array.ndim}>",
                args={
                    "name": array._array,
                    "padWidthBefore": tuple(pad_widths_b),
                    "padWidthAfter": tuple(pad_widths_a),
                    "padValsBefore": pad_vals_b,
                    "padValsAfter": pad_vals_a,
                },
            ),
        )
    )
