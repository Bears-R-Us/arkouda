from __future__ import annotations

from ._array_object import Array
from ._manipulation_functions import concat

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
    Array API compatible wrapper for :py:func:`np.all <numpy.all>`.

    See its docstring for more information.
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
    Array API compatible wrapper for :py:func:`np.any <numpy.any>`.

    See its docstring for more information.
    """
    return Array._new(scalar_array(ak.any(x._array)))


def clip(a: Array, a_min, a_max, /) -> Array:
    return Array._new(
        create_pdarray(
            generic_msg(
                cmd=f"clip{a.ndim}D",
                args={
                    "name": a._array,
                    "min": a_min,
                    "max": a_max,
                },
            ),
        )
    )


def diff(a: Array, /, n: int = 1, axis: int = -1, prepend=None, append=None) -> Array:
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
                cmd=f"diff{a.ndim}D",
                args={
                    "name": a_._array,
                    "n": n,
                    "axis": axis,
                },
            ),
        )
    )


def pad(
    array: Array,
    pad_width,  # Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]]
    mode='constant',
    **kwargs
) -> Array:
    if mode != 'constant':
        raise NotImplementedError(f"pad mode '{mode}' is not supported")

    if 'constant_values' not in kwargs:
        cvals = 0
    else:
        cvals = kwargs['constant_values']

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
                cmd=f"pad{array.ndim}D",
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
