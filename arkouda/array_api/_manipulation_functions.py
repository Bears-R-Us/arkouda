from __future__ import annotations

from ._array_object import Array

from typing import List, Optional, Tuple, Union, cast
from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray


def broadcast_arrays(*arrays: Array) -> List[Array]:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_arrays <numpy.broadcast_arrays>`.

    See its docstring for more information.
    """
    raise ValueError("broadcast_arrays not implemented")


def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    """
    Broadcast the array to the specified shape.
    """

    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"broadcastTo{x.ndim}Dx{len(shape)}D",
                args={
                    "name": x,
                    "shape": shape,
                },
            ),
        )
    )


# Note: the function name is different here
def concat(
    arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.concatenate <numpy.concatenate>`.

    See its docstring for more information.
    """
    # TODO: add type check for arrays before calling 'concat'

    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"concat{arrays[0].ndim}D" if axis is not None else f"concatFlat{arrays[0].ndim}D",
                args={
                    "n": len(arrays),
                    "names": arrays,
                    "axis": axis,
                },
            ),
        )
    )


def expand_dims(x: Array, /, *, axis: int) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.expand_dims <numpy.expand_dims>`.

    See its docstring for more information.
    """
    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"expandDims{x.ndim}D",
                args={
                    "name": x,
                    "axis": axis,
                },
            ),
        )
    )


def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.flip <numpy.flip>`.

    See its docstring for more information.
    """
    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"flipAll{x.ndim}D" if axis is None else f"flip{x.ndim}D",
                args={
                    "name": x,
                    "axis": axis,
                },
            ),
        )
    )


# Note: The function name is different here (see also matrix_transpose).
# Unlike transpose(), the axes argument is required.
def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    # return Array._new(ak.transpose(x._array, axes))
    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"permuteDims{x.ndim}D",
                args={
                    "name": x,
                    "perm": axes,
                },
            ),
        )
    )


# Note: the optional argument is called 'shape', not 'newshape'
def reshape(
    x: Array, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.reshape <numpy.reshape>`.

    See its docstring for more information.
    """

    # TODO: figure out copying semantics (currently always creates a copy)
    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"reshape{x.ndim}D",
                args={
                    "name": x,
                    "shape": shape,
                },
            ),
        )
    )


def roll(
    x: Array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.roll <numpy.roll>`.

    See its docstring for more information.
    """
    nAxes = 0
    if axis is not None:
        nAxes = len(axis) if isinstance(axis, tuple) else 1
    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"rollFlattened{x.ndim}D" if axis is None else f"roll{x.ndim}D",
                args={
                    "name": x,
                    "nShifts": len(shift) if isinstance(shift, tuple) else 1,
                    "nAxes": nAxes,
                    "shift": shift,
                    "axis": axis,
                },
            ),
        )
    )


def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.squeeze <numpy.squeeze>`.

    See its docstring for more information.
    """
    nAxes = len(axis) if isinstance(axis, tuple) else 1
    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"squeeze{x.ndim}Dx{x.ndim - nAxes}D",
                args={
                    "name": x,
                    "nAxes": nAxes,
                    "axes": axis,
                },
            ),
        )
    )


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.stack <numpy.stack>`.

    See its docstring for more information.
    """
    # # Call result type here just to raise on disallowed type combinations
    # result_type(*arrays)
    # arrays = tuple(a._array for a in arrays)
    # return Array._new(np.stack(arrays, axis=axis))
    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"stack{arrays[0].ndim}D",
                args={
                    "names": arrays,
                    "n": len(arrays),
                    "axis": axis,
                },
            ),
        )
    )
