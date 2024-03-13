from __future__ import annotations

from ._array_object import Array, implements_numpy

from typing import List, Optional, Tuple, Union, cast
from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray
from arkouda.util import broadcast_dims

import numpy as np


def broadcast_arrays(*arrays: Array) -> List[Array]:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_arrays <numpy.broadcast_arrays>`.

    See its docstring for more information.
    """

    shapes = [a.shape for a in arrays]
    bcShape = shapes[0]
    for shape in shapes[1:]:
        bcShape = broadcast_dims(bcShape, shape)

    return [broadcast_to(a, shape=bcShape) for a in arrays]


@implements_numpy(np.broadcast_to)
def broadcast_to(x: Array, /, shape: Tuple[int, ...]) -> Array:
    """
    Broadcast the array to the specified shape.
    """

    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"broadcastTo{x.ndim}Dx{len(shape)}D",
                        args={
                            "name": x._array,
                            "shape": shape,
                        },
                    ),
                )
            )
        )
    except RuntimeError as e:
        raise ValueError(f"Failed to broadcast array: {e}")


# Note: the function name is different here
def concat(
    arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.concatenate <numpy.concatenate>`.

    See its docstring for more information.
    """
    # TODO: type promotion across input arrays

    return Array._new(
        create_pdarray(
            cast(
                str,
                generic_msg(
                    cmd=f"concat{arrays[0].ndim}D"
                    if axis is not None
                    else f"concatFlat{arrays[0].ndim}D",
                    args={
                        "n": len(arrays),
                        "names": [a._array for a in arrays],
                        "axis": axis,
                    },
                ),
            )
        )
    )


def expand_dims(x: Array, /, *, axis: int) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.expand_dims <numpy.expand_dims>`.

    See its docstring for more information.
    """
    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"expandDims{x.ndim}D",
                        args={
                            "name": x._array,
                            "axis": axis,
                        },
                    ),
                )
            )
        )
    except RuntimeError as e:
        raise (IndexError(f"Failed to expand array dimensions: {e}"))


def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.flip <numpy.flip>`.

    See its docstring for more information.
    """
    axisList = []
    if axis is not None:
        axisList = list(axis) if isinstance(axis, tuple) else [axis]
    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"flipAll{x.ndim}D" if axis is None else f"flip{x.ndim}D",
                        args={
                            "name": x._array,
                            "nAxes": len(axisList),
                            "axis": axisList,
                        },
                    ),
                )
            )
        )
    except RuntimeError as e:
        raise IndexError(f"Failed to flip array: {e}")


def moveaxis(x: Array, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]], /) -> Array:
    raise NotImplementedError("moveaxis is not yet implemented")


def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"permuteDims{x.ndim}D",
                        args={
                            "name": x._array,
                            "axes": axes,
                        },
                    ),
                )
            )
        )
    except RuntimeError as e:
        raise IndexError(f"Failed to permute array dimensions: {e}")


def repeat(x: Array, repeats: Union[int, Array], /, *, axis: Optional[int] = None) -> Array:
    raise NotImplementedError("repeat is not yet implemented")


def reshape(
    x: Array, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.reshape <numpy.reshape>`.

    See its docstring for more information.
    """

    # TODO: figure out copying semantics (currently always creates a copy)
    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"reshape{x.ndim}Dx{len(shape)}D",
                        args={
                            "name": x._array,
                            "shape": shape,
                        },
                    ),
                )
            )
        )
    except RuntimeError as e:
        raise ValueError(f"Failed to reshape array: {e}")


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
    axisList = []
    if axis is not None:
        axisList = list(axis) if isinstance(axis, tuple) else [axis]
    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"rollFlattened{x.ndim}D"
                        if axis is None
                        else f"roll{x.ndim}D",
                        args={
                            "name": x._array,
                            "nShifts": len(shift) if isinstance(shift, tuple) else 1,
                            "shift": list(shift)
                            if isinstance(shift, tuple)
                            else [shift],
                            "nAxes": len(axisList),
                            "axis": axisList,
                        },
                    ),
                )
            )
        )
    except RuntimeError as e:
        raise IndexError(f"Failed to roll array: {e}")


def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.squeeze <numpy.squeeze>`.

    See its docstring for more information.
    """
    nAxes = len(axis) if isinstance(axis, tuple) else 1
    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"squeeze{x.ndim}Dx{x.ndim - nAxes}D",
                        args={
                            "name": x._array,
                            "nAxes": nAxes,
                            "axes": list(axis) if isinstance(axis, tuple) else [axis],
                        },
                    ),
                )
            )
        )
    except RuntimeError as e:
        raise ValueError(f"Failed to squeeze array: {e}")


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.stack <numpy.stack>`.

    See its docstring for more information.
    """
    # TODO: type promotion across input arrays
    return Array._new(
        create_pdarray(
            cast(
                str,
                generic_msg(
                    cmd=f"stack{arrays[0].ndim}D",
                    args={
                        "names": [a._array for a in arrays],
                        "n": len(arrays),
                        "axis": axis,
                    },
                ),
            )
        )
    )


def tile(x: Array, repetitions: Tuple[int, ...], /) -> Array:
    raise NotImplementedError("tile is not yet implemented")


def unstack(x: Array, /, *, axis: int = 0) -> Tuple[Array, ...]:
    raise NotImplementedError("unstack is not yet implemented")
