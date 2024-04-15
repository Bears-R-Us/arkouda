from __future__ import annotations

from ._array_object import Array, implements_numpy

from typing import List, Optional, Tuple, Union, cast
from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray
from arkouda.pdarraycreation import scalar_array
from arkouda.util import broadcast_dims

import numpy as np


def broadcast_arrays(*arrays: Array) -> List[Array]:
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


def concat(
    arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0
) -> Array:
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


def moveaxis(
    x: Array, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]], /
) -> Array:
    perm = list(range(x.ndim))
    if isinstance(source, tuple):
        if isinstance(destination, tuple):
            for s, d in zip(source, destination):
                perm[s] = d
        else:
            raise ValueError("source and destination must both be tuples if source is a tuple")
    elif isinstance(destination, int):
        perm[source] = destination
    else:
        raise ValueError("source and destination must both be integers if source is a tuple")

    return permute_dims(x, axes=tuple(perm))


def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
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
    if isinstance(repeats, int):
        reps = Array._new(scalar_array(repeats))
    else:
        reps = repeats

    if axis is None:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"repeatFlat{x.ndim}D",
                        args={
                            "name": x._array,
                            "repeats": reps._array,
                        },
                    ),
                )
            )
        )
    else:
        raise NotImplementedError("repeat with 'axis' argument is not yet implemented")


def reshape(
    x: Array, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None
) -> Array:

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
    if len(repetitions) > x.ndim:
        xr = reshape(x, (1,) * (len(repetitions) - x.ndim) + x.shape)
        reps = repetitions
    elif len(repetitions) < x.ndim:
        xr = x
        reps = (1,) * (x.ndim - len(repetitions)) + repetitions
    else:
        xr = x
        reps = repetitions

    return Array._new(
        create_pdarray(
            cast(
                str,
                generic_msg(
                    cmd=f"tile{xr.ndim}D",
                    args={
                        "name": xr._array,
                        "reps": reps,
                    },
                ),
            )
        )
    )


def unstack(x: Array, /, *, axis: int = 0) -> Tuple[Array, ...]:
    resp = cast(
                str,
                generic_msg(
                    cmd=f"unstack{x.ndim}D",
                    args={
                        "name": x._array,
                        "axis": axis,
                        "numReturnArrays": x.shape[axis],
                    },
                ),
            )

    return tuple([Array._new(create_pdarray(a)) for a in resp.split("+")])
