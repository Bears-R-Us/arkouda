from __future__ import annotations

from .array_object import Array, implements_numpy

from typing import List, Optional, Tuple, Union, cast
from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray, create_pdarrays
from arkouda.pdarraycreation import scalar_array, promote_to_common_dtype
from arkouda.util import broadcast_dims

import numpy as np


def broadcast_arrays(*arrays: Array) -> List[Array]:
    """
    Broadcast arrays to a common shape.

    Throws a ValueError if a common shape cannot be determined.
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

    See: https://data-apis.org/array-api/latest/API_specification/broadcasting.html for details.
    """

    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"broadcast<{x.dtype},{x.ndim},{len(shape)}>",
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


def concat(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0) -> Array:
    """
    Concatenate arrays along an axis.

    Parameters
    ----------
    arrays : Tuple[Array, ...] or List[Array]
        The arrays to concatenate. Must have the same shape except along the concatenation axis.
    axis : int, optional
        The axis along which to concatenate the arrays. The default is 0. If None, the arrays are
        flattened before concatenation.
    """

    ndim = arrays[0].ndim
    for a in arrays:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions to concatenate")

    (common_dt, _arrays) = promote_to_common_dtype([a._array for a in arrays])

    return Array._new(
        create_pdarray(
            cast(
                str,
                generic_msg(
                    cmd=(
                        f"concat<{common_dt},{ndim}>"
                        if axis is not None
                        else f"concatFlat<{common_dt},{ndim}>"
                    ),
                    args={
                        "n": len(arrays),
                        "names": _arrays,
                        "axis": axis,
                    },
                ),
            )
        )
    )


def expand_dims(x: Array, /, *, axis: int) -> Array:
    """
    Create a new array with an additional dimension inserted at the specified axis.

    Parameters
    ----------
    x : Array
        The array to expand
    axis : int
        The axis at which to insert the new (size one) dimension. Must be in the range
        `[-x.ndim-1, x.ndim]`.
    """
    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"expandDims<{x.dtype},{x.ndim}>",
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
    Reverse an array's values along a particular axis or axes.

    Parameters
    ----------
    x : Array
        The array to flip
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to flip the array. If None, flip the array along all axes.
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
                        cmd=(
                            f"flipAll<{x.dtype},{x.ndim}>"
                            if axis is None
                            else f"flip<{x.dtype},{x.ndim}>"
                        ),
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
    x: Array,
    source: Union[int, Tuple[int, ...]],
    destination: Union[int, Tuple[int, ...]],
    /,
) -> Array:
    """
    Move axes of an array to new positions.

    Parameters
    ----------
    x : Array
        The array whose axes are to be reordered
    source : int or Tuple[int, ...]
        Original positions of the axes to move. Values must be unique and fall within the range
        `[-x.ndim, x.ndim)`.
    destination : int or Tuple[int, ...]
        Destination positions for each of the original axes. Must be the same length as `source`.
        Values must be unique and fall within the range `[-x.ndim, x.ndim)`.
    """
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
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    x : Array
        The array whose dimensions are to be permuted
    axes : Tuple[int, ...]
        The new order of the dimensions. Must be a permutation of the integers from 0 to `x.ndim-1`.
    """
    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"permuteDims<{x.dtype},{x.ndim}>",
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
    """
    Repeat elements of an array.

    Parameters
    ----------
    x : Array
        The array whose values to repeat
    repeats : int or Array
        The number of repetitions for each element.
         * If axis is None, must be an integer, or a 1D array of integers with the same size as `x`.
         * If axis is not None, must be an integer, or a 1D array of integers whose size matches the
           number of elements along the specified axis.
    axis : int, optional
        The axis along which to repeat elements. If None, the array is flattened before repeating.
    """
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
                        cmd=f"repeatFlat<{x.dtype},{x.ndim}>",
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


def reshape(x: Array, /, shape: Tuple[int, ...], *, copy: Optional[bool] = None) -> Array:
    """
    Reshape an array to a new shape.

    Parameters
    ----------
    x : Array
        The array to reshape
    shape : Tuple[int, ...]
        The new shape for the array. Must have the same number of elements as the original array.
    copy : bool, optional
        Whether to create a copy of the array.
        WARNING: currently always creates a copy, ignoring the value of this parameter.
    """

    # TODO: figure out copying semantics (currently always creates a copy)
    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"reshape<{x.dtype},{x.ndim},{len(shape)}>",
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
    Roll the values in an array by the specified shift(s) along the specified axis or axes.
    Elements that roll beyond the last position are re-introduced at the first position.

    Parameters
    ----------
    x : Array
        The array to roll
    shift : int or Tuple[int, ...]
        The number of positions by which to shift each axis. If `axis` and `shift` are both tuples, they
        must have the same length and the `i`-th element of `shift` is the number of positions to shift
        `axis[i]`. If axis is a tuple and shift is an integer, the same shift is applied to each axis.
        If axis is None, must be an integer or a one-tuple.
    axis: int or Tuple[int, ...], optional
        The axis or axes along which to roll the array. If None, the array is flattened before
        rolling.
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
                        cmd=(
                            f"rollFlattened<{x.dtype},{x.ndim}>"
                            if axis is None
                            else f"roll<{x.dtype},{x.ndim}>"
                        ),
                        args={
                            "name": x._array,
                            "nShifts": len(shift) if isinstance(shift, tuple) else 1,
                            "shift": (list(shift) if isinstance(shift, tuple) else [shift]),
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
    Remove degenerate (size one) dimensions from an array.

    Parameters
    ----------
    x : Array
        The array to squeeze
    axis : int or Tuple[int, ...]
        The axis or axes to squeeze (must have a size of one).
    """
    from arkouda.numpy import squeeze

    return Array._new(squeeze(x._array, axis))


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    """
    Stack arrays along a new axis.

    The resulting array will have one more dimension than the input arrays with a size
    equal to the number of input arrays.

    Parameters
    ----------
    arrays : Tuple[Array, ...] or List[Array]
        The arrays to stack. Must have the same shape.
    axis : int, optional
        The axis along which to stack the arrays. Must be in the range `[-N, N)`, where N is the number
        of dimensions in the input arrays. The default is 0.
    """

    ndim = arrays[0].ndim
    for a in arrays:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions to stack")

    (common_dt, _arrays) = promote_to_common_dtype([a._array for a in arrays])

    # TODO: type promotion across input arrays
    return Array._new(
        create_pdarray(
            cast(
                str,
                generic_msg(
                    cmd=f"stack<{common_dt},{ndim}>",
                    args={
                        "names": _arrays,
                        "n": len(arrays),
                        "axis": axis,
                    },
                ),
            )
        )
    )


def tile(x: Array, repetitions: Tuple[int, ...], /) -> Array:
    """
    Tile an array with the specified number of repetitions along each dimension.

    Parameters
    ----------
    x : Array
        The array to tile
    repetitions : Tuple[int, ...]
        The number of repetitions along each dimension. If there are more repetitions than array
        dimensions, singleton dimensions are prepended to the array to make it match the number of
        repetitions. If there are more array dimensions than repetitions, ones are prepended to the
        repetitions tuple to make it's length match the number of array dimensions.
    """
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
                    cmd=f"tile<{xr.dtype},{xr.ndim}>",
                    args={
                        "name": xr._array,
                        "reps": reps,
                    },
                ),
            )
        )
    )


def unstack(x: Array, /, *, axis: int = 0) -> Tuple[Array, ...]:
    """
    Decompose an array along an axis into multiple arrays of the same shape.

    Parameters
    ----------
    x : Array
        The array to unstack
    axis : int, optional
        The axis along which to unstack the array. The default is 0.
    """
    return tuple(
        Array._new(
            create_pdarrays(
                cast(
                    str,
                    generic_msg(
                        cmd=f"unstack<{x.dtype},{x.ndim}>",
                        args={
                            "name": x._array,
                            "axis": axis,
                            "numReturnArrays": x.shape[axis],
                        },
                    ),
                )
            )
        )
    )
