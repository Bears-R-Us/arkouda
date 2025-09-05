from __future__ import annotations

from typing import List, Optional, Tuple, Union, cast

import numpy as np

from arkouda.numpy.pdarrayclass import create_pdarray, create_pdarrays
from arkouda.numpy.pdarraycreation import promote_to_common_dtype, scalar_array
from arkouda.numpy.util import broadcast_dims

from .array_object import Array, implements_numpy

__all__ = [
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "repeat",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
]


def broadcast_arrays(*arrays: Array) -> List[Array]:
    """
    Broadcast arrays to a common shape.

    Parameters
    ----------
    arrays : Array
        The arrays to broadcast. Must be broadcastable to a common shape.

    Raises
    ------
    ValueError
        Raised by broadcast_dims if a common shape cannot be determined.

    Returns
    -------
    List
        A list whose elements are the given Arrays broadcasted to the common shape.
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

    Parameters
    ----------
    x: Array
        The array to be broadcast.
    shape: Tuple[int, ...]
        The shape to which the array is to be broadcast.

    Raises
    ------
    ValueError
        Raised server-side if the broadcast fails.

    Returns
    -------
    Array
        A new array which is x broadcast to the provided shape.

    See: https://data-apis.org/array-api/latest/API_specification/broadcasting.html for details.
    """
    from arkouda.client import generic_msg

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

    Raises
    ------
    IndexError
        Raised if axis is not a valid axis for the given arrays.
    ValueError
        Raised if array shapes are incompatible with concat.

    Returns
    -------
    Array
        A new Array which is the concatention of the given Arrays along the given axis.

    """
    from arkouda.client import generic_msg
    from arkouda.numpy.util import _integer_axis_validation

    # Check array ranks

    ndim = arrays[0].ndim
    for a in arrays:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions to concatenate")

    # Check axis

    axis_ = axis
    if axis_ is not None:
        valid, axis_ = _integer_axis_validation(axis, ndim)
        if not valid:
            raise IndexError(f"{axis} is not a valid axis for array of rank {ndim}")

    # Make all arrays a common type
    # TODO: use a different approach to the common type.  The promotion function
    # below uses numpy.common_types, which returns float, even when given all ints.
    # That means this concat function will output a float, even if all arrays are ints.
    # numpy concat does not do that.

    (common_dt, _arrays) = promote_to_common_dtype([a._array for a in arrays])

    return Array._new(
        create_pdarray(
            cast(
                str,
                generic_msg(
                    cmd=(
                        f"concat<{common_dt},{ndim}>"
                        if axis_ is not None
                        else f"concatFlat<{common_dt},{ndim}>"
                    ),
                    args={
                        "n": len(arrays),
                        "names": _arrays,
                        "axis": axis_,
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

    Raises
    ------
    IndexError
        Raised if axis is not a valid axis for the given result.

    Returns
    -------
    Array
        A new Array with a new dimension equal to 1, inserted at the given axis.
    """
    from arkouda.client import generic_msg
    from arkouda.numpy.util import _integer_axis_validation

    # expand_dims and stack test the axis validity against an array of rank ndim+1

    valid, axis_ = _integer_axis_validation(axis, x.ndim + 1)
    if not valid:
        raise IndexError(f"{axis} is not a valid axis for expanding rank of {x.ndim}")

    try:
        return Array._new(
            create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"expandDims<{x.dtype},{x.ndim}>",
                        args={
                            "name": x._array,
                            "axis": axis_,
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

    Raises
    ------
    IndexError
        Raised if the axis/axes is/are invalid for the given array, or if the flip
        fails server-side.

    Returns
    -------
    Array
        A copy of x with the results reversed along the given axis or axes.
    """
    from arkouda.client import generic_msg
    from arkouda.numpy.util import _axis_validation

    axisList = []
    if axis is not None:
        valid, axisList = _axis_validation(axis, x.ndim)
        if not valid:
            raise IndexError(f"{axis} is not a valid axis/axes for array of rank {x.ndim}")

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


#   The checking for source and destination axis validity in moveaxis is virtually identical
#   whether the axes are supplied as integers or tuples, so it's done here.


def _check_moveaxis_axes(s, d, nd, check_function):
    valid_s, s_ = check_function(s, nd)
    valid_d, d_ = check_function(d, nd)
    if not (valid_s or valid_d):
        raise IndexError(f"Neither source {s} nor destination {d} are valid for rank {nd}")
    elif not valid_s:
        raise IndexError(f"Source {s} is not valid for rank {nd}")
    elif not valid_d:
        raise IndexError(f"Destination {d} is not valid for rank {nd}")
    return s_, d_


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

    Raises
    ------
    ValueError
        Raised if source and destination are not the same type (tuple or int).
    IndexError
        Raised if source, destination, or both are not valid for x.ndim.

    Returns
    -------
    Array
        A new Array with the axes shifted per the givern source, destination.
    """
    from arkouda.numpy.util import _axis_validation, _integer_axis_validation

    perm = list(range(x.ndim))

    # Do axis checking on the "both tuple" case.

    if isinstance(source, tuple) and isinstance(destination, tuple):
        source_, destination_ = _check_moveaxis_axes(source, destination, x.ndim, _axis_validation)
        for s, d in zip(source_, destination_):
            perm[s] = d

    # Do axis checking on the "both integer" case.

    elif isinstance(source, int) and isinstance(destination, int):
        source_, destination_ = _check_moveaxis_axes(
            source, destination, x.ndim, _integer_axis_validation
        )
        perm.pop(source_)
        perm.insert(destination_, source_)

    # Raise an error if they're not (both tuples) nor (both integers).

    else:
        raise ValueError("source and destination must both be integers or both be tuples.")

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

    Raises
    ------
    IndexError
        Raised if the given axes are not a valid reordering of the axes of x.

    Returns
    -------
    Array
        A copy of x with the axes permuted as per the axes argument.
    """
    from arkouda.client import generic_msg

    # Check axes, e.g. if x.ndim = 3, axes must be some permutation of 0,1,2.
    # If it is, then a sort will turn it into (0,1,2).

    if not (np.sort(axes) == np.arange(x.ndim)).all():
        raise IndexError(f"{axes} is not a valid permutation of axes for arrays of rank {x.ndim}")

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
        The axis along which to repeat elements. If None, the array is flattened before repeating,
        and each element is repeated repeats times.

    Raises
    ------
    NotYetImplementedError
        Raised if axis arg is used.

    Returns
    -------
    Array
        A new 1D array with each element of x repeated repeats times.
    """
    from arkouda.client import generic_msg

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

    Raises
    ------
    ValueError
        Raised if the given shape is invalid, or if more than one unknown dimension is specified.

    Returns
    -------
    Array
        A reshaped version of x, as specified in shape.

    """
    from arkouda.client import generic_msg

    # Check the validity of the new shape.  At most one -1 is allowed, as in numpy.

    if shape.count(-1) > 1:
        raise ValueError("can only specify one unknown dimension")

    # A -1 represents an "unknown shape."  That means "fill in the rest of it here,"
    # e.g. reshaping (24,1) to (3,2,-1) creates a shape of (3,2,4).

    elif shape.count(-1) == 1:
        partial_shape = -np.prod(shape)  # figure out what the -1 equates to
        if x.size % partial_shape != 0:  # raise error if it doesn't make sense (e.g. (12) to (7,-1)
            raise ValueError(f"cannot reshape array of size {x.size} into shape {shape}")
        else:  # but if it does make sense, fill it in as appropriate
            fillin = x.size // partial_shape
            shape_copy = [fillin if val == -1 else val for val in shape]
            shape = tuple(int(x) for x in shape_copy)  # avoids mypy "int vs np.int64" error

    # Finally, if there are no unknown shapes, just check the the new size matches the old.

    elif np.prod(shape) != x.size:  # e.g. can't reshape (4,3,2) to (3,3,3)
        raise ValueError(f"cannot reshape array of size {x.size} into shape {shape}")

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

    Raises
    ------
    IndexError
        Raised if the axis/axes aren't valid for the given array, or if axis and shift are both
        tuples but not of the same length, or if roll fails server-side.

    Returns
    -------
    Array
        An array with the same shape as x, but with elements shifted as per axis and shift.
    """
    from arkouda.client import generic_msg
    from arkouda.numpy.util import _axis_validation

    if isinstance(shift, tuple) and isinstance(axis, tuple) and (len(axis) != len(shift)):
        raise IndexError("When shift and axis are both tuples, they must have the same length.")

    axisList = []
    if axis is not None:
        valid, axisList = _axis_validation(axis, x.ndim)
        if not valid:
            raise IndexError(f"{axis} is not a valid axis/axes for array of rank {x.ndim}")

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

    Returns
    -------
    Array
        The input array, but with the axes specified in axis (which must be of length 1) removed.

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

    Raises
    ------
    ValueError
        Raised if the arrays aren't all the same shape.
    IndexError
        Raised if axis isn't valid for the given arrays.

    Returns
    -------
    Array
        A stacked array with rank 1 greater than the input arrays.
    """
    from arkouda.client import generic_msg
    from arkouda.numpy.util import _integer_axis_validation

    ndim = arrays[0].ndim

    base_shape = arrays[0].shape
    for a in arrays[1:]:
        if a.shape != base_shape:
            raise ValueError("all input arrays must have the same shape to stack")

    # expand_dims and stack test the axis validity against an array of rank ndim+1

    valid, axis_ = _integer_axis_validation(axis, ndim + 1)
    if not valid:
        raise IndexError(f"{axis} is not a valid axis for stacking arrays of rank {ndim}")

    # TODO: fix the type promotion here, as in concat above.  This is always producing
    # floats, even when all inputs are integer.  numpy stack doesn't do that.

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
                        "axis": axis_,
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
        repetitions tuple to make its length match the number of array dimensions.

    Returns
    -------
    Array
        The tiled output array.
    """
    from arkouda.client import generic_msg

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

    Raises
    ------
    IndexError
        Raised if the axis is not valid for the given Array.

    Returns
    -------
    Tuple
        A Tuple of unstacked Arrays.
    """
    from arkouda.client import generic_msg
    from arkouda.numpy.util import _integer_axis_validation

    valid, axis_ = _integer_axis_validation(axis, x.ndim)
    if not valid:
        raise IndexError(f"{axis} is not a valid axis for arrays of rank {x.ndim}")

    return tuple(
        Array._new(
            create_pdarrays(
                cast(
                    str,
                    generic_msg(
                        cmd=f"unstack<{x.dtype},{x.ndim}>",
                        args={
                            "name": x._array,
                            "axis": axis_,
                            "numReturnArrays": x.shape[axis],
                        },
                    ),
                )
            )
        )
    )
