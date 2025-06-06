# from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, cast, overload

from typeguard import typechecked

from arkouda.categorical import Categorical
from arkouda.client import generic_msg
from arkouda.numpy.dtypes import bool_scalars, int_scalars, numeric_scalars
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.strings import Strings

__all__ = ["flip", "repeat", "squeeze", "tile"]


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def flip(
    x: pdarray, /, *, axis: Optional[Union[int_scalars, Tuple[int_scalars, ...]]] = None
) -> pdarray: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def flip(
    x: Strings, /, *, axis: Optional[Union[int_scalars, Tuple[int_scalars, ...]]] = None
) -> Strings: ...


# docstr-coverage:excused `overload-only, docs live on impl`
@overload
def flip(
    x: Categorical, /, *, axis: Optional[Union[int_scalars, Tuple[int_scalars, ...]]] = None
) -> Categorical: ...


def flip(
    x: Union[pdarray, Strings, Categorical],
    /,
    *,
    axis: Optional[Union[int_scalars, Tuple[int_scalars, ...]]] = None,
) -> Union[pdarray, Strings, Categorical]:
    """
    Reverse an array's values along a particular axis or axes.

    Parameters
    ----------
    x : pdarray, Strings, or Categorical
        Reverse the order of elements in an array along the given axis.

        The shape of the array is preserved, but the elements are reordered.
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to flip the array. If None, flip the array along all axes.
    Returns
    -------
    pdarray, Strings, or Categorical
        An array with the entries of axis reversed.
    Note
    ----
    This differs from numpy as it actually reverses the data, rather than presenting a view.
    """
    axisList = []
    if axis is not None:
        axisList = list(axis) if isinstance(axis, tuple) else [axis]

    if isinstance(x, pdarray):
        try:
            return create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=(
                            f"flipAll<{x.dtype},{x.ndim}>"
                            if axis is None
                            else f"flip<{x.dtype},{x.ndim}>"
                        ),
                        args={
                            "name": x,
                            "nAxes": len(axisList),
                            "axis": axisList,
                        },
                    ),
                )
            )

        except RuntimeError as e:
            raise IndexError(f"Failed to flip array: {e}")
    elif isinstance(x, Categorical):
        if isinstance(x.permutation, pdarray):
            return Categorical.from_codes(
                codes=flip(x.codes),
                categories=x.categories,
                permutation=flip(x.permutation),
                segments=x.segments,
            )
        else:
            return Categorical.from_codes(
                codes=flip(x.codes),
                categories=x.categories,
                permutation=None,
                segments=x.segments,
            )

    elif isinstance(x, Strings):
        rep_msg = generic_msg(
            cmd="flipString", args={"objType": x.objType, "obj": x.entry, "size": x.size}
        )
        return Strings.from_return_msg(cast(str, rep_msg))
    else:
        raise TypeError("flip only accepts type pdarray, Strings, or Categorical.")


def repeat(
    a: Union[int, Sequence[int], pdarray],
    repeats: Union[int, Sequence[int], pdarray],
    axis: Union[None, int] = None,
) -> pdarray:
    """
    Repeat each element of an array after themselves

    Parameters
    ----------
    a : int, Sequence of int, or pdarray
        Input array.
    repeats: int, Sequence of int, or pdarray
        The number of repetitions for each element.
        `repeats` is broadcasted to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values.
        By default, use the flattened input array, and return a flat output array.

    Returns
    -------
    pdarray
        Output array which has the same shape as `a`, except along the given axis.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.repeat(3, 4)
    array([3 3 3 3])
    >>> x = ak.array([[1,2],[3,4]])
    >>> ak.repeat(x, 2)
    array([1 1 2 2 3 3 4 4])
    >>> ak.repeat(x, 3, axis=1)
    array([array([1 1 1 2 2 2]) array([3 3 3 4 4 4])])
    >>> ak.repeat(x, [1, 2], axis=0)
    array([array([1 2]) array([3 4]) array([3 4])])
    """
    from arkouda.pdarrayclass import any as akany

    if isinstance(repeats, int):
        ak_repeats = ak_array([repeats], int)
        if isinstance(ak_repeats, pdarray):
            repeats = ak_repeats
        else:
            raise TypeError("This should never happen because repeats was an int.")
    elif isinstance(repeats, Sequence):
        ak_repeats = ak_array(repeats, int)
        if isinstance(ak_repeats, pdarray):
            repeats = ak_repeats
        else:
            raise TypeError("This should never happen because repeats was a Sequence of int.")
    if isinstance(a, int):
        ak_a = ak_array([a], int)
        if isinstance(ak_a, pdarray):
            a = ak_a
        else:
            raise TypeError("This should never happen because a was an int.")
    elif isinstance(a, Sequence):
        ak_a = ak_array(a, int)
        if isinstance(ak_a, pdarray):
            a = ak_a
        else:
            raise TypeError("This should never happen because a was a Sequence of int.")
    if repeats.ndim > 1:
        raise ValueError(
            f"Expected repeats to be a 1-dimensional array or constant, but "
            f"received {repeats.ndim}-dimensional array instead."
        )
    if akany(repeats < 0):
        raise ValueError("repeats may not contain negative values.")
    if not akany(repeats > 0):
        temp = cast(pdarray, ak_array([], a.dtype))
        temp_shape = list(a.shape)
        if axis is None:
            return temp
        elif isinstance(axis, int):
            temp_shape[axis] = 0
            temp = temp.reshape(temp_shape)
            return temp
        else:
            raise TypeError("Axis should have been None or an int")
    if axis is None:
        try:
            return create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"repeatFlat<{a.dtype},{a.ndim}>",
                        args={
                            "name": a,
                            "repeats": repeats,
                        },
                    ),
                )
            )

        except RuntimeError as e:
            raise ValueError(f"Failed to repeat array: {e}")
    if repeats.size != 1 and repeats.size != a.shape[axis]:
        raise ValueError(
            f"repeats must either be a constant or match the length of a in axis. "
            f"Instead, repeats size of {repeats.size} != {a.shape[axis]}"
        )
    try:
        return create_pdarray(
            cast(
                str,
                generic_msg(
                    cmd=f"repeat<{a.dtype},{a.ndim},{repeats.dtype},{1}>",
                    args={
                        "eIn": a,
                        "reps": repeats,
                        "axis": axis,
                    },
                ),
            )
        )

    except RuntimeError as e:
        raise ValueError(f"Failed to repeat array: {e}")


@typechecked
def squeeze(
    x: Union[pdarray, numeric_scalars, bool_scalars], /, axis: Union[None, int, Tuple[int, ...]] = None
) -> pdarray:
    """
    Remove degenerate (size one) dimensions from an array.

    Parameters
    ----------
    x : pdarray
        The array to squeeze
    axis : int or Tuple[int, ...]
        The axis or axes to squeeze (must have a size of one).
        If axis = None, all dimensions of size 1 will be squeezed.

    Returns
    -------
    pdarray
        A copy of x with the dimensions specified in the axis argument removed.

    Examples
    --------
    >>> import arkouda as ak
    >>> x = ak.arange(10).reshape((1, 10, 1))
    >>> x.shape
    (1, 10, 1)
    >>> ak.squeeze(x, axis=None).shape
    (10,)
    >>> ak.squeeze(x, axis=2).shape
    (1, 10)
    >>> ak.squeeze(x, axis=(0, 2)).shape
    (10,)

    """
    from arkouda.numpy.dtypes import _val_isinstance_of_union

    if _val_isinstance_of_union(x, numeric_scalars) or _val_isinstance_of_union(x, bool_scalars):
        ret = ak_array([x])
        if isinstance(ret, pdarray):
            return ret

    if isinstance(x, pdarray):
        if axis is None:
            _axis = [i for i in range(x.ndim) if x.shape[i] == 1]
            #   Can't squeeze over every dimension, so remove one if necessary
            if len(_axis) == len(x.shape):
                _axis.pop()
            axis = tuple(_axis)

        nAxes = len(axis) if isinstance(axis, tuple) else 1
        try:
            return create_pdarray(
                cast(
                    str,
                    generic_msg(
                        cmd=f"squeeze<{x.dtype},{x.ndim},{x.ndim - nAxes}>",
                        args={
                            "name": x,
                            "nAxes": nAxes,
                            "axes": list(axis) if isinstance(axis, tuple) else [axis],
                        },
                    ),
                )
            )

        except RuntimeError as e:
            raise ValueError(f"Failed to squeeze array: {e}")

    raise RuntimeError("Failed to squeeze array.")


def tile(A: pdarray, /, reps: Union[int, Tuple[int, ...]]) -> pdarray:
    """
    Construct an array by repeating A the number of times given by reps.

    If reps has length ``d``, the result will have dimension of ``max(d, A.ndim)``.

    If ``A.ndim < d``, A is promoted to be d-dimensional by prepending new axes. So a shape (3,) \
array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. \
If this is not the desired behavior, promote A to d-dimensions manually before calling this function.

    If ``A.ndim > d``, reps is promoted to A.ndim by prepending 1’s to it. \
Thus for an A of shape (2, 3, 4, 5), a reps of (2, 2) is treated as (1, 1, 2, 2).

    Parameters
    ----------
    A : pdarray
        The input pdarray to be tiled
    reps : int or Tuple of int
        The number of repetitions of A along each axis.

    Returns
    -------
    pdarray
        A new pdarray with the tiled data.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([0, 1, 2])
    >>> ak.tile(a, 2)
    array([0 1 2 0 1 2])
    >>> ak.tile(a, (2, 2))
    array([array([0 1 2 0 1 2]) array([0 1 2 0 1 2])])
    >>> ak.tile(a, (2, 1, 2))
    array([array([array([0 1 2 0 1 2])]) array([array([0 1 2 0 1 2])])])

    >>> b = ak.array([[1, 2], [3, 4]])
    >>> ak.tile(b, 2)
    array([array([1 2 1 2]) array([3 4 3 4])])
    >>> ak.tile(b, (2, 1))
    array([array([1 2]) array([3 4]) array([1 2]) array([3 4])])

    >>> c = ak.array([1, 2, 3, 4])
    >>> ak.tile(c, (4, 1))
    array([array([1 2 3 4]) array([1 2 3 4]) array([1 2 3 4]) array([1 2 3 4])])
    """
    # Ensure 'reps' is a list
    if isinstance(reps, int):
        reps_2 = [cast(int, reps)]
        l_reps = 1
    else:
        reps_2 = list(cast(tuple, reps))
        l_reps = len(reps)

    A_shape = A.shape
    dim_difference = abs(len(A_shape) - l_reps)
    if len(A_shape) < l_reps:
        A = A.reshape((1,) * dim_difference + A_shape)
    elif len(A_shape) > l_reps:
        reps_2 = [1] * dim_difference + reps_2

    # Construct the command to send to the server
    cmd = f"tile<{A.dtype},{A.ndim}>"
    args = {"name": A, "reps": reps_2}

    # Send the command to the Arkouda server
    rep_msg = generic_msg(cmd=cmd, args=args)

    # Create and return the resulting pdarray
    return create_pdarray(cast(str, rep_msg))
