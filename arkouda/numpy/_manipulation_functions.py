# from __future__ import annotations

from typing import Optional, Tuple, Union, cast

from typeguard import typechecked

from arkouda.categorical import Categorical
from arkouda.client import generic_msg
from arkouda.numpy.dtypes import numeric_scalars, bool_scalars
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.pdarraycreation import array as ak_array
from arkouda.strings import Strings

__all__ = ["flip", "squeeze"]


def flip(
    x: Union[pdarray, Strings, Categorical], /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None
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
    >>> ak.connect()
    >>> x = ak.arange(10).reshape((1, 10, 1))
    >>> x
    array([array([array([0]) array([1]) array([2]) array([3])....
     array([4]) array([5]) array([6]) array([7]) array([8]) array([9])])])
    >>> x.shape
    (1, 10, 1)
    >>> ak.squeeze(x,axis=None)
    array([0 1 2 3 4 5 6 7 8 9])
    >>> ak.squeeze(x,axis=None).shape
    (10,)
    >>> ak.squeeze(x,axis=2)
    array([array([0 1 2 3 4 5 6 7 8 9])])
    >>> ak.squeeze(x,axis=2).shape
    (1, 10)
    >>> ak.squeeze(x,axis=(0,2))
    array([0 1 2 3 4 5 6 7 8 9])
    >>> ak.squeeze(x,axis=(0,2)).shape
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
