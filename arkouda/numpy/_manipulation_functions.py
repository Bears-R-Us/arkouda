# from __future__ import annotations

from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray
from arkouda.pdarrayclass import pdarray
from arkouda.strings import Strings
from arkouda.categorical import Categorical


__all__ = ["flip"]


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
                cmd="flipString", args={"objType": x.objType, "obj": x.entry,  "size": x.size}
            )
        return Strings.from_return_msg(cast(str, rep_msg))
    else:
        raise TypeError("flip only accepts type pdarray, Strings, or Categorical.")
