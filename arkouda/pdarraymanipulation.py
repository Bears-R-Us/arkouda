from typing import Tuple, List, Literal, Union, Optional, Sequence
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.numpy.dtypes import dtype as akdtype

import numpy as np

__all__ = ["hstack", "vstack", "delete"]

@typechecked
def hstack(
    tup: Sequence[pdarray],
    *,
    dtype: Optional[Union[str, type]] = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> pdarray:
    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D arrays
    where it concatenates along the first axis. Rebuilds arrays divided by ``hsplit``.

    This function makes most sense for arrays with up to 3 dimensions. For instance, for pixel-data
    with a height (first axis), width (second axis), and r/g/b channels (third axis). The functions
    ``concatenate``, ``stack`` and ``block`` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of pdarray
        The arrays must have the same shape along all but the second axis, except 1-D arrays which
        can be any length. In the case of a single array_like input, it will be treated as a sequence of
        arrays; i.e., each element along the zeroth axis is treated as a separate array.
    dtype : str or type, optional
        If provided, the destination array will have this type.
    casting : {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional
        Controls what kind of data casting may occur. Defaults to ‘same_kind’. Currently unused.

    Returns
    -------
    pdarray
        The stacked array
    """

    if casting != "same_kind":
        # TODO: align with https://numpy.org/doc/stable/glossary.html#term-casting
        raise NotImplementedError(f"casting={casting} is not yet supported")

    # ensure all arrays have the same number of dimensions
    ndim = tup[0].ndim
    for a in tup:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions")

    # establish the dtype of the output array
    if dtype is None:
        dtype_ = np.result_type(*[np.dtype(a.dtype) for a in tup])
    else:
        dtype_ = akdtype(dtype)

    # cast the input arrays to the output dtype if necessary
    arrays = [a.astype(dtype_) if a.dtype != dtype_ else a for a in tup]

    if ndim == 1:
        return create_pdarray(
            generic_msg(
                cmd=f"concatenate<{np.dtype(dtype_).name},{len(arrays)}>",
                args={
                    "names": list(arrays),
                    "n": len(arrays),
                    "axis": 0,
                },
            )
        )

    # stack the arrays along the horizontal axis
    return create_pdarray(
        generic_msg(
            cmd=f"concatenate<{np.dtype(dtype_).name},{len(arrays)}>",
            args={
                "names": list(arrays),
                "n": len(arrays),
                "axis": 1,
            },
        )
    )

@typechecked
def vstack(
    tup: Union[Tuple[pdarray, ...], List[pdarray]],
    *,
    dtype: Optional[Union[type, str]] = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> pdarray:
    """
    Stack a sequence of arrays vertically (row-wise).

    This is equivalent to concatenation along the first axis after 1-D arrays of
    shape `(N,)` have been reshaped to `(1,N)`.

    Parameters
    ----------
    tup : Tuple[pdarray]
        The arrays to be stacked
    dtype : Optional[Union[type, str]], optional
        The data-type of the output array. If not provided, the output
        array will be determined using `np.common_type` on the
        input arrays Defaults to None
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"], optional
        Controls what kind of data casting may occur - currently unused

    Returns
    -------
    pdarray
        The stacked array
    """

    if casting != "same_kind":
        # TODO: align with https://numpy.org/doc/stable/glossary.html#term-casting
        raise NotImplementedError(f"casting={casting} is not yet supported")

    # ensure all arrays have the same number of dimensions
    ndim = tup[0].ndim
    for a in tup:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions")

    # establish the dtype of the output array
    if dtype is None:
        dtype_ = np.result_type(*[np.dtype(a.dtype) for a in tup])
    else:
        dtype_ = akdtype(dtype)

    # cast the input arrays to the output dtype if necessary
    arrays = [a.astype(dtype_) if a.dtype != dtype_ else a for a in tup]

    if ndim == 1:
        arrays = [a.reshape((1, len(tup[0]))) for a in arrays]
        return create_pdarray(
            generic_msg(
                cmd=f"concatenate<{np.dtype(dtype_).name},{len(arrays)}>",
                args={
                    "names": list(arrays),
                    "n": len(arrays),
                    "axis": 0,
                },
            )
        )

    # stack the arrays along the first axis
    return create_pdarray(
        generic_msg(
            cmd=f"concatenate<{np.dtype(dtype_).name},{len(arrays)}>",
            args={
                "names": list(arrays),
                "n": len(arrays),
                "axis": 0,
            },
        )
    )


@typechecked
def delete(
    arr: pdarray,
    obj: Union[pdarray, slice, int],
    axis: Optional[int] = None,
) -> pdarray:
    """
    Return a copy of 'arr' with elements along the specified axis removed.

    Parameters
    ----------
    arr : pdarray
        The array to remove elements from
    obj : Union[pdarray, slice, int]
        The indices to remove from 'arr'. If obj is a pdarray, it must
        have an integer dtype.
    axis : Optional[int], optional
        The axis along which to remove elements. If None, the array will
        be flattened before removing elements. Defaults to None.

    Returns
    -------
    pdarray
        A copy of 'arr' with elements removed
    """

    if axis is None:
        # flatten the array if axis is None
        _arr = create_pdarray(
            generic_msg(
                cmd=f"reshape{arr.ndim}Dx1D",
                args={
                    "name": arr,
                    "shape": (arr.size,),
                },
            )
        )
        _axis = 0
    else:
        _arr = arr
        _axis = axis

    if isinstance(obj, pdarray):
        return create_pdarray(
            generic_msg(
                cmd=f"delete{_arr.ndim}D",
                args={
                    "arr": _arr,
                    "obj": obj,
                    # TODO: maybe expose this as an optional argument? Or sort the array first?
                    "obj_sorted": False,
                    "axis": _axis,
                },
            )
        )
    else:
        if isinstance(obj, int):
            start = obj
            stop = obj + 1
            stride = 1
        elif isinstance(obj, slice):
            start, stop, stride = obj.indices(_arr.shape[_axis])
        else:
            raise ValueError("obj must be an integer, pdarray, or slice")

        return create_pdarray(
            generic_msg(
                cmd=f"deleteSlice{_arr.ndim}D",
                args={
                    "arr": _arr,
                    "start": start,
                    "stop": stop,
                    "stride": stride,
                    "axis": _axis,
                },
            )
        )
