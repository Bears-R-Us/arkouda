from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import arange, array

__all__ = ["vstack", "delete"]


@typechecked
def vstack(
    tup: Union[Tuple[pdarray], List[pdarray]],
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
        # TODO: wasn't clear from the docs what each of the casting options does
        raise NotImplementedError(f"casting={casting} is not yet supported")

    # ensure all arrays have the same number of dimensions
    ndim = tup[0].ndim
    for a in tup:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions")

    # establish the dtype of the output array
    if dtype is None:
        dtype_ = np.common_type(*[np.empty(0, dtype=a.dtype) for a in tup])
    else:
        dtype_ = akdtype(dtype)

    # cast the input arrays to the output dtype if necessary
    arrays = [a.astype(dtype_) if a.dtype != dtype_ else a for a in tup]

    # stack the arrays along the first axis
    return create_pdarray(
        generic_msg(
            cmd=f"stack{ndim}D",
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
    obj: Union[slice, int, Sequence[int], Sequence[bool], pdarray],
    axis: Optional[int] = None,
) -> pdarray:
    """
    Return a copy of 'arr' with elements along the specified axis removed.

    Parameters
    ----------
    arr : pdarray
        The array to remove elements from
    obj : slice, int, Sequence of int, Sequence of bool, or pdarray
        The indices to remove from 'arr'. If obj is a pdarray, it must
        have an integer or bool dtype.
    axis : Optional[int], optional
        The axis along which to remove elements. If None, the array will
        be flattened before removing elements. Defaults to None.

    Returns
    -------
    pdarray
        A copy of 'arr' with elements removed
    """

    shape = arr.shape

    if axis is None and arr.ndim != 1:
        # flatten the array if axis is None
        _arr = create_pdarray(
            generic_msg(
                cmd=f"reshape<{arr.dtype},{arr.ndim},1>",
                args={
                    "name": arr.name,
                    "shape": arr.size,
                },
            ),
        )
        _axis = 0
        shape = _arr.shape
    elif axis is None:
        _axis = 0
        _arr = arr
    else:
        _arr = arr
        _axis = axis
    slice_weight = 1
    for i in range(_axis + 1, len(shape)):
        slice_weight *= shape[i]
    if isinstance(obj, pdarray):
        if obj.dtype == int and (shape[_axis] / max(obj.size, 1)) * slice_weight >= 100:
            alg_choice = "BulkCopy"
        else:
            alg_choice = "AggCopy"
        return create_pdarray(
            generic_msg(
                cmd=f"delete{alg_choice}<{_arr.dtype},{_arr.ndim},{obj.dtype},{obj.ndim}>",
                args={
                    "eIn": _arr,
                    "del": obj,
                    "axis": _axis,
                },
            )
        )
    elif isinstance(obj, Sequence):
        obj2 = array(obj)
        if obj2.dtype == int and (shape[_axis] / max(obj2.size, 1)) * slice_weight >= 100:
            alg_choice = "BulkCopy"
        else:
            alg_choice = "AggCopy"
        return create_pdarray(
            generic_msg(
                cmd=f"delete{alg_choice}<{_arr.dtype},{_arr.ndim},{obj2.dtype},{obj2.ndim}>",
                args={
                    "eIn": _arr,
                    "del": obj2,
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
            raise ValueError("obj must be a slice, int, Sequence of int, or Sequence of bool")
        del_slice = arange(start, stop, stride)
        if del_slice.dtype == int and (shape[_axis] / max(del_slice.size, 1)) * slice_weight >= 100:
            alg_choice = "BulkCopy"
        else:
            alg_choice = "AggCopy"
        return create_pdarray(
            generic_msg(
                cmd=f"delete{alg_choice}<{_arr.dtype},{_arr.ndim},{del_slice.dtype},{del_slice.ndim}>",
                args={
                    "eIn": _arr,
                    "axis": _axis,
                    "del": del_slice,
                },
            )
        )
