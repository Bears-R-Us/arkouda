from typing import Literal, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import numpy as np

from typeguard import typechecked

from arkouda.numpy.dtypes import bigint
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import result_type as ak_result_type
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import arange, array


__all__ = ["hstack", "vstack", "delete", "append"]


def _max_bits_list(pda_list: Sequence[pdarray]) -> Tuple[bool, int]:
    """
    Finds the minimum `max_bits` when there are bigint arrays in the input.

    Determines whether any arrays in the input list use the `bigint` dtype
    and returns the minimum bit width among those that do.

    Parameters
    ----------
        pda_list : Sequence[pdarray]
            A sequence of `pdarray` objects to examine.

    Returns
    -------
        bool
            A boolean indicating whether any array uses the `bigint` dtype.
        int
            An integer representing the smallest `max_bits` value among the
            `bigint` arrays. Returns -1 if no `bigint` arrays are present.
    """
    has_bigint = False
    m_bits = -1
    do_warn = False
    for a in pda_list:
        if a.dtype == bigint:
            if has_bigint:
                if a.max_bits != m_bits:
                    do_warn = True
            else:
                has_bigint = True
            curr_bits = a.max_bits
            if curr_bits > 0 and (m_bits == -1 or curr_bits < m_bits):
                m_bits = curr_bits
    if do_warn:
        warn("Because two arrays with different max_bits were used, truncating to smaller max_bits")
    return has_bigint, m_bits


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
        The array formed by stacking the given arrays.

    See Also
    --------
    concatenate, stack, block, vstack, dstack, column_stack, hsplit, unstack

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([1, 2, 3])
    >>> b = ak.array([4, 5, 6])
    >>> ak.hstack((a, b))
    array([1 2 3 4 5 6])
    >>> a = ak.array([[1],[2],[3]])
    >>> b = ak.array([[4],[5],[6]])
    >>> ak.hstack((a, b))
    array([array([1 4]) array([2 5]) array([3 6])])
    """
    from arkouda.core.client import generic_msg

    if casting != "same_kind":
        # TODO: align with https://numpy.org/doc/stable/glossary.html#term-casting
        raise NotImplementedError(f"casting={casting} is not yet supported")

    # ensure all arrays have the same number of dimensions
    ndim = tup[0].ndim
    for a in tup:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions")

    has_bigint, m_bits = _max_bits_list(tup)

    # establish the dtype of the output array
    if has_bigint and dtype is None:
        dtype = bigint
    if dtype is None:
        dtype_ = np.result_type(*[np.dtype(a.dtype) for a in tup])
    else:
        dtype_ = akdtype(dtype)

    # cast the input arrays to the output dtype if necessary
    arrays = [a.astype(dtype_) if a.dtype != dtype_ else a for a in tup]

    if has_bigint:
        for i in range(len(arrays)):
            arrays[i].max_bits = m_bits

    offsets = [0 for _ in range(len(arrays))]

    if ndim == 1:
        for i in range(1, len(arrays)):
            offsets[i] = offsets[i - 1] + arrays[i - 1].shape[0]
        return create_pdarray(
            generic_msg(
                cmd=f"concatenate<{akdtype(dtype_).name},{arrays[0].ndim}>",
                args={
                    "names": list(arrays),
                    "axis": 0,
                    "offsets": offsets,
                },
            )
        )

    for i in range(1, len(arrays)):
        offsets[i] = offsets[i - 1] + arrays[i - 1].shape[1]

    # stack the arrays along the horizontal axis
    return create_pdarray(
        generic_msg(
            cmd=f"concatenate<{akdtype(dtype_).name},{arrays[0].ndim}>",
            args={
                "names": list(arrays),
                "axis": 1,
                "offsets": offsets,
            },
        )
    )


@typechecked
def vstack(
    tup: Sequence[pdarray],
    *,
    dtype: Optional[Union[str, type]] = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> pdarray:
    """
    Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after
    1-D arrays of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by ``vsplit``.

    This function makes most sense for arrays with up to 3 dimensions.
    For instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions ``concatenate``, ``stack`` and ``block``
    provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of pdarray
        The arrays must have the same shape along all but the first axis. 1-D arrays
        must have the same length. In the case of a single array_like input, it will be
        treated as a sequence of arrays; i.e., each element along the zeroth axis is treated
        as a separate array.
    dtype : str or type, optional
        If provided, the destination array will have this dtype.
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"], optional
        Controls what kind of data casting may occur. Defaults to ‘same_kind’. Currently unused.

    Returns
    -------
    pdarray
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    concatenate, stack, block, hstack, dstack, column_stack, hsplit, unstack

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([1, 2, 3])
    >>> b = ak.array([4, 5, 6])
    >>> ak.vstack((a, b))
    array([array([1 2 3]) array([4 5 6])])

    >>> a = ak.array([[1],[2],[3]])
    >>> b = ak.array([[4],[5],[6]])
    >>> ak.vstack((a, b))
    array([array([1]) array([2]) array([3]) array([4]) array([5]) array([6])])

    """
    from arkouda.core.client import generic_msg

    if casting != "same_kind":
        # TODO: align with https://numpy.org/doc/stable/glossary.html#term-casting
        raise NotImplementedError(f"casting={casting} is not yet supported")

    # From docstring: "This is equivalent to concatenation along the first axis after 1-D arrays
    # of shape (N,) have been reshaped to (1,N)."
    arrays = [a if a.ndim != 1 else a.reshape((1, len(a))) for a in tup]

    # ensure all arrays have the same number of dimensions
    ndim = arrays[0].ndim
    for a in arrays:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions")

    has_bigint, m_bits = _max_bits_list(tup)

    # establish the dtype of the output array
    if has_bigint and dtype is None:
        dtype = bigint
    if dtype is None:
        dtype_ = np.result_type(*[np.dtype(a.dtype) for a in arrays])
    else:
        dtype_ = akdtype(dtype)

    # cast the input arrays to the output dtype if necessary
    arrays = [a.astype(dtype_) if a.dtype != dtype_ else a for a in arrays]

    if has_bigint:
        for i in range(len(arrays)):
            arrays[i].max_bits = m_bits

    offsets = [0 for _ in range(len(arrays))]

    for i in range(1, len(arrays)):
        offsets[i] = offsets[i - 1] + arrays[i - 1].shape[0]

    # stack the arrays along the first axis
    return create_pdarray(
        generic_msg(
            cmd=f"concatenate<{akdtype(dtype_).name},{arrays[0].ndim}>",
            args={
                "names": list(arrays),
                "axis": 0,
                "offsets": offsets,
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

    Examples
    --------
    >>> import arkouda as ak
    >>> arr = ak.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    >>> arr
    array([array([1 2 3 4]) array([5 6 7 8]) array([9 10 11 12])])
    >>> ak.delete(arr, 1, 0)
    array([array([1 2 3 4]) array([9 10 11 12])])

    >>> ak.delete(arr, slice(0, 4, 2), 1)
    array([array([2 4]) array([6 8]) array([10 12])])
    >>> ak.delete(arr, [1, 3, 5], None)
    array([1 3 5 7 8 9 10 11 12])
    """
    from arkouda.core.client import generic_msg

    shape = arr.shape

    if axis is None and arr.ndim != 1:
        # flatten the array if axis is None
        _arr = arr.flatten()
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
        _del = obj
    elif isinstance(obj, Sequence):
        _del = cast(pdarray, array(obj))
    else:
        if isinstance(obj, int):
            start = obj
            stop = obj + 1
            stride = 1
        elif isinstance(obj, slice):
            start, stop, stride = obj.indices(_arr.shape[_axis])
        else:
            raise ValueError("obj must be a slice, int, Sequence of int, Sequence of bool, or pdarray")
        _del = arange(start, stop, stride)
    if _del.dtype == int and (shape[_axis] / max(int(_del.size), 1)) * slice_weight >= 100:
        alg_choice = "BulkCopy"
    else:
        alg_choice = "AggCopy"
    return create_pdarray(
        generic_msg(
            cmd=f"delete{alg_choice}<{_arr.dtype},{_arr.ndim},{_del.dtype},{_del.ndim}>",
            args={
                "eIn": _arr,
                "axis": _axis,
                "del": _del,
            },
        )
    )


@typechecked
def append(
    arr: pdarray,
    values: pdarray,
    axis: Optional[int] = None,
) -> pdarray:
    """
    Append values to the end of an array.

    Parameters
    ----------
    arr : pdarray
        Values are appended to a copy of this array.
    values : pdarray
        These values are appended to a copy of arr.
        It must be of the correct shape (the same shape as arr, excluding axis).
        If axis is not specified, values can be any shape and will be flattened before use.
    axis : Optional[int], default=None
        The axis along which values are appended.
        If axis is not given, both arr and values are flattened before use.

    Returns
    -------
    pdarray
        A copy of arr with values appended to axis.
        Note that append does not occur in-place: a new array is allocated and filled.
        If axis is None, out is a flattened array.

    See Also
    --------
    delete

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.array([1, 2, 3])
    >>> b = ak.array([[4, 5, 6], [7, 8, 9]])
    >>> ak.append(a, b)
    array([1 2 3 4 5 6 7 8 9])
    >>> ak.append(b, b, axis = 0)
    array([array([4 5 6]) array([7 8 9]) array([4 5 6]) array([7 8 9])])
    """
    from arkouda.core.client import generic_msg

    if axis is None:
        axis = 0
        if arr.ndim != 1:
            arr = arr.flatten()
        if values.ndim != 1:
            values = values.flatten()
    else:
        # ensure both arrays have the same number of dimensions
        if arr.ndim != values.ndim:
            raise ValueError("all input arrays must have the same number of dimensions")
        if axis >= arr.ndim or axis < -arr.ndim:
            raise ValueError(f"Axis {axis} out of bounds for {arr.ndim} dimensions")
        axis = cast(int, (axis + arr.ndim) % arr.ndim)

    has_bigint, m_bits = _max_bits_list([arr, values])

    # establish the dtype of the output array
    dtype_ = ak_result_type(arr, values)

    # cast the input arrays to the output dtype if necessary
    arrays = [a.astype(dtype_) if a.dtype != dtype_ else a for a in (arr, values)]

    if has_bigint and akdtype(dtype_).name == "bigint":
        for i in range(len(arrays)):
            arrays[i].max_bits = m_bits

    offsets = [0 for _ in range(len(arrays))]

    for i in range(1, len(arrays)):
        offsets[i] = offsets[i - 1] + arrays[i - 1].shape[axis]

    # stack the arrays along the given axis
    return create_pdarray(
        generic_msg(
            cmd=f"concatenate<{akdtype(dtype_).name},{arrays[0].ndim}>",
            args={
                "names": list(arrays),
                "axis": axis,
                "offsets": offsets,
            },
        )
    )
