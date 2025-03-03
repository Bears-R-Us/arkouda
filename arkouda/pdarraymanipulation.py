from typing import Literal, Optional, Sequence, Union

import numpy as np
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.numpy.dtypes import bigint
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.pdarrayclass import create_pdarray, pdarray

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
        The array formed by stacking the given arrays.

    See Also
    --------
    concatenate, stack, block, vstack, dstack, column_stack, hsplit, unstack

    Examples
    --------
    >>> a = ak.array([1, 2, 3])
    >>> b = ak.array([4, 5, 6])
    >>> ak.hstack((a, b))
    array([1 2 3 4 5 6])
    >>> a = ak.array([[1],[2],[3]])
    >>> b = ak.array([[4],[5],[6]])
    >>> ak.hstack((a, b))
    array([array([1 4]) array([2 5]) array([3 6])])
    """

    if casting != "same_kind":
        # TODO: align with https://numpy.org/doc/stable/glossary.html#term-casting
        raise NotImplementedError(f"casting={casting} is not yet supported")

    # ensure all arrays have the same number of dimensions
    ndim = tup[0].ndim
    for a in tup:
        if a.ndim != ndim:
            raise ValueError("all input arrays must have the same number of dimensions")

    has_bigint = False
    for a in tup:
        if a.dtype == bigint:
            has_bigint = True
            # Should this be min or max?
            try:
                m_bits = min([a.max_bits for a in tup if a.dtype == bigint and a.max_bits > 0])
            except ValueError:
                m_bits = -1
            break

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
                    "n": len(arrays),
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
                "n": len(arrays),
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
    >>> a = ak.array([1, 2, 3])
    >>> b = ak.array([4, 5, 6])
    >>> ak.vstack((a, b))
    array([array([1 2 3]) array([4 5 6])])

    >>> a = ak.array([[1],[2],[3]])
    >>> b = ak.array([[4],[5],[6]])
    >>> ak.vstack((a, b))
    array([array([1]) array([2]) array([3]) array([4]) array([5]) array([6])])

    """

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

    has_bigint = False
    for a in arrays:
        if a.dtype == bigint:
            has_bigint = True
            # Should this be min or max?
            try:
                m_bits = min([a.max_bits for a in arrays if a.dtype == bigint and a.max_bits > 0])
            except ValueError:
                m_bits = -1
            break

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
                "n": len(arrays),
                "axis": 0,
                "offsets": offsets,
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
