from __future__ import annotations


from typing import TYPE_CHECKING, List, Optional, Tuple, Union, cast

from arkouda.client import generic_msg
import numpy as np
from arkouda.pdarrayclass import create_pdarray, pdarray
from arkouda.pdarraycreation import _array_memview
from arkouda.dtypes import dtype as akdtype
from arkouda.dtypes import resolve_scalar_dtype
from arkouda.client import maxTransferBytes

if TYPE_CHECKING:
    from ._typing import (
        Array,
        Device,
        Dtype,
        NestedSequence,
        SupportsBufferProtocol,
    )
from ._dtypes import _all_dtypes

import arkouda as ak


def _check_valid_dtype(dtype):
    # Note: Only spelling dtypes as the dtype objects is supported.

    # We use this instead of "dtype in _all_dtypes" because the dtype objects
    # define equality with the sorts of things we want to disallow.
    for d in (None,) + _all_dtypes:
        if dtype is d:
            return
    raise ValueError("dtype must be one of the supported dtypes")


def asarray(
    obj: Union[
        Array,
        bool,
        int,
        float,
        NestedSequence[bool | int | float],
        SupportsBufferProtocol,
        ak.pdarray,
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> Array:
    from ._array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(obj, ak.pdarray):
        return Array._new(obj)
    elif (
        isinstance(obj, bool)
        or isinstance(obj, int)
        or isinstance(obj, float)
        or isinstance(obj, complex)
    ):
        if dtype is None:
            xdtype = akdtype(resolve_scalar_dtype(obj))
        else:
            xdtype = akdtype(dtype)
        res = ak.full(1, obj, xdtype)
        return Array._new(res)
    elif isinstance(obj, Array):
        return Array._new(ak.array(obj._array))
    elif isinstance(obj, ak.pdarray):
        return Array._new(obj)
    elif isinstance(obj, np.ndarray):
        obj_flat = obj.flatten()
        if dtype is None:
            xdtype = akdtype(obj_flat.dtype)
        else:
            xdtype = akdtype(dtype)

        if obj_flat.nbytes > maxTransferBytes:
            raise RuntimeError(
                f"Creating Array would require transferring {obj_flat.nbytes} bytes, which exceeds "
                f"allowed transfer size. Increase ak.client.maxTransferBytes to force."
            )

        if obj.shape == ():
            return Array._new(
                create_pdarray(
                    generic_msg(
                        cmd="create0D",
                        args={"dtype": xdtype, "value": obj.item()},
                    )
                )
            )
        else:
            return Array._new(
                create_pdarray(
                    generic_msg(
                        cmd=f"array{obj.ndim}D",
                        args={"dtype": xdtype, "shape": np.shape(obj) , "seg_string": False},
                        payload=_array_memview(obj_flat),
                        send_binary=True,
                    )
                )
            )
    else:
        raise ValueError("asarray not implemented for 'NestedSequence'")


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if stop is None:
        return Array._new(ak.arange(0, start, step, dtype=dtype))
    else:
        return Array._new(ak.arange(start, stop, step, dtype=dtype))


def empty(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(shape, tuple):
        size = 1
        tshape = cast(Tuple, shape)
        for s in tshape:
            size *= s
        return Array._new(
            pdarray("__empty__", akdtype(dtype), size, len(tshape), tshape, 0, None),
            empty=True,
        )
    else:
        vshape = cast(int, shape)
        return Array._new(
            pdarray("__empty__", akdtype(dtype), vshape, 1, (vshape,), 0, None),
            empty=True,
        )


def empty_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    from ._array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    t = x.dtype if dtype is None else akdtype(dtype)

    return Array._new(
        pdarray(
            "__empty__",
            t,
            x._array.size,
            x._array.ndim,
            x._array.shape,
            x._array.itemsize,
            x._array.max_bits,
        ),
        empty=True,
    )


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    cols = n_rows
    if n_cols is not None:
        cols = n_cols

    repMsg = generic_msg(
        cmd="eye",
        args={
            "dtype": np.dtype(dtype).name,
            "rows": n_rows,
            "cols": cols,
            "diag": k,
        },
    )

    return Array._new(create_pdarray(repMsg))


def from_dlpack(x: object, /) -> Array:
    raise ValueError("Not implemented")


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    a = zeros(shape, dtype=dtype, device=device)
    a._array.fill(fill_value)
    return a


def full_like(
    x: Array,
    /,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return full(x.shape, fill_value, dtype=dtype, device=device)


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> Array:
    from ._array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    return Array._new(ak.linspace(start, stop, num))


def meshgrid(*arrays: Array, indexing: str = "xy") -> List[Array]:
    # if indexing not in ['xy', 'ij']:
    #     raise ValueError(
    #         "Valid values for `indexing` are 'xy' and 'ij'.")

    # array_names = "["
    # first = True
    # dim = 1
    # for a in arrays:
    #     if first:
    #         dim = a._array.ndim
    #         first = False
    #     else:
    #         if a._array.dim != dim:
    #             raise ValueError(f"all arrays must have the same dimensionality for 'meshgrid'")
    #         array_names += ","
    #     array_names += x._array.name
    # array_names += "]"

    # repMsg = generic_msg(
    #     cmd=f"meshgrid{dim}D",
    #     args={
    #         "num": len(array_names),
    #         "arrays": array_names,
    #         "indexing": indexing,
    #     },
    # )

    # arrayMsgs = repMsg.split(",")
    # return [Array._new(create_pdarray(msg)) for msg in arrayMsgs]
    raise ValueError("Not implemented")


def ones(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    a = zeros(shape, dtype=dtype, device=device)
    a._array.fill(1)
    return a


def ones_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    return ones(x.shape, dtype=dtype, device=device)


def tril(x: Array, /, *, k: int = 0) -> Array:
    from ._array_object import Array

    repMsg = generic_msg(
        cmd=f"tril{x._array.ndim}D",
        args={
            "array": x._array.name,
            "diag": k,
        },
    )

    return Array._new(create_pdarray(repMsg))


def triu(x: Array, /, *, k: int = 0) -> Array:
    from ._array_object import Array

    repMsg = generic_msg(
        cmd=f"triu{x._array.ndim}D",
        args={
            "array": x._array.name,
            "diag": k,
        },
    )

    return Array._new(create_pdarray(repMsg))


def zeros(
    shape: Union[int, Tuple[int, ...]],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    from ._array_object import Array

    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

    if isinstance(shape, tuple):
        ndim = len(shape)
    else:
        if shape == 0:
            ndim = 0
        else:
            ndim = 1

    dtype = akdtype(dtype)  # normalize dtype
    dtype_name = cast(np.dtype, dtype).name

    repMsg = generic_msg(
        cmd=f"create{ndim}D",
        args={
            "dtype": dtype_name,
            "shape": shape,
            "value": 0,
        },
    )

    return Array._new(create_pdarray(repMsg))


def zeros_like(
    x: Array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> Array:
    return zeros(x.shape, dtype=dtype, device=device)
