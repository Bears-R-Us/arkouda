from ._array_object import Array

from arkouda.client import generic_msg
from arkouda.util import broadcast_dims
from arkouda.pdarrayclass import create_pdarray, broadcast_to_shape


def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Matrix product of two arrays.
    """
    from ._array_object import Array

    if x1._array.ndim < 2 and x2._array.ndim < 2:
        raise ValueError("matmul requires at least one array argument to have more than two dimensions")

    outShape = broadcast_dims(x1._array.shape, x2._array.shape)

    x1b = broadcast_to_shape(x1._array, outShape)
    x2b = broadcast_to_shape(x2._array, outShape)

    repMsg = generic_msg(
        cmd=f"matMul{len(outShape)}D",
        args={
            "x1": x1b.name,
            "x2": x2b.name,
        },
    )

    return Array._new(create_pdarray(repMsg))


def tensordot():
    raise ValueError("tensordot not implemented")


def matrix_transpose(x: Array) -> Array:
    """
    Matrix product of two arrays.
    """
    from ._array_object import Array

    if x._array.ndim < 2:
        raise ValueError("matrix_transpose requires the array to have more than two dimensions")

    repMsg = generic_msg(
        cmd=f"transpose{x._array.ndim}D",
        args={
            "array": x._array.name,
        },
    )

    return Array._new(create_pdarray(repMsg))


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    from ._array_object import Array

    outShape = broadcast_dims(x1._array.shape, x2._array.shape)

    x1b = broadcast_to_shape(x1._array, outShape)
    x2b = broadcast_to_shape(x2._array, outShape)

    repMsg = generic_msg(
        cmd=f"vecdot{len(outShape)}D",
        args={
            "x1": x1b.name,
            "x2": x2b.name,
            "bcShape": outShape,
            "axis": axis,
        },
    )

    return Array._new(create_pdarray(repMsg))
