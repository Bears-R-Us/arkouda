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

    x1b, x2b = broadcast_if_needed(x1._array, x2._array)

    repMsg = generic_msg(
        cmd=f"matMul{len(x1b.shape)}D",
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

    x1b, x2b = broadcast_if_needed(x1._array, x2._array)

    repMsg = generic_msg(
        cmd=f"vecdot{len(x1b.shape)}D",
        args={
            "x1": x1b.name,
            "x2": x2b.name,
            "bcShape": x1b.shape,
            "axis": axis,
        },
    )

    return Array._new(create_pdarray(repMsg))
