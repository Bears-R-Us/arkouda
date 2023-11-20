import arkouda as ak
from arkouda.client import generic_msg
from arkouda.util import broadcast_dims

def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Matrix product of two arrays.
    """
    from ._array_object import Array

    if x1._array.ndim < 2 and x2._array.ndim < 2:
        raise ValueError("matmul requires at least one array argument to have more than two dimensions")

    outShape = broadcast_dims(x1._array.shape, x2._array.shape)

    x1b = x1._array.broadcast_to_shape(outShape)
    x2b = x2._array.broadcast_to_shape(outShape)

    repMsg = generic_msg(
        cmd=f"matMul{len(outShape)}D",
        args={
            "x1": x1b.name,
            "x2": x2b.name,
        },
    )

    return Array._new(create_pdarray(repMsg))

def tensordot():
    return 0

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
            "array": x.name,
        },
    )

def vecdot(x1: Array, x2: Array, /) -> Array:
    from ._array_object import Array

    outShape = broadcast_dims(x1._array.shape, x2._array.shape)

    x1b = x1._array.broadcast_to_shape(outShape)
    x2b = x2._array.broadcast_to_shape(outShape)

    repMsg = generic_msg(
        cmd=f"vecdot{len(outShape)}D",
        args={
            "x1": x1b.name,
            "x2": x2b.name,
        },
    )

    return Array._new(create_pdarray(repMsg))
