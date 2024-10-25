from .array_object import Array


def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Matrix product of two arrays.
    """
    from arkouda import matmul as ak_matmul

    from .array_object import Array

    return Array._new(ak_matmul(x1._array, x2._array))


def tensordot():
    """
    WARNING: not yet implemented
    """
    raise ValueError("tensordot not implemented")


def matrix_transpose(x: Array) -> Array:
    """
    Matrix product of two arrays.
    """
    from arkouda import transpose as ak_transpose

    from .array_object import Array

    return Array._new(ak_transpose(x._array))


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    from arkouda import vecdot as ak_vecdot

    from .array_object import Array

    return Array._new(ak_vecdot(x1._array, x2._array))
