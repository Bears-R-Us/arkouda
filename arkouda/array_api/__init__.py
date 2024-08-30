import warnings

from .array_object import Array

from ._constants import e, inf, nan, pi

from .creation_functions import (
    asarray,
    arange,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)

from .data_type_functions import (
    astype,
    can_cast,
    finfo,
    iinfo,
    result_type,
)

from ._dtypes import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
    bool_,
)

from .elementwise_functions import (
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_left_shift,
    bitwise_invert,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expm1,
    floor,
    floor_divide,
    greater,
    greater_equal,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    remainder,
    round,
    sign,
    sin,
    sinh,
    square,
    sqrt,
    subtract,
    tan,
    tanh,
    trunc,
)

from .indexing_functions import take

# linalg is an extension in the array API spec, which is a sub-namespace. Only
# a subset of functions in it are imported into the top-level namespace.
from . import linalg

from .linalg import matmul, tensordot, matrix_transpose, vecdot

from .manipulation_functions import (
    broadcast_arrays,
    broadcast_to,
    concat,
    expand_dims,
    flip,
    moveaxis,
    permute_dims,
    repeat,
    reshape,
    roll,
    squeeze,
    stack,
    tile,
    unstack,
)

from .searching_functions import argmax, argmin, nonzero, where, searchsorted

from .set_functions import unique_all, unique_counts, unique_inverse, unique_values

from .sorting_functions import argsort, sort

from .statistical_functions import max, mean, min, prod, std, sum, var, cumulative_sum

from .utility_functions import all, any, clip, diff, pad

__array_api_version__ = "2022.12"

__all__ = ["Array"]

__all__ += ["__array_api_version__"]

__all__ += ["e", "inf", "nan", "pi"]

__all__ += [
    "asarray",
    "arange",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]

__all__ += [
    "astype",
    "can_cast",
    "finfo",
    "iinfo",
    "result_type",
]

__all__ += [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool_",
]

__all__ += [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]

__all__ += ["take"]

__all__ += ["linalg"]

__all__ += ["matmul", "tensordot", "matrix_transpose", "vecdot"]

__all__ += [
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "repeat",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
]

__all__ += ["argmax", "argmin", "nonzero", "where", "searchsorted"]

__all__ += ["unique_all", "unique_counts", "unique_inverse", "unique_values"]

__all__ += ["argsort", "sort"]

__all__ += ["max", "mean", "min", "prod", "std", "sum", "var", "cumulative_sum"]

__all__ += ["all", "any", "clip", "diff", "pad"]

warnings.warn("The arkouda.array_api submodule is still experimental.")
