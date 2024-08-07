from __future__ import annotations

import builtins
import sys
from enum import Enum
from typing import Tuple, Union, cast

import numpy as np
from typeguard import typechecked

__all__ = [
    "DTypes",
    "DTypeObjects",
    "ScalarDTypes",
    "dtype",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool_",
    "str_",
    "bigint",
    "intTypes",
    "bitType",
    "check_np_dtype",
    "translate_np_dtype",
    "resolve_scalar_dtype",
    "ARKOUDA_SUPPORTED_DTYPES",
    "bool_scalars",
    "float_scalars",
    "int_scalars",
    "numeric_scalars",
    "numpy_scalars",
    "str_scalars",
    "all_scalars",
    "get_byteorder",
    "get_server_byteorder",
    "isSupportedNumber",
]

NUMBER_FORMAT_STRINGS = {
    "bool": "{}",
    "int64": "{:n}",
    "float64": "{:.17f}",
    "uint8": "{:n}",
    "np.float64": "f",
    "uint64": "{:n}",
    "bigint": "{:n}",
}


def dtype(x):
    # we had to create our own bigint type since numpy
    # gives them dtype=object there's no np equivalent
    if (
        (isinstance(x, str) and x == "bigint")
        or isinstance(x, bigint)
        or (hasattr(x, "name") and x.name == "bigint")
    ):
        return bigint()
    if isinstance(x, str) and x in ["Strings"]:
        return np.dtype(np.str_)
    else:
        return np.dtype(x)


def _is_dtype_in_union(dtype, union_type) -> builtins.bool:
    """
    Check if a given type is in a typing.Union.

    Args
    ----
        dtype (type): The type to check for.
        union_type (type): The typing.Union type to check against.

    Returns
    -------
        bool: True if the dtype is in the union_type, False otherwise.
    """
    return hasattr(union_type, "__args__") and dtype in union_type.__args__


def _val_isinstance_of_union(val, union_type) -> builtins.bool:
    """
    Check if a given val is an instance of one of the types in the typing.Union

    Args
    ----
        val: The val to do the isinstance check on.
        union_type (type): The typing.Union type to check against.

    Returns
    -------
        bool: True if the val is an instance of one
            of the types in the union_type, False otherwise.
    """
    return hasattr(union_type, "__args__") and isinstance(val, union_type.__args__)


class bigint:
    # an estimate of the itemsize of bigint (128 bytes)
    itemsize = 128
    name = "bigint"
    ndim = 0
    shape = ()

    def __init__(self):
        self.kind = "ui"

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"dtype({self.name})"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(dtype(other), bigint):
            return True
        return False

    def __neq__(self, other):
        return not (self == other)

    def type(self, x):
        return int(x)


uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
float32 = np.float32
float64 = np.float64
complex64 = np.complex64
complex128 = np.complex128
bool_ = np.bool_
str_ = np.str_
intTypes = frozenset((dtype("int64"), dtype("uint64"), dtype("uint8")))
bitType = uint64

# Union aliases used for static and runtime type checking
bool_scalars = Union[builtins.bool, np.bool_]
float_scalars = Union[float, np.float64, np.float32]
int_scalars = Union[
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
numeric_scalars = Union[float_scalars, int_scalars]
numeric_and_bool_scalars = Union[bool_scalars, numeric_scalars]
numpy_scalars = Union[
    np.float64,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.bool_,
    np.str_,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
str_scalars = Union[str, np.str_]
all_scalars = Union[bool_scalars, numeric_scalars, numpy_scalars, str_scalars]

"""
The DType enum defines the supported Arkouda data types in string form.
"""


class DType(Enum):
    FLOAT = "float"
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    INT = "int"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT = "uint"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BOOL = "bool"
    BIGINT = "bigint"
    STR = "str"

    def __str__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter
        """
        return self.value

    def __repr__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter
        """
        return self.value


ARKOUDA_SUPPORTED_INTS = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    bigint,
)

ARKOUDA_SUPPORTED_FLOATS = (float, np.float64)
ARKOUDA_SUPPORTED_NUMBERS = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    bigint,
)

# TODO: bring supported data types into parity with all numpy dtypes
# missing full support for: float32, int32, int16, int8, uint32, uint16, complex64, complex128
# ARKOUDA_SUPPORTED_DTYPES = frozenset([member.value for _, member in DType.__members__.items()])
ARKOUDA_SUPPORTED_DTYPES = frozenset(
    ["bool_", "float", "float64", "int", "int64", "uint", "uint64", "uint8", "bigint", "str"]
)

DTypes = frozenset([member.value for _, member in DType.__members__.items()])
DTypeObjects = frozenset([bool_, float, float64, int, int64, str, str_, uint8, uint64])
NumericDTypes = frozenset(["bool_", "bool", "float", "float64", "int", "int64", "uint64", "bigint"])
SeriesDTypes = {
    "string": np.str_,
    "<class 'str'>": np.str_,
    "int64": np.int64,
    "uint64": np.uint64,
    "<class 'numpy.int64'>": np.int64,
    "float64": np.float64,
    "<class 'numpy.float64'>": np.float64,
    "bool": np.bool_,
    "<class 'bool'>": np.bool_,
    "datetime64[ns]": np.int64,
    "timedelta64[ns]": np.int64,
}
ScalarDTypes = frozenset(["bool_", "float64", "int64"])


def isSupportedInt(num):
    return isinstance(num, ARKOUDA_SUPPORTED_INTS)


def isSupportedFloat(num):
    return isinstance(num, ARKOUDA_SUPPORTED_FLOATS)


def isSupportedNumber(num):
    return isinstance(num, ARKOUDA_SUPPORTED_NUMBERS)


@typechecked
def check_np_dtype(dt: Union[np.dtype, "bigint"]) -> None:
    """
    Assert that numpy dtype dt is one of the dtypes supported
    by arkouda, otherwise raise TypeError.

    Raises
    ------
    TypeError
        Raised if the dtype is not in supported dtypes or if
        dt is not a np.dtype
    """

    if dtype(dt).name not in DTypes:
        raise TypeError(f"Unsupported type: {dt}")


@typechecked
def translate_np_dtype(dt) -> Tuple[builtins.str, int]:
    """
    Split numpy dtype dt into its kind and byte size, raising
    TypeError for unsupported dtypes.

    Raises
    ------
    TypeError
        Raised if the dtype is not in supported dtypes or if
        dt is not a np.dtype
    """
    # Assert that dt is one of the arkouda supported dtypes
    dt = dtype(dt)
    check_np_dtype(dt)
    trans = {"i": "int", "f": "float", "b": "bool", "u": "uint", "U": "str", "c": "complex"}
    kind = trans[dt.kind]
    return kind, dt.itemsize


def resolve_scalar_dtype(val: object) -> str:
    """
    Try to infer what dtype arkouda_server should treat val as.
    """

    # Python bool or np.bool
    if isinstance(val, builtins.bool) or (
        hasattr(val, "dtype") and cast(np.bool_, val).dtype.kind == "b"
    ):
        return "bool"
    # Python int or np.int* or np.uint*
    elif isinstance(val, int) or (hasattr(val, "dtype") and cast(np.uint, val).dtype.kind in "ui"):
        # we've established these are int, uint, or bigint,
        # so we can do comparisons
        if isSupportedInt(val) and val >= 2**64:  # type: ignore
            return "bigint"
        elif isinstance(val, np.uint64) or val >= 2**63:  # type: ignore
            return "uint64"
        else:
            return "int64"
    # Python float or np.float*
    elif isinstance(val, float) or (hasattr(val, "dtype") and cast(np.float_, val).dtype.kind == "f"):
        return "float64"
    elif isinstance(val, complex) or (hasattr(val, "dtype") and cast(np.float_, val).dtype.kind == "c"):
        return "float64"  # TODO: actually support complex values in the backend
    elif isinstance(val, builtins.str) or isinstance(val, np.str_):
        return "str"
    # Other numpy dtype
    elif hasattr(val, "dtype"):
        return cast(np.dtype, val).name
    # Other python type
    else:
        return builtins.str(type(val))


def get_byteorder(dt: np.dtype) -> str:
    """
    Get a concrete byteorder (turns '=' into '<' or '>')
    """
    if dt.byteorder == "=":
        if sys.byteorder == "little":
            return "<"
        elif sys.byteorder == "big":
            return ">"
        else:
            raise ValueError("Client byteorder must be 'little' or 'big'")
    else:
        return dt.byteorder


def get_server_byteorder() -> str:
    """
    Get the server's byteorder
    """
    from arkouda.client import get_config

    order = get_config()["byteorder"]
    if order not in ("little", "big"):
        raise ValueError("Server byteorder must be 'little' or 'big'")
    return cast("str", order)
