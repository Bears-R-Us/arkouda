import builtins
import sys
from enum import Enum
from typing import Tuple, Union, cast

import numpy as np  # type: ignore
from typeguard import typechecked

__all__ = [
    "DTypes",
    "DTypeObjects",
    "dtype",
    "bool",
    "int64",
    "float64",
    "uint8",
    "uint64",
    "str_",
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
]

NUMBER_FORMAT_STRINGS = {
    "bool": "{}",
    "int64": "{:n}",
    "float64": "{:.17f}",
    "uint8": "{:n}",
    "np.float64": "f",
    "uint64": "{:n}",
}

dtype = np.dtype
bool = np.dtype(bool)
int64 = np.dtype(np.int64)
float64 = np.dtype(np.float64)
uint8 = np.dtype(np.uint8)
uint64 = np.dtype(np.uint64)
str_ = np.dtype(np.str_)
npstr = np.dtype(str)
intTypes = frozenset((int64, uint64, uint8))
bitType = uint64

# Union aliases used for static and runtime type checking
bool_scalars = Union[builtins.bool, np.bool_]
float_scalars = Union[float, np.float64]
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

    BOOL = "bool"
    FLOAT = "float"
    FLOAT64 = "float64"
    INT = "int"
    INT64 = "int64"
    STR = "str"
    UINT8 = "uint8"
    UINT64 = "uint64"

    def __str__(self) -> str:  # type: ignore
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter
        """
        return self.value

    def __repr__(self) -> str:  # type: ignore
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
)
ARKOUDA_SUPPORTED_FLOATS = (float, np.float64)
ARKOUDA_SUPPORTED_NUMBERS = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    float,
    np.float64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
ARKOUDA_SUPPORTED_DTYPES = frozenset(
    [member.value for _, member in DType.__members__.items()]
)

DTypes = frozenset([member.value for _, member in DType.__members__.items()])
DTypeObjects = frozenset([bool, float, float64, int, int64, str, str_, uint8, uint64])
NumericDTypes = frozenset(["bool", "float", "float64", "int", "int64", "uint64"])
SeriesDTypes = {
    "string": np.str_,
    "<class 'str'>": np.str_,
    "int64": np.int64,
    "uint64": np.uint64,
    "<class 'numpy.int64'>": np.int64,
    "float64": np.float64,
    "<class 'numpy.float64'>": np.float64,
    "bool": bool,
    "<class 'bool'>": bool,
    "datetime64[ns]": np.int64,
    "timedelta64[ns]": np.int64,
}


def isSupportedInt(num):
    return isinstance(num, ARKOUDA_SUPPORTED_INTS)


def isSupportedFloat(num):
    return isinstance(num, ARKOUDA_SUPPORTED_FLOATS)


def isSupportedNumber(num):
    return isinstance(num, ARKOUDA_SUPPORTED_NUMBERS)


def _as_dtype(dt) -> np.dtype:
    if not isinstance(dt, np.dtype):
        return np.dtype(dt)
    return dt


@typechecked
def check_np_dtype(dt: np.dtype) -> None:
    """
    Assert that numpy dtype dt is one of the dtypes supported
    by arkouda, otherwise raise TypeError.

    Raises
    ------
    TypeError
        Raised if the dtype is not in supported dtypes or if
        dt is not a np.dtype
    """

    if _as_dtype(dt).name not in DTypes:
        raise TypeError(f"Unsupported type: {dt}")


@typechecked
def translate_np_dtype(dt: np.dtype) -> Tuple[builtins.str, int]:
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
    dt = _as_dtype(dt)
    check_np_dtype(dt)
    trans = {"i": "int", "f": "float", "b": "bool", "u": "uint", "U": "str"}
    kind = trans[dt.kind]
    return kind, dt.itemsize


def resolve_scalar_dtype(val: object) -> str:  # type: ignore
    """
    Try to infer what dtype arkouda_server should treat val as.
    """
    # Python bool or np.bool
    if isinstance(val, builtins.bool) or (
        hasattr(val, "dtype") and cast(np.bool_, val).dtype.kind == "b"
    ):
        return "bool"
    # Python int or np.int* or np.uint*
    elif isinstance(val, int) or (
        hasattr(val, "dtype") and cast(np.uint, val).dtype.kind in "ui"
    ):
        if isinstance(val, np.uint64):
            return "uint64"
        else:
            return "int64"
    # Python float or np.float*
    elif isinstance(val, float) or (
        hasattr(val, "dtype") and cast(np.float_, val).dtype.kind == "f"
    ):
        return "float64"
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
