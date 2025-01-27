from __future__ import annotations

import builtins
import json
from functools import reduce
from math import ceil
from sys import modules
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.infoclass import information, pretty_print_information
from arkouda.logger import getArkoudaLogger
from arkouda.numpy.dtypes import (
    ARKOUDA_SUPPORTED_INTS,
    NUMBER_FORMAT_STRINGS,
    DTypes,
    bigint,
)
from arkouda.numpy.dtypes import bool_ as akbool
from arkouda.numpy.dtypes import bool_scalars, dtype
from arkouda.numpy.dtypes import float64 as akfloat64
from arkouda.numpy.dtypes import get_byteorder, get_server_byteorder
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import (
    int_scalars,
    isSupportedInt,
    isSupportedNumber,
    numeric_scalars,
    numpy_scalars,
    resolve_scalar_dtype,
)
from arkouda.numpy.dtypes import str_ as akstr_
from arkouda.numpy.dtypes import uint64 as akuint64

module = modules[__name__]

if TYPE_CHECKING:
    #   These are dummy functions that are used only for type checking.
    #   They communicate to mypy the expected input and output types,
    #   in cases where the function is generated at runtime.
    def numeric_reduce(
        pda: pdarray,
        axis: Optional[Union[int_scalars, Tuple[int_scalars, ...]]] = None,
        keepdims: bool = False,
    ) -> Union[numpy_scalars, pdarray]:
        pass

    sum = numeric_reduce
    prod = numeric_reduce
    max = numeric_reduce
    min = numeric_reduce

    def boolean_reduce(
        pda, axis: Optional[Union[int_scalars, Tuple[int_scalars, ...]]] = None, keepdims: bool = False
    ) -> Union[bool_scalars, pdarray]:
        pass

    is_sorted = boolean_reduce
    is_locally_sorted = boolean_reduce
    all = boolean_reduce
    any = boolean_reduce

    def index_reduce(
        pda: pdarray, axis: Optional[Union[int_scalars, None]] = None, keepdims: bool = False
    ) -> Union[akint64, akuint64, pdarray]:
        pass

    argmax = index_reduce
    argmin = index_reduce


__all__ = [
    "pdarray",
    "clear",
    "any",
    "all",
    "is_sorted",
    "sum",
    "dot",
    "prod",
    "min",
    "max",
    "argmin",
    "argmax",
    "mean",
    "var",
    "std",
    "mink",
    "maxk",
    "argmink",
    "argmaxk",
    "popcount",
    "parity",
    "clz",
    "ctz",
    "rotl",
    "rotr",
    "cov",
    "corr",
    "divmod",
    "sqrt",
    "power",
    "mod",
    "fmod",
    "attach_pdarray",
    "unregister_pdarray_by_name",
    "RegistrationError",
    "broadcast_to_shape",
    "_to_pdarray",
]
logger = getArkoudaLogger(name="pdarrayclass")

SUPPORTED_REDUCTION_OPS = ["any", "all", "isSorted", "isSortedLocally", "max", "min", "sum", "prod"]

SUPPORTED_INDEX_REDUCTION_OPS = ["argmin", "argmax"]


@typechecked
def parse_single_value(msg: str) -> Union[numpy_scalars, int]:
    """
    Attempt to convert a scalar return value from the arkouda server to a
    numpy scalar in Python. The user should not call this function directly.

    Parameters
    ----------
    msg : str
        scalar value in string form to be converted to a numpy scalar

    Returns
    -------
    object numpy scalar
    """

    def unescape(s):
        escaping = False
        res = ""
        for c in s:
            if escaping:
                res += c
                escaping = False
            elif c == "\\":
                escaping = True
            else:
                res += c
        return res

    dtname, value = msg.split(maxsplit=1)
    mydtype = dtype(dtname)
    if mydtype == bigint:
        # we have to strip off quotes prior to 1.32
        if value[0] == '"':
            return int(value[1:-1])
        else:
            return int(value)
    if mydtype == akbool:
        if value == "True" or value == "true":
            return mydtype.type(True)
        elif value == "False" or value == "false":
            return mydtype.type(False)
        else:
            raise ValueError(f"unsupported value from server {mydtype.name} {value}")
    try:
        if mydtype == akstr_:
            # String value will always be surrounded with double quotes, so remove them
            return mydtype.type(unescape(value[1:-1]))
        return mydtype.type(value)
    except Exception:
        raise ValueError(f"unsupported value from server {mydtype.name} {value}")


def _create_scalar_array(value):
    """
    Create a pdarray from a single scalar value
    """
    return create_pdarray(
        generic_msg(
            cmd=f"createScalarArray<{resolve_scalar_dtype(value)}>",
            args={
                "value": value,
            },
        )
    )


def _slice_index(array: pdarray, starts: List[int], stops: List[int], strides: List[int]):
    """
    Slice a pdarray with a set of start, stop and stride values
    """
    return create_pdarray(
        generic_msg(
            cmd=f"[slice]<{array.dtype},{array.ndim}>",
            args={
                "array": array,
                "starts": tuple(starts) if array.ndim > 1 else starts[0],
                "stops": tuple(stops) if array.ndim > 1 else stops[0],
                "strides": tuple(strides) if array.ndim > 1 else strides[0],
                "max_bits": array.max_bits if array.max_bits is not None else 0,
            },
        )
    )


def _parse_index_tuple(key, shape):
    """
    Parse a tuple of indices into slices, scalars, and pdarrays

    Returns a tuple of (starts, stops and strides) for the slices and scalar indices,
    as well as lists indicating which axes are indexed by scalars and pdarrays
    """
    scalar_axes = []
    pdarray_axes = []
    slices = []

    for dim, k in enumerate(key):
        if isinstance(k, slice):
            slices.append(k.indices(shape[dim]))
        elif np.isscalar(k) and (resolve_scalar_dtype(k) in ["int64", "uint64"]):
            scalar_axes.append(dim)

            if k < 0:
                # Interpret negative key as offset from end of array
                k += int(shape[dim])
            if k < 0 or k >= int(shape[dim]):
                raise IndexError(f"index {k} is out of bounds in dimension {dim} with size {shape[dim]}")
            else:
                # treat this as a single-element slice
                slices.append((k, k + 1, 1))
        elif isinstance(k, pdarray):
            pdarray_axes.append(dim)
            if k.dtype not in ("bool", "int", "uint"):
                raise TypeError(f"unsupported pdarray index type {k.dtype}")
            # select all indices (needed for mixed slice+pdarray indexing)
            slices.append((0, shape[dim], 1))
        else:
            raise IndexError(f"Unhandled key type: {k} ({type(k)})")

    return (tuple(zip(*slices)), scalar_axes, pdarray_axes)


def _parse_none_and_ellipsis_keys(key, ndim):
    """
    Parse a key tuple for None and Ellipsis values

    Return a tuple of the key with None values removed and the ellipsis replaced
    with the appropriate number of colons

    Also returns a tuple without the 'None' values removed
    """

    # create a copy to avoid modifying the original key
    ret_key = key

    # how many 'None' arguments are in the key tuple
    num_none = reduce(lambda x, y: x + (1 if y is None else 0), ret_key, 0)

    # replace '...' with the appropriate number of ':'
    elipsis_axis_idx = -1
    for dim, k in enumerate(ret_key):
        if isinstance(k, type(Ellipsis)):
            if elipsis_axis_idx != -1:
                raise IndexError("array index can only have one ellipsis")
            else:
                elipsis_axis_idx = dim

    if elipsis_axis_idx != -1:
        ret_key = tuple(
            ret_key[:elipsis_axis_idx]
            + (slice(None),) * (ndim - (len(ret_key) - num_none) + 1)
            + ret_key[(elipsis_axis_idx + 1) :]
        )

    key_with_none = ret_key

    if num_none > 0:
        # remove all 'None' indices
        ret_key = tuple([k for k in ret_key if k is not None])

    if len(ret_key) != ndim:
        raise IndexError(f"cannot index {ndim}D array with {len(ret_key)} indices")

    return (ret_key, num_none, key_with_none)


def _to_pdarray(value: np.ndarray, dt=None) -> pdarray:
    from arkouda.client import maxTransferBytes

    if dt is None:
        _dtype = dtype(value.dtype)
    else:
        _dtype = dtype(dt)

    if value.nbytes > maxTransferBytes:
        raise RuntimeError(
            f"Creating pdarray from ndarray would require transferring {value.nbytes} bytes from "
            + f"the client to server, which exceeds the maximum size of {maxTransferBytes} bytes. "
            + "Try increasing ak.maxTransferBytes"
        )

    if value.shape == ():
        return _create_scalar_array(value.item())
    else:
        value_flat = value.flatten()
        return create_pdarray(
            generic_msg(
                cmd=f"array<{_dtype},{value.ndim}>",
                args={"shape": np.shape(value)},
                payload=_array_memview(value_flat),
                send_binary=True,
            )
        )


def _array_memview(a) -> memoryview:
    if (get_byteorder(a.dtype) == "<" and get_server_byteorder() == "big") or (
        get_byteorder(a.dtype) == ">" and get_server_byteorder() == "little"
    ):
        return memoryview(a.byteswap())
    else:
        return memoryview(a)


# class for the pdarray
class pdarray:
    """
    The basic arkouda array class. This class contains only the
    attributies of the array; the data resides on the arkouda
    server. When a server operation results in a new array, arkouda
    will create a pdarray instance that points to the array data on
    the server. As such, the user should not initialize pdarray
    instances directly.

    Attributes
    ----------
    name : str
        The server-side identifier for the array
    dtype : dtype
        The element type of the array
    size : int_scalars
        The number of elements in the array
    ndim : int_scalars
        The rank of the array (currently only rank 1 arrays supported)
    shape : Sequence[int]
        A list or tuple containing the sizes of each dimension of the array
    itemsize : int_scalars
        The size in bytes of each element
    """

    BinOps = frozenset(
        [
            "+",
            "-",
            "*",
            "/",
            "//",
            "%",
            "<",
            ">",
            "<=",
            ">=",
            "!=",
            "==",
            "&",
            "|",
            "^",
            "<<",
            ">>",
            ">>>",
            "<<<",
            "**",
        ]
    )
    OpEqOps = frozenset(["+=", "-=", "*=", "/=", "%=", "//=", "&=", "|=", "^=", "<<=", ">>=", "**="])
    objType = "pdarray"

    __array_priority__ = 1000

    def __init__(
        self,
        name: str,
        mydtype: Union[np.dtype, str],
        size: int_scalars,
        ndim: int_scalars,
        shape: Sequence[int],
        itemsize: int_scalars,
        max_bits: Optional[int] = None,
    ) -> None:
        self.name = name
        self.dtype = dtype(mydtype)
        self.size = size
        self.ndim = ndim
        self._shape = tuple(shape)
        self.itemsize = itemsize
        if max_bits:
            self.max_bits = max_bits

        self.registered_name: Optional[str] = None

    def __del__(self):
        try:
            logger.debug(f"deleting pdarray with name {self.name}")
            generic_msg(cmd="delete", args={"name": self.name})
        except (RuntimeError, AttributeError):
            pass

    def __bool__(self) -> builtins.bool:
        if self.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous."
                "Use a.any() or a.all()"
            )
        return builtins.bool(self[0])

    def __len__(self):
        return self.size

    def __str__(self):
        from arkouda.client import pdarrayIterThresh

        return generic_msg(cmd="str", args={"array": self, "printThresh": pdarrayIterThresh})

    def __repr__(self):
        from arkouda.client import pdarrayIterThresh

        return generic_msg(cmd="repr", args={"array": self, "printThresh": pdarrayIterThresh})

    @property
    def shape(self):
        """
        Return the shape of an array.

        Returns
        -------
        tuple of int
            The elements of the shape tuple give the lengths of the corresponding array dimensions.
        """
        return tuple(self._shape)

    @property
    def max_bits(self):
        if self.dtype == bigint:
            if not hasattr(self, "_max_bits"):
                # if _max_bits hasn't been set, fetch value from server
                self._max_bits = generic_msg(cmd="get_max_bits", args={"array": self})
            return int(self._max_bits)
        return None

    @max_bits.setter
    def max_bits(self, max_bits):
        if self.dtype == bigint:
            generic_msg(cmd="set_max_bits", args={"array": self, "max_bits": max_bits})
            self._max_bits = max_bits

    def equals(self, other) -> bool_scalars:
        """
        Whether pdarrays are the same size and all entries are equal.

        Parameters
        ----------
        other : object
            object to compare.

        Returns
        -------
        bool
            True if the pdarrays are the same, o.w. False.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> a = ak.array([1, 2, 3])
        >>> a_cpy = ak.array([1, 2, 3])
        >>> a.equals(a_cpy)
        True
        >>> a2 = ak.array([1, 2, 5)
        >>> a.equals(a2)
        False
        """
        if isinstance(other, pdarray):
            if other.size != self.size:
                return False
            else:
                result = all(self == other)
                if isinstance(result, (bool, np.bool_)):
                    return result
        return False

    def format_other(self, other) -> str:
        """
        Attempt to cast scalar other to the element dtype of this pdarray,
        and print the resulting value to a string (e.g. for sending to a
        server command). The user should not call this function directly.

        Parameters
        ----------
        other : object
            The scalar to be cast to the pdarray.dtype

        Returns
        -------
        string representation of np.dtype corresponding to the other parameter

        Raises
        ------
        TypeError
            Raised if the other parameter cannot be converted to
            Numpy dtype

        """
        try:
            if self.dtype != bigint:
                other = np.array([other]).astype(self.dtype)[0]
            else:
                other = int(other)
        except Exception:
            raise TypeError(f"Unable to convert {other} to {self.dtype.name}")
        if self.dtype == "bool_":
            return str(other)
        fmt = NUMBER_FORMAT_STRINGS[self.dtype.name]
        return fmt.format(other)

    # binary operators
    def _binop(self, other: pdarray, op: str) -> pdarray:
        """
        Executes binary operation specified by the op string

        Parameters
        ----------
        other : pdarray
            The pdarray upon which the binop is to be executed
        op : str
            The binop to be executed

        Returns
        -------
        pdarray
            A pdarray encapsulating the binop result

        Raises
        ------
        ValueError
            Raised if the op is not within the pdarray.BinOps set, or if the
            pdarray sizes don't match
        TypeError
            Raised if other is not a pdarray or the pdarray.dtype is not
            a supported dtype

        """
        # For pdarray subclasses like ak.Datetime and ak.Timedelta, defer to child logic
        if type(other) is not pdarray and issubclass(type(other), pdarray):
            return NotImplemented
        if op not in self.BinOps:
            raise ValueError(f"bad operator {op}")
        # pdarray binop pdarray
        if isinstance(other, pdarray):
            try:
                x1, x2, tmp_x1, tmp_x2 = broadcast_if_needed(self, other)
            except ValueError:
                raise ValueError(f"shape mismatch {self.shape} {other.shape}")
            repMsg = generic_msg(
                cmd=f"binopvv<{self.dtype},{other.dtype},{x1.ndim}>", args={"op": op, "a": x1, "b": x2}
            )
            if tmp_x1:
                del x1
            if tmp_x2:
                del x2
            return create_pdarray(repMsg)
        # pdarray binop scalar
        # If scalar cannot be safely cast, server will infer the return dtype
        dt = resolve_scalar_dtype(other)
        if self.dtype != bigint and np.can_cast(other, self.dtype):
            # If scalar can be losslessly cast to array dtype,
            # do the cast so that return array will have same dtype
            dt = self.dtype.name
            other = self.dtype.type(other)
        if dt not in DTypes:
            raise TypeError(f"Unhandled scalar type: {other} ({type(other)})")
        repMsg = generic_msg(
            cmd=f"binopvs<{self.dtype},{dt},{self.ndim}>",
            args={"op": op, "a": self, "value": other},
        )
        return create_pdarray(repMsg)

    # reverse binary operators
    # pdarray binop pdarray: taken care of by binop function
    def _r_binop(self, other: pdarray, op: str) -> pdarray:
        """
        Executes reverse binary operation specified by the op string

        Parameters
        ----------
        other : pdarray
            The pdarray upon which the reverse binop is to be executed
        op : str
            The name of the reverse binop to be executed

        Returns
        -------
        pdarray
            A pdarray encapsulating the reverse binop result

        Raises
        ------
        ValueError
            Raised if the op is not within the pdarray.BinOps set
        TypeError
            Raised if other is not a pdarray or the pdarray.dtype is not
            a supported dtype
        """

        if op not in self.BinOps:
            raise ValueError(f"bad operator {op}")
        # pdarray binop scalar
        # If scalar cannot be safely cast, server will infer the return dtype
        dt = resolve_scalar_dtype(other)
        if self.dtype != bigint and np.can_cast(other, self.dtype):
            # If scalar can be losslessly cast to array dtype,
            # do the cast so that return array will have same dtype
            dt = self.dtype.name
            other = self.dtype.type(other)
        if dt not in DTypes:
            raise TypeError(f"Unhandled scalar type: {other} ({type(other)})")
        repMsg = generic_msg(
            cmd=f"binopsv<{self.dtype},{dt},{self.ndim}>",
            args={"op": op, "dtype": dt, "a": self, "value": other},
        )
        return create_pdarray(repMsg)

    def transfer(self, hostname: str, port: int_scalars):
        """
        Sends a pdarray to a different Arkouda server

        Parameters
        ----------
        hostname : str
            The hostname where the Arkouda server intended to
            receive the pdarray is running.
        port : int_scalars
            The port to send the array over. This needs to be an
            open port (i.e., not one that the Arkouda server is
            running on). This will open up `numLocales` ports,
            each of which in succession, so will use ports of the
            range {port..(port+numLocales)} (e.g., running an
            Arkouda server of 4 nodes, port 1234 is passed as
            `port`, Arkouda will use ports 1234, 1235, 1236,
            and 1237 to send the array data).
            This port much match the port passed to the call to
            `ak.receive_array()`.


        Returns
        -------
        A message indicating a complete transfer

        Raises
        ------
        ValueError
            Raised if the op is not within the pdarray.BinOps set
        TypeError
            Raised if other is not a pdarray or the pdarray.dtype is not
            a supported dtype
        """
        # hostname is the hostname to send to
        return generic_msg(
            cmd="sendArray",
            args={"arg1": self, "hostname": hostname, "port": port, "objType": "pdarray"},
        )

    # overload + for pdarray, other can be {pdarray, int, float}
    def __add__(self, other):
        return self._binop(other, "+")

    def __radd__(self, other):
        return self._r_binop(other, "+")

    # overload - for pdarray, other can be {pdarray, int, float}
    def __sub__(self, other):
        return self._binop(other, "-")

    def __rsub__(self, other):
        return self._r_binop(other, "-")

    # overload * for pdarray, other can be {pdarray, int, float}
    def __mul__(self, other):
        return self._binop(other, "*")

    def __rmul__(self, other):
        return self._r_binop(other, "*")

    # overload / for pdarray, other can be {pdarray, int, float}
    def __truediv__(self, other):
        return self._binop(other, "/")

    def __rtruediv__(self, other):
        return self._r_binop(other, "/")

    # overload // for pdarray, other can be {pdarray, int, float}
    def __floordiv__(self, other):
        return self._binop(other, "//")

    def __rfloordiv__(self, other):
        return self._r_binop(other, "//")

    def __mod__(self, other):
        return self._binop(other, "%")

    def __rmod__(self, other):
        return self._r_binop(other, "%")

    # overload << for pdarray, other can be {pdarray, int}
    def __lshift__(self, other):
        return self._binop(other, "<<")

    def __rlshift__(self, other):
        return self._r_binop(other, "<<")

    # overload >> for pdarray, other can be {pdarray, int}
    def __rshift__(self, other):
        return self._binop(other, ">>")

    def __rrshift__(self, other):
        return self._r_binop(other, ">>")

    # overload & for pdarray, other can be {pdarray, int}
    def __and__(self, other):
        return self._binop(other, "&")

    def __rand__(self, other):
        return self._r_binop(other, "&")

    # overload | for pdarray, other can be {pdarray, int}
    def __or__(self, other):
        return self._binop(other, "|")

    def __ror__(self, other):
        return self._r_binop(other, "|")

    # overload | for pdarray, other can be {pdarray, int}
    def __xor__(self, other):
        return self._binop(other, "^")

    def __rxor__(self, other):
        return self._r_binop(other, "^")

    def __pow__(self, other):
        return self._binop(other, "**")

    def __rpow__(self, other):
        return self._r_binop(other, "**")

    # overloaded comparison operators
    def __lt__(self, other):
        return self._binop(other, "<")

    def __gt__(self, other):
        return self._binop(other, ">")

    def __le__(self, other):
        return self._binop(other, "<=")

    def __ge__(self, other):
        return self._binop(other, ">=")

    def __eq__(self, other):  # type: ignore
        if other is None:
            return False
        elif (self.dtype == "bool_") and (isinstance(other, pdarray) and (other.dtype == "bool_")):
            return ~(self ^ other)
        else:
            return self._binop(other, "==")

    def __ne__(self, other):  # type: ignore
        if (self.dtype == "bool_") and (isinstance(other, pdarray) and (other.dtype == "bool_")):
            return self ^ other
        else:
            return self._binop(other, "!=")

    # overload unary- for pdarray implemented as pdarray*(-1)
    def __neg__(self):
        return self._binop(-1, "*")

    # overload unary~ for pdarray implemented as pdarray^(~0)
    def __invert__(self):
        if self.dtype == akint64:
            return self._binop(~0, "^")
        if self.dtype == akuint64:
            return self._binop(~np.uint(0), "^")
        if self.dtype == "bool_":
            return self._binop(True, "^")
        raise TypeError(f"Unhandled dtype: {self} ({self.dtype})")

    @property
    def inferred_type(self) -> Union[str, None]:
        """
        Return a string of the type inferred from the values.
        """
        from arkouda.numpy.dtypes import float_scalars, int_scalars
        from arkouda.util import _is_dtype_in_union

        if _is_dtype_in_union(self.dtype, int_scalars):
            return "integer"
        elif _is_dtype_in_union(self.dtype, float_scalars):
            return "floating"
        elif self.dtype == "<U":
            return "string"
        return None

    # op= operators
    def opeq(self, other, op):
        if op not in self.OpEqOps:
            raise ValueError(f"bad operator {op}")
        # pdarray op= pdarray
        if isinstance(other, pdarray):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch {self.shape} {other.shape}")
            generic_msg(
                cmd=f"opeqvv<{self.dtype},{other.dtype},{self.ndim}>",
                args={"op": op, "a": self, "b": other},
            )
            return self
        # pdarray binop scalar
        # opeq requires scalar to be cast as pdarray dtype
        try:
            if self.dtype != bigint:
                other = np.array([other]).astype(self.dtype)[0]
            else:
                other = self.dtype.type(other)
        except Exception:
            # Can't cast other as dtype of pdarray
            raise TypeError(f"Unhandled scalar type: {other} ({type(other)})")

        generic_msg(
            # TODO: does opeqvs really need to select over pairs of dtypes?
            cmd=f"opeqvs<{self.dtype},{self.dtype},{self.ndim}>",
            args={"op": op, "a": self, "value": self.format_other(other)},
        )
        return self

    # overload += pdarray, other can be {pdarray, int, float}
    def __iadd__(self, other):
        return self.opeq(other, "+=")

    # overload -= pdarray, other can be {pdarray, int, float}
    def __isub__(self, other):
        return self.opeq(other, "-=")

    # overload *= pdarray, other can be {pdarray, int, float}
    def __imul__(self, other):
        return self.opeq(other, "*=")

    # overload /= pdarray, other can be {pdarray, int, float}
    def __itruediv__(self, other):
        return self.opeq(other, "/=")

    # overload %= pdarray, other can be {pdarray, int, float}
    def __imod__(self, other):
        return self.opeq(other, "%=")

    # overload //= pdarray, other can be {pdarray, int, float}
    def __ifloordiv__(self, other):
        return self.opeq(other, "//=")

    # overload <<= pdarray, other can be {pdarray, int, float}
    def __ilshift__(self, other):
        return self.opeq(other, "<<=")

    # overload >>= pdarray, other can be {pdarray, int, float}
    def __irshift__(self, other):
        return self.opeq(other, ">>=")

    # overload &= pdarray, other can be {pdarray, int, float}
    def __iand__(self, other):
        return self.opeq(other, "&=")

    # overload |= pdarray, other can be {pdarray, int, float}
    def __ior__(self, other):
        return self.opeq(other, "|=")

    # overload ^= pdarray, other can be {pdarray, int, float}
    def __ixor__(self, other):
        return self.opeq(other, "^=")

    def __ipow__(self, other):
        return self.opeq(other, "**=")

    def __iter__(self):
        raise NotImplementedError(
            "pdarray does not support iteration. To force data transfer from server, use to_ndarray"
        )

    # overload a[] to treat like list
    def __getitem__(self, key):
        if self.ndim == 1 and np.isscalar(key) and (resolve_scalar_dtype(key) in ["int64", "uint64"]):
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if key >= 0 and key < self.size:
                repMsg = generic_msg(
                    cmd=f"[int]<{self.dtype},1>",
                    args={
                        "array": self,
                        "idx": key,
                    },
                )
                return parse_single_value(repMsg)
            else:
                raise IndexError(f"[int] {orig_key} is out of bounds with size {self.size}")

        if self.ndim == 1 and isinstance(key, slice):
            (start, stop, stride) = key.indices(self.size)
            repMsg = generic_msg(
                cmd=f"[slice]<{self.dtype},1>",
                args={
                    "array": self,
                    "starts": start,
                    "stops": stop,
                    "strides": stride,
                    "max_bits": self.max_bits if self.max_bits is not None else 0,
                },
            )
            return create_pdarray(repMsg)

        if isinstance(key, tuple):
            # handle None and Ellipsis in the key tuple
            (clean_key, num_none, key_with_none) = _parse_none_and_ellipsis_keys(key, self.ndim)

            # parse the tuple key into slices, scalars, and pdarrays
            ((starts, stops, strides), scalar_axes, pdarray_axes) = _parse_index_tuple(
                clean_key, self.shape
            )

            if len(scalar_axes) == len(clean_key):
                # all scalars: use simpler indexing (and return a scalar)
                repMsg = generic_msg(
                    cmd=f"[int]<{self.dtype},{self.ndim}>",
                    args={
                        "array": self,
                        "idx": clean_key,
                    },
                )
                ret_array = parse_single_value(repMsg)

            elif len(pdarray_axes) > 0:
                if len(pdarray_axes) == len(clean_key):
                    # all pdarray indices: skip slice indexing
                    temp1 = self

                    # will return a 1D array where all but the first
                    # dimensions are squeezed out
                    degen_axes = pdarray_axes[1:]
                else:
                    # mix of pdarray and slice indices: do slice indexing first
                    temp1 = _slice_index(self, starts, stops, strides)

                    # will return a reduced-rank array, where all but the first
                    # pdarray dimensions are squeezed out
                    degen_axes = pdarray_axes[1:] + scalar_axes

                # ensure all indexing arrays have the same dtype (either int64 or uint64)
                idx_dtype = clean_key[pdarray_axes[0]].dtype
                for dim in pdarray_axes:
                    if clean_key[dim].dtype != idx_dtype:
                        raise TypeError("all pdarray indices must have the same dtype")

                # apply pdarray indexing (returning an ndim array with degenerate dimensions
                # along all the indexed axes except the first one)
                temp2 = create_pdarray(
                    generic_msg(
                        cmd=f"[pdarray]<{self.dtype},{idx_dtype},{self.ndim}>",
                        args={
                            "array": temp1,
                            "nIdxArrays": len(pdarray_axes),
                            "idx": [clean_key[dim] for dim in pdarray_axes],
                            "idxDims": pdarray_axes,
                        },
                    )
                )
                from arkouda.numpy import squeeze

                # remove any degenerate dimensions
                ret_array = squeeze(temp2, tuple(degen_axes))

            else:
                # all slice or scalar indices: use slice indexing only
                maybe_degen_arr = _slice_index(self, starts, stops, strides)

                if len(scalar_axes) > 0:
                    from arkouda.numpy import squeeze

                    # reduce the array rank if there are any scalar indices
                    ret_array = squeeze(maybe_degen_arr, tuple(scalar_axes))
                else:
                    ret_array = maybe_degen_arr

            # expand the dimensions of the array if there were any 'None' values in the key
            if num_none > 0:
                # If scalar return value, put it into an array so it can be reshaped
                if len(scalar_axes) == len(clean_key):
                    ret_array = _create_scalar_array(ret_array)

                # use 'None' values in the original key to expand the dimensions
                shape = []
                rs = list(ret_array.shape)
                for k in key_with_none:
                    if k is None:
                        shape.append(1)
                    else:
                        if len(rs) > 0:
                            shape.append(rs.pop(0))

                return ret_array.reshape(shape)
            else:
                return ret_array

        if isinstance(key, pdarray) and self.ndim == 1:
            if key.dtype not in ("bool", "int", "uint"):
                raise TypeError(f"unsupported pdarray index type {key.dtype}")
            if key.dtype == "bool" and self.size != key.size:
                raise ValueError(f"size mismatch {self.size} {key.size}")
            repMsg = generic_msg(
                cmd="[pdarray]",
                args={
                    "array": self,
                    "idx": key,
                },
            )
            return create_pdarray(repMsg)

        if isinstance(key, slice):
            # handle the arr[:] case
            if key == slice(None):
                # TODO: implement a cloneMsg to make this more efficient
                return _slice_index(
                    self, [0 for _ in range(self.ndim)], self.shape, [1 for _ in range(self.ndim)]
                )
            else:
                # TODO: mimic numpy's behavior of applying the slice to only the first dimension?
                raise ValueError(f"Unhandled slice for multidimensional array: {key}")
        else:
            raise TypeError(f"Unhandled key type: {key} ({type(key)})")

    def __setitem__(self, key, value):
        # convert numpy array value to pdarray value
        if isinstance(value, np.ndarray):
            _value = _to_pdarray(value)
        else:
            _value = value

        if self.ndim == 1:
            if np.isscalar(key) and (resolve_scalar_dtype(key) in ["int64", "uint64"]):
                orig_key = key
                if key < 0:
                    # Interpret negative key as offset from end of array
                    key += self.size
                if key >= 0 and key < self.size:
                    generic_msg(
                        cmd=f"[int]=val<{self.dtype},1>",
                        args={
                            "array": self,
                            "idx": key,
                            "value": self.format_other(_value),
                            "max_bits": self.max_bits if self.max_bits is not None else 0,
                        },
                    )
                else:
                    raise IndexError(f"index {orig_key} is out of bounds with size {self.size}")
            elif isinstance(key, pdarray):
                if isinstance(_value, pdarray):
                    generic_msg(
                        cmd="[pdarray]=pdarray", args={"array": self, "idx": key, "value": _value}
                    )
                else:
                    generic_msg(
                        cmd="[pdarray]=val",
                        args={
                            "array": self,
                            "idx": key,
                            "dtype": self.dtype,
                            "value": self.format_other(_value),
                        },
                    )
            elif isinstance(key, slice):
                (start, stop, stride) = key.indices(self.size)
                logger.debug(f"start: {start} stop: {stop} stride: {stride}")
                if isinstance(_value, pdarray):
                    generic_msg(
                        cmd=f"[slice]=pdarray<{self.dtype},{_value.dtype},1>",
                        args={
                            "array": self,
                            "starts": start,
                            "stops": stop,
                            "strides": stride,
                            "value": _value,
                        },
                    )
                else:
                    generic_msg(
                        cmd=f"[slice]=val<{self.dtype},1>",
                        args={
                            "array": self,
                            "starts": start,
                            "stops": stop,
                            "strides": stride,
                            "value": self.format_other(_value),
                            "max_bits": self.max_bits if self.max_bits is not None else 0,
                        },
                    )
            else:
                raise TypeError(f"Unhandled key type: {key} ({type(key)})")
        else:
            if isinstance(key, tuple):
                # TODO: add support for an Ellipsis in the key tuple
                # (inserts ':' for any unspecified dimensions)
                all_scalar_keys = True
                starts = []
                stops = []
                strides = []
                for dim, k in enumerate(key):
                    if isinstance(k, slice):
                        all_scalar_keys = False
                        (start, stop, stride) = k.indices(self.shape[dim])
                        starts.append(start)
                        stops.append(stop)
                        strides.append(stride)
                    elif np.isscalar(k) and (resolve_scalar_dtype(k) in ["int64", "uint64"]):
                        if k < 0:
                            # Interpret negative key as offset from end of array
                            k += int(self.shape[dim])
                        if k < 0 or k >= int(self.shape[dim]):
                            raise IndexError(
                                f"index {k} is out of bounds in dimension"
                                + f"{dim} with size {self.shape[dim]}"
                            )
                        else:
                            # treat this as a single element slice
                            starts.append(k)
                            stops.append(k + 1)
                            strides.append(1)

                if isinstance(_value, pdarray):
                    if len(starts) == self.ndim:
                        # TODO: correctly handle non-unit strides when determining whether the
                        # value shape matches the slice shape
                        slice_shape = tuple([stops[i] - starts[i] for i in range(self.ndim)])

                        # check that the slice is within the bounds of the array
                        for i in range(self.ndim):
                            if slice_shape[i] > self.shape[i]:
                                raise ValueError(
                                    f"slice indices ({key}) out of bounds for array of "
                                    + f"shape {self.shape}"
                                )

                        if _value.ndim == len(slice_shape):
                            # check that the slice shape matches the value shape
                            for i in range(self.ndim):
                                if slice_shape[i] != _value.shape[i]:
                                    raise ValueError(
                                        f"slice shape ({slice_shape}) must match shape of value "
                                        + f"array ({value.shape})"
                                    )
                            _value_r = _value
                        elif _value.ndim < len(slice_shape):
                            # check that the value shape is compatible with the slice shape
                            iv = 0
                            for i in range(self.ndim):
                                if slice_shape[i] == 1:
                                    continue
                                elif slice_shape[i] == _value.shape[iv]:
                                    iv += 1
                                else:
                                    raise ValueError(
                                        f"slice shape ({slice_shape}) must be compatible with shape "
                                        + f"of value array ({value.shape})"
                                    )

                            # reshape to add singleton dimensions as needed
                            _value_r = _value.reshape(slice_shape)
                        else:
                            raise ValueError(
                                f"value array must not have more dimensions ({_value.ndim}) than the"
                                + f"slice ({len(slice_shape)})"
                            )
                    else:
                        raise ValueError(
                            f"slice rank ({len(starts)}) must match array rank ({self.ndim})"
                        )

                    generic_msg(
                        cmd=f"[slice]=pdarray<{self.dtype},{_value_r.dtype},{self.ndim}>",
                        args={
                            "array": self,
                            "starts": tuple(starts),
                            "stops": tuple(stops),
                            "strides": tuple(strides),
                            "value": _value_r,
                        },
                    )
                elif all_scalar_keys:
                    # use simpler indexing if we got a tuple of only scalars
                    generic_msg(
                        cmd=f"[int]=val<{self.dtype},{self.ndim}>",
                        args={
                            "array": self,
                            "idx": key,
                            "value": self.format_other(_value),
                            "max_bits": self.max_bits if self.max_bits is not None else 0,
                        },
                    )
                else:
                    generic_msg(
                        cmd=f"[slice]=val<{self.dtype},{self.ndim}>",
                        args={
                            "array": self,
                            "starts": tuple(starts),
                            "stops": tuple(stops),
                            "strides": tuple(strides),
                            "value": self.format_other(_value),
                            "max_bits": self.max_bits if self.max_bits is not None else 0,
                        },
                    )
            elif isinstance(key, slice):
                # handle the arr[:] = case
                if key == slice(None):
                    if isinstance(_value, pdarray):
                        generic_msg(
                            cmd=f"[slice]=pdarray<{self.dtype},{_value.dtype},{self.ndim}>",
                            args={
                                "array": self,
                                "starts": tuple([0 for _ in range(self.ndim)]),
                                "stops": tuple(self.shape),
                                "strides": tuple([1 for _ in range(self.ndim)]),
                                "value": _value,
                            },
                        )
                    else:
                        generic_msg(
                            cmd=f"[slice]=val<{self.dtype},{self.ndim}>",
                            args={
                                "array": self,
                                "starts": tuple([0 for _ in range(self.ndim)]),
                                "stops": tuple(self.shape),
                                "strides": tuple([1 for _ in range(self.ndim)]),
                                "value": self.format_other(_value),
                                "max_bits": self.max_bits if self.max_bits is not None else 0,
                            },
                        )
                else:
                    raise ValueError(f"Incompatable slice for multidimensional array: {key}")
            else:
                raise TypeError(f"Unhandled key type for ND arrays: {key} ({type(key)})")

    @property
    def nbytes(self):
        """
        The size of the pdarray in bytes.

        Returns
        -------
        int
            The size of the pdarray in bytes.

        """
        return self.size * self.dtype.itemsize

    @typechecked
    def fill(self, value: numeric_scalars) -> None:
        """
        Fill the array (in place) with a constant value.

        Parameters
        ----------
        value : numeric_scalars

        Raises
        -------
        TypeError
            Raised if value is not an int, int64, float, or float64
        """
        cmd = f"set<{self.dtype},{self.ndim}>"
        generic_msg(
            cmd=cmd, args={"array": self, "dtype": self.dtype.name, "val": self.format_other(value)}
        )

    def any(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Union[bool_scalars, pdarray]:
        """
        Return True iff any element of the array evaluates to True.
        """
        return any(self, axis=axis, keepdims=keepdims)

    def all(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Union[bool_scalars, pdarray]:
        """
        Return True iff all elements of the array evaluate to True.
        """
        return all(self, axis=axis, keepdims=keepdims)

    def is_registered(self) -> np.bool_:
        """
        Return True iff the object is contained in the registry

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RuntimeError
            Raised if there's a server-side error thrown
        Note
        -----
        This will return True if the object is registered itself or as a component
        of another object
        """
        from arkouda.util import is_registered

        if self.registered_name is None:
            return np.bool_(is_registered(self.name, as_component=True))
        else:
            return np.bool_(is_registered(self.registered_name))

    def _list_component_names(self) -> List[str]:
        """
        Internal Function that returns a list of all component names

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            List of all component names
        """
        return [self.name]

    def info(self) -> str:
        """
        Returns a JSON formatted string containing information about all components of self

        Parameters
        ----------
        None

        Returns
        -------
        str
            JSON string containing information about all components of self
        """
        return information(self._list_component_names())

    def pretty_print_info(self) -> None:
        """
        Prints information about all components of self in a human readable format

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pretty_print_information(self._list_component_names())

    def is_sorted(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> Union[bool_scalars, pdarray]:
        """
        Return True iff the array is monotonically non-decreasing.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Indicates if the array is monotonically non-decreasing

        Raises
        ------
        TypeError
            Raised if pda is not a pdarray instance
        RuntimeError
            Raised if there's a server-side error thrown
        """
        return is_sorted(self, axis=axis, keepdims=keepdims)  # noqa: F821

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Union[numpy_scalars, pdarray]:
        """
        Return the sum of all elements in the array.
        """
        #   Function is generated at runtime with _make_reduction_func.
        return sum(self, axis=axis, keepdims=keepdims)

    def prod(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Union[numpy_scalars, pdarray]:
        """
        Return the product of all elements in the array. Return value is
        always a np.float64 or np.int64.
        """
        #   Function is generated at runtime with _make_reduction_func.
        return prod(self, axis=axis, keepdims=keepdims)  # noqa: F821

    def min(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Union[numpy_scalars, pdarray]:
        """
        Return the minimum value of the array.
        """
        #   Function is generated at runtime with _make_reduction_func.
        return min(self, axis=axis, keepdims=keepdims)

    def max(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Union[numpy_scalars, pdarray]:
        """
        Return the maximum value of the array.
        """
        #   Function is generated at runtime with _make_reduction_func.
        return max(self, axis=axis, keepdims=keepdims)

    def argmin(
        self, axis: Optional[Union[int, None]] = None, keepdims: bool = False
    ) -> Union[np.int64, np.uint64, pdarray]:
        """
        Return the index of the first occurrence of the array min value
        """
        #   Function is generated at runtime with _make_index_reduction_func.
        return argmin(self, axis=axis, keepdims=keepdims)

    def argmax(
        self, axis: Optional[Union[int, None]] = None, keepdims: bool = False
    ) -> Union[np.int64, np.uint64, pdarray]:
        """
        Return the index of the first occurrence of the array max value.
        """
        #   Function is generated at runtime with _make_index_reduction_func.
        return argmax(self, axis=axis, keepdims=keepdims)

    def mean(self) -> np.float64:
        """
        Return the mean of the array.
        """
        return mean(self)

    def var(self, ddof: int_scalars = 0) -> np.float64:
        """
        Compute the variance. See ``arkouda.var`` for details.

        Parameters
        ----------
        ddof : int_scalars
            "Delta Degrees of Freedom" used in calculating var

        Returns
        -------
        np.float64
            The scalar variance of the array

        Raises
        ------
        TypeError
            Raised if pda is not a pdarray instance
        ValueError
            Raised if the ddof >= pdarray size
        RuntimeError
            Raised if there's a server-side error thrown

        """
        return var(self, ddof=ddof)

    def std(self, ddof: int_scalars = 0) -> np.float64:
        """
        Compute the standard deviation. See ``arkouda.std`` for details.

        Parameters
        ----------
        ddof : int_scalars
            "Delta Degrees of Freedom" used in calculating std

        Returns
        -------
        np.float64
            The scalar standard deviation of the array

        Raises
        ------
        TypeError
            Raised if pda is not a pdarray instance
        RuntimeError
            Raised if there's a server-side error thrown
        """
        return std(self, ddof=ddof)

    def cov(self, y: pdarray) -> np.float64:
        """
        Compute the covariance between self and y.

        Parameters
        ----------
        y : pdarray
            Other pdarray used to calculate covariance

        Returns
        -------
        np.float64
            The scalar covariance of the two arrays

        Raises
        ------
        TypeError
            Raised if y is not a pdarray instance
        RuntimeError
            Raised if there's a server-side error thrown
        """
        return cov(self, y)

    def corr(self, y: pdarray) -> np.float64:
        """
        Compute the correlation between self and y using pearson correlation coefficient.

        Parameters
        ----------
        y : pdarray
            Other pdarray used to calculate correlation

        Returns
        -------
        np.float64
            The scalar correlation of the two arrays

        Raises
        ------
        TypeError
            Raised if y is not a pdarray instance
        RuntimeError
            Raised if there's a server-side error thrown
        """
        return corr(self, y)

    def mink(self, k: int_scalars) -> pdarray:
        """
        Compute the minimum "k" values.

        Parameters
        ----------
        k : int_scalars
            The desired count of maximum values to be returned by the output.

        Returns
        -------
        pdarray, int
            The maximum `k` values from pda

        Raises
        ------
        TypeError
            Raised if pda is not a pdarray
        """
        return mink(self, k)

    @typechecked
    def maxk(self, k: int_scalars) -> pdarray:
        """
        Compute the maximum "k" values.

        Parameters
        ----------
        k : int_scalars
            The desired count of maximum values to be returned by the output.

        Returns
        -------
        pdarray, int
            The maximum `k` values from pda

        Raises
        ------
        TypeError
            Raised if pda is not a pdarray
        """
        return maxk(self, k)

    def argmink(self, k: int_scalars) -> pdarray:
        """
        Compute the minimum "k" values.

        Parameters
        ----------
        k : int_scalars
            The desired count of maximum values to be returned by the output.

        Returns
        -------
        pdarray, int
            Indices corresponding to the maximum `k` values from pda

        Raises
        ------
        TypeError
            Raised if pda is not a pdarray
        """
        return argmink(self, k)

    def argmaxk(self, k: int_scalars) -> pdarray:
        """
        Finds the indices corresponding to the maximum "k" values.

        Parameters
        ----------
        k : int_scalars
            The desired count of maximum values to be returned by the output.

        Returns
        -------
        pdarray, int
            Indices corresponding to the  maximum `k` values, sorted

        Raises
        ------
        TypeError
            Raised if pda is not a pdarray
        """
        return argmaxk(self, k)

    def popcount(self) -> pdarray:
        """
        Find the population (number of bits set) in each element. See `ak.popcount`.
        """
        return popcount(self)

    def parity(self) -> pdarray:
        """
        Find the parity (XOR of all bits) in each element. See `ak.parity`.
        """
        return parity(self)

    def clz(self) -> pdarray:
        """
        Count the number of leading zeros in each element. See `ak.clz`.
        """
        return clz(self)

    def ctz(self) -> pdarray:
        """
        Count the number of trailing zeros in each element. See `ak.ctz`.
        """
        return ctz(self)

    def rotl(self, other) -> pdarray:
        """
        Rotate bits left by <other>.
        """
        return rotl(self, other)

    def rotr(self, other) -> pdarray:
        """
        Rotate bits right by <other>.
        """
        return rotr(self, other)

    def value_counts(self):
        """
        Count the occurrences of the unique values of self.

        Returns
        -------
        unique_values : pdarray
            The unique values, sorted in ascending order

        counts : pdarray, int64
            The number of times the corresponding unique value occurs

        Examples
        --------
        >>> ak.array([2, 0, 2, 4, 0, 0]).value_counts()
        (array([0, 2, 4]), array([3, 2, 1]))
        """
        from arkouda.numpy import value_counts

        return value_counts(self)

    def astype(self, dtype) -> pdarray:
        """
        Cast values of pdarray to provided dtype

        Parameters
        __________
        dtype: np.dtype or str
            Dtype to cast to

        Returns
        _______
        ak.pdarray
            An arkouda pdarray with values converted to the specified data type

        Notes
        _____
        This is essentially shorthand for ak.cast(x, '<dtype>') where x is a pdarray.
        """
        from arkouda.numpy import cast as akcast

        return akcast(self, dtype)

    def slice_bits(self, low, high) -> pdarray:
        """
        Returns a pdarray containing only bits from low to high of self.

        This is zero indexed and inclusive on both ends, so slicing the bottom 64 bits is
        pda.slice_bits(0, 63)

        Parameters
        __________
        low: int
            The lowest bit included in the slice (inclusive)
            zero indexed, so the first bit is 0
        high: int
            The highest bit included in the slice (inclusive)

        Returns
        -------
        pdarray
            A new pdarray containing the bits of self from low to high

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        Examples
        --------
        >>> p = ak.array([2**65 + (2**64 - 1)])
        >>> bin(p[0])
        '0b101111111111111111111111111111111111111111111111111111111111111111'

        >>> bin(p.slice_bits(64, 65)[0])
        '0b10'
        """
        if low > high:
            raise ValueError("low must not exceed high")
        return (self >> low) % 2 ** (high - low + 1)

    @typechecked()
    def bigint_to_uint_arrays(self) -> List[pdarray]:
        """
        Creates a list of uint pdarrays from a bigint pdarray.
        The first item in return will be the highest 64 bits of the
        bigint pdarray and the last item will be the lowest 64 bits.

        Returns
        -------
        List[pdarrays]
            A list of uint pdarrays where:
            The first item in return will be the highest 64 bits of the
            bigint pdarray and the last item will be the lowest 64 bits.

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        pdarraycreation.bigint_from_uint_arrays

        Examples
        --------
        >>> a = ak.arange(2**64, 2**64 + 5)
        >>> a
        array(["18446744073709551616" "18446744073709551617" "18446744073709551618"
        "18446744073709551619" "18446744073709551620"])

        >>> a.bigint_to_uint_arrays()
        [array([1 1 1 1 1]), array([0 1 2 3 4])]
        """
        ret_list = json.loads(generic_msg(cmd="bigint_to_uint_list", args={"array": self}))
        return list(reversed([create_pdarray(a) for a in ret_list]))

    def reshape(self, *shape):
        """
        Gives a new shape to an array without changing its data.

        Parameters
        ----------
        shape : int, tuple of ints, or pdarray
            The new shape should be compatible with the original shape.

        Returns
        -------
        pdarray
            a pdarray with the same data, reshaped to the new shape
        """
        # allows the elements of the shape parameter to be passed in as separate arguments
        # For example, a.reshape(10, 11) is equivalent to a.reshape((10, 11))
        # the lenshape variable addresses an error that occurred when a single integer was
        # passed
        if len(shape) == 1:
            shape = shape[0]
            lenshape = 1
        if (not isinstance(shape, int)) and (not isinstance(shape, pdarray)):
            shape = [i for i in shape]
            lenshape = len(shape)
        return create_pdarray(
            generic_msg(
                cmd=f"reshape<{self.dtype},{self.ndim},{lenshape}>",
                args={
                    "name": self.name,
                    "shape": shape,
                },
            ),
        )

    def flatten(self):
        """
        Return a copy of the array collapsed into one dimension.

        Returns
        -------
        A copy of the input array, flattened to one dimension.
        """
        return create_pdarray(
            generic_msg(
                cmd=f"flatten<{self.dtype.name},{self.ndim}>",
                args={"a": self},
            )
        )

    def to_ndarray(self) -> np.ndarray:
        """
        Convert the array to a np.ndarray, transferring array data from the
        Arkouda server to client-side Python. Note: if the pdarray size exceeds
        client.maxTransferBytes, a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            A numpy ndarray with the same attributes and data as the pdarray

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown, if the pdarray size
            exceeds the built-in client.maxTransferBytes size limit, or if the bytes
            received does not match expected number of bytes
        Notes
        -----
        The number of bytes in the array cannot exceed ``client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting client.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        array()
        to_list()

        Examples
        --------
        >>> a = ak.arange(0, 5, 1)
        >>> a.to_ndarray()
        array([0, 1, 2, 3, 4])

        >>> type(a.to_ndarray())
        numpy.ndarray
        """
        from arkouda.client import maxTransferBytes

        dt = dtype(self.dtype)

        if dt == bigint:
            # convert uint pdarrays into object ndarrays and recombine
            arrs = [n.to_ndarray().astype("O") for n in self.bigint_to_uint_arrays()]
            return builtins.sum(n << (64 * (len(arrs) - i - 1)) for i, n in enumerate(arrs))

        # Total number of bytes in the array data
        arraybytes = self.size * self.dtype.itemsize
        # Guard against overflowing client memory
        if arraybytes > maxTransferBytes:
            raise RuntimeError(
                "Array exceeds allowed size for transfer. Increase client.maxTransferBytes to allow"
            )
        # The reply from the server will be binary data
        data = cast(
            memoryview,
            generic_msg(
                cmd=f"tondarray<{self.dtype},{self.ndim}>", args={"array": self}, recv_binary=True
            ),
        )
        # Make sure the received data has the expected length
        if len(data) != self.size * self.dtype.itemsize:
            raise RuntimeError(
                f"Expected {self.size * self.dtype.itemsize} bytes but received {len(data)}"
            )
        # The server sends us native-endian data so we need to account for
        # that. If the view is readonly, copy so the np array is mutable
        if get_server_byteorder() == "big":
            dt = dt.newbyteorder(">")
        else:
            dt = dt.newbyteorder("<")

        if data.readonly:
            x = np.frombuffer(data, dt).copy()
        else:
            x = np.frombuffer(data, dt)

        if self.ndim == 1:
            return x
        else:
            return x.reshape(self.shape)

    def to_list(self) -> List:
        """
        Convert the array to a list, transferring array data from the
        Arkouda server to client-side Python. Note: if the pdarray size exceeds
        client.maxTransferBytes, a RuntimeError is raised.

        Returns
        -------
        list
            A list with the same data as the pdarray

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown, if the pdarray size
            exceeds the built-in client.maxTransferBytes size limit, or if the bytes
            received does not match expected number of bytes
        Notes
        -----
        The number of bytes in the array cannot exceed ``client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting client.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        to_ndarray()

        Examples
        --------
        >>> a = ak.arange(0, 5, 1)
        >>> a.to_list()
        [0, 1, 2, 3, 4]

        >>> type(a.to_list())
        list
        """
        return self.to_ndarray().tolist()

    def to_cuda(self):
        """
        Convert the array to a Numba DeviceND array, transferring array data from the
        arkouda server to Python via ndarray. If the array exceeds a builtin size limit,
        a RuntimeError is raised.

        Returns
        -------
        numba.DeviceNDArray
            A Numba ndarray with the same attributes and data as the pdarray; on GPU

        Raises
        ------
        ImportError
            Raised if CUDA is not available
        ModuleNotFoundError
            Raised if Numba is either not installed or not enabled
        RuntimeError
            Raised if there is a server-side error thrown in the course of retrieving
            the pdarray.

        Notes
        -----
        The number of bytes in the array cannot exceed ``client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting client.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        array

        Examples
        --------
        >>> a = ak.arange(0, 5, 1)
        >>> a.to_cuda()
        array([0, 1, 2, 3, 4])

        >>> type(a.to_cuda())
        numpy.devicendarray
        """
        try:
            from numba import cuda  # type: ignore

            if not (cuda.is_available()):
                raise ImportError(
                    "CUDA is not available. Check for the CUDA toolkit and ensure a GPU is installed."
                )
        except (ModuleNotFoundError, ImportError):
            raise ModuleNotFoundError(
                "Numba is not enabled or installed and is required for GPU support."
            )

        # Return a numba devicendarray
        return cuda.to_device(self.to_ndarray())

    @typechecked
    def to_parquet(
        self,
        prefix_path: str,
        dataset: str = "array",
        mode: str = "truncate",
        compression: Optional[str] = None,
    ) -> str:
        """
        Save the pdarray to Parquet. The result is a collection of files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.
        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        compression : str (Optional)
            (None | "snappy" | "gzip" | "brotli" | "zstd" | "lz4")
            Sets the compression type used with Parquet files
        Returns
        -------
        string message indicating result of save operation
        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        Notes
        -----
        - The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        - Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales`` for `file_type='distribute'`.
        - 'append' write mode is supported, but is not efficient.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        - Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        Examples
        --------
        >>> a = ak.arange(25)
        >>> # Saving without an extension
        >>> a.to_parquet('path/prefix', dataset='array')
        Saves the array to numLocales HDF5 files with the name ``cwd/path/name_prefix_LOCALE####``
        >>> # Saving with an extension (HDF5)
        >>> a.to_parqet('path/prefix.parquet', dataset='array')
        Saves the array to numLocales HDF5 files with the name
        ``cwd/path/name_prefix_LOCALE####.parquet`` where #### is replaced by each locale number
        """
        from arkouda.io import _mode_str_to_int

        return cast(
            str,
            generic_msg(
                cmd="writeParquet",
                args={
                    "values": self,
                    "dset": dataset,
                    "mode": _mode_str_to_int(mode),
                    "prefix": prefix_path,
                    "objType": "pdarray",
                    "dtype": self.dtype,
                    "compression": compression,
                },
            ),
        )

    @typechecked
    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "array",
        mode: str = "truncate",
        file_type: str = "distribute",
    ) -> str:
        """
        Save the pdarray to HDF5.
        The object can be saved to a collection of files or single file.
        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        file_type: str ("single" | "distribute")
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.
        Returns
        -------
        string message indicating result of save operation
        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        Notes
        -----
        - The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        - Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales`` for `file_type='distribute'`. Otherwise,
        the file name will be `prefix_path`.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        - Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        Examples
        --------
        >>> a = ak.arange(25)
        >>> # Saving without an extension
        >>> a.to_hdf('path/prefix', dataset='array')
        Saves the array to numLocales HDF5 files with the name ``cwd/path/name_prefix_LOCALE####``
        >>> # Saving with an extension (HDF5)
        >>> a.to_hdf('path/prefix.h5', dataset='array')
        Saves the array to numLocales HDF5 files with the name
        ``cwd/path/name_prefix_LOCALE####.h5`` where #### is replaced by each locale number
        >>> # Saving to a single file
        >>> a.to_hdf('path/prefix.hdf5', dataset='array', file_type='single')
        Saves the array in to single hdf5 file on the root node.
        ``cwd/path/name_prefix.hdf5``
        """
        from arkouda.io import _file_type_to_int, _mode_str_to_int

        return cast(
            str,
            generic_msg(
                cmd="tohdf",
                args={
                    "values": self,
                    "dset": dataset,
                    "write_mode": _mode_str_to_int(mode),
                    "filename": prefix_path,
                    "dtype": self.dtype,
                    "objType": self.objType,
                    "file_format": _file_type_to_int(file_type),
                },
            ),
        )

    def update_hdf(self, prefix_path: str, dataset: str = "array", repack: bool = True):
        """
        Overwrite the dataset with the name provided with this pdarray. If
        the dataset does not exist it is added

        Parameters
        -----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files
        repack: bool
            Default: True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Returns
        --------
        str - success message if successful

        Raises
        -------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray

        Notes
        ------
        - If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        - If the dataset provided does not exist, it will be added
        """
        from arkouda.io import (
            _file_type_to_int,
            _get_hdf_filetype,
            _mode_str_to_int,
            _repack_hdf,
        )

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        generic_msg(
            cmd="tohdf",
            args={
                "values": self,
                "dset": dataset,
                "write_mode": _mode_str_to_int("append"),
                "filename": prefix_path,
                "dtype": self.dtype,
                "objType": "pdarray",
                "file_format": _file_type_to_int(file_type),
                "overwrite": True,
            },
        )

        if repack:
            _repack_hdf(prefix_path)

    @typechecked
    def to_csv(
        self,
        prefix_path: str,
        dataset: str = "array",
        col_delim: str = ",",
        overwrite: bool = False,
    ):
        """
        Write pdarray to CSV file(s). File will contain a single column with the pdarray data.
        All CSV Files written by Arkouda include a header denoting data types of the columns.

        Parameters
        -----------
        prefix_path: str
            The filename prefix to be used for saving files. Files will have _LOCALE#### appended
            when they are written to disk.
        dataset: str
            Column name to save the pdarray under. Defaults to "array".
        col_delim: str
            Defaults to ",". Value to be used to separate columns within the file.
            Please be sure that the value used DOES NOT appear in your dataset.
        overwrite: bool
            Defaults to False. If True, any existing files matching your provided prefix_path will
            be overwritten. If False, an error will be returned if existing files are found.

        Returns
        --------
        str reponse message

        Raises
        ------
        ValueError
            Raised if all datasets are not present in all parquet files or if one or
            more of the specified files do not exist
        RuntimeError
            Raised if one or more of the specified files cannot be opened.
            If `allow_errors` is true this may be raised if no values are returned
            from the server.
        TypeError
            Raised if we receive an unknown arkouda_type returned from the server

        Notes
        ------
        - CSV format is not currently supported by load/load_all operations
        - The column delimiter is expected to be the same for column names and data
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline (`\n`) at this time.
        """
        return cast(
            str,
            generic_msg(
                cmd="writecsv",
                args={
                    "datasets": [self],
                    "col_names": [dataset],
                    "filename": prefix_path,
                    "num_dsets": 1,
                    "col_delim": col_delim,
                    "dtypes": [self.dtype.name],
                    "row_count": self.size,
                    "overwrite": overwrite,
                },
            ),
        )

    def save(
        self,
        prefix_path: str,
        dataset: str = "array",
        mode: str = "truncate",
        compression: Optional[str] = None,
        file_format: str = "HDF5",
        file_type: str = "distribute",
    ) -> str:
        """
        DEPRECATED
        Save the pdarray to HDF5 or Parquet. The result is a collection of files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. HDF5 support single files, in which case the file name will
        only be that provided. Each locale saves its chunk of the array to its
        corresponding file.
        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        compression : str (Optional)
            (None | "snappy" | "gzip" | "brotli" | "zstd" | "lz4")
            Sets the compression type used with Parquet files
        file_format : str {'HDF5', 'Parquet'}
            By default, saved files will be written to the HDF5 file format. If
            'Parquet', the files will be written to the Parquet file format. This
            is case insensitive.
        file_type: str ("single" | "distribute")
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.
        Returns
        -------
        string message indicating result of save operation
        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray
        ValueError
            Raised if there is an error in parsing the prefix path pointing to
            file write location or if the mode parameter is neither truncate
            nor append
        TypeError
            Raised if any one of the prefix_path, dataset, or mode parameters
            is not a string
        See Also
        --------
        save_all, load, read, to_parquet, to_hdf
        Notes
        -----
        The prefix_path must be visible to the arkouda server and the user must
        have write permission.
        Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales``. If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.
        Previously all files saved in Parquet format were saved with a ``.parquet`` file extension.
        This will require you to use load as if you saved the file with the extension. Try this if
        an older file is not being found.
        Any file extension can be used.The file I/O does not rely on the extension to
        determine the file format.
        Examples
        --------
        >>> a = ak.arange(25)
        >>> # Saving without an extension
        >>> a.save('path/prefix', dataset='array')
        Saves the array to numLocales HDF5 files with the name ``cwd/path/name_prefix_LOCALE####``
        >>> # Saving with an extension (HDF5)
        >>> a.save('path/prefix.h5', dataset='array')
        Saves the array to numLocales HDF5 files with the name
        ``cwd/path/name_prefix_LOCALE####.h5`` where #### is replaced by each locale number
        >>> # Saving with an extension (Parquet)
        >>> a.save('path/prefix.parquet', dataset='array', file_format='Parquet')
        Saves the array in numLocales Parquet files with the name
        ``cwd/path/name_prefix_LOCALE####.parquet`` where #### is replaced by each locale number
        """
        from warnings import warn

        warn(
            "ak.pdarray.save has been deprecated. Please use ak.pdarray.to_parquet or ak.pdarray.to_hdf",
            DeprecationWarning,
        )
        if mode.lower() not in ["append", "truncate"]:
            raise ValueError("Allowed modes are 'truncate' and 'append'")

        if file_format.lower() == "hdf5":
            return self.to_hdf(prefix_path, dataset=dataset, mode=mode, file_type=file_type)
        elif file_format.lower() == "parquet":
            return self.to_parquet(prefix_path, dataset=dataset, mode=mode, compression=compression)
        else:
            raise ValueError("Valid file types are HDF5 or Parquet")

    @typechecked
    def register(self, user_defined_name: str) -> pdarray:
        """
        Register this pdarray with a user defined name in the arkouda server
        so it can be attached to later using pdarray.attach()
        This is an in-place operation, registering a pdarray more than once will
        update the name in the registry and remove the previously registered name.
        A name can only be registered to one pdarray at a time.

        Parameters
        ----------
        user_defined_name : str
            user defined name array is to be registered under

        Returns
        -------
        pdarray
            The same pdarray which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support a
            fluid programming style.
            Please note you cannot register two different pdarrays with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the pdarray with the user_defined_name
            If the user is attempting to register more than one pdarray with the same name,
            the former should be unregistered first to free up the registration name.

        See also
        --------
        attach, unregister, is_registered, list_registry, unregister_pdarray_by_name

        Notes
        -----
        Registered names/pdarrays in the server are immune to deletion
        until they are unregistered.

        Examples
        --------
        >>> a = zeros(100)
        >>> a.register("my_zeros")
        >>> # potentially disconnect from server and reconnect to server
        >>> b = ak.pdarray.attach("my_zeros")
        >>> # ...other work...
        >>> b.unregister()
        """
        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")
        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "array": self.name,
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self) -> None:
        """
        Unregister a pdarray in the arkouda server which was previously
        registered using register() and/or attahced to using attach()

        Parameters
        ----------

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Raised if the server could not find the internal name/symbol to remove

        See also
        --------
        register, unregister, is_registered, unregister_pdarray_by_name, list_registry

        Notes
        -----
        Registered names/pdarrays in the server are immune to deletion until
        they are unregistered.

        Examples
        --------
        >>> a = zeros(100)
        >>> a.register("my_zeros")
        >>> # potentially disconnect from server and reconnect to server
        >>> b = ak.pdarray.attach("my_zeros")
        >>> # ...other work...
        >>> b.unregister()
        """
        from arkouda.util import unregister

        if self.registered_name is None:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    # class method self is not passed in
    # invoke with ak.pdarray.attach('user_defined_name')
    @staticmethod
    @typechecked
    def attach(user_defined_name: str) -> pdarray:
        """
        class method to return a pdarray attached to the registered name in the arkouda
        server which was registered using register()

        Parameters
        ----------
        user_defined_name : str
            user defined name which array was registered under

        Returns
        -------
        pdarray
            pdarray which is bound to the corresponding server side component which was registered
            with user_defined_name

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str

        See also
        --------
        register, unregister, is_registered, unregister_pdarray_by_name, list_registry

        Notes
        -----
        Registered names/pdarrays in the server are immune to deletion
        until they are unregistered.

        Examples
        --------
        >>> a = zeros(100)
        >>> a.register("my_zeros")
        >>> # potentially disconnect from server and reconnect to server
        >>> b = ak.pdarray.attach("my_zeros")
        >>> # ...other work...
        >>> b.unregister()
        """
        import warnings

        from arkouda.util import attach

        warnings.warn(
            "ak.pdarray.attach() is deprecated. Please use ak.attach() instead.",
            DeprecationWarning,
        )
        return attach(user_defined_name)

    def _float_to_uint(self):
        return generic_msg(cmd="transmuteFloat", args={"name": self})

    def _get_grouping_keys(self) -> List[pdarray]:
        """
        Private method for generating grouping keys used by GroupBy.

        API: this method must be defined by all groupable arrays, and it
        must return a list of arrays that can be (co)argsorted.
        """
        if self.dtype == akbool:
            from arkouda.numpy import cast as akcast

            return [akcast(self, akint64)]
        elif self.dtype in (akint64, akuint64):
            # Integral pdarrays are their own grouping keys
            return [self]
        elif self.dtype == akfloat64:
            return [create_pdarray(self._float_to_uint())]
        elif self.dtype == bigint:
            return self.bigint_to_uint_arrays()
        else:
            raise TypeError("Grouping is only supported on numeric data (integral types) and bools.")


# end pdarray class def


# creates pdarray object
#   only after:
#       all values have been checked by python module and...
#       server has created pdarray already before this is called
@typechecked
def create_pdarray(repMsg: str, max_bits=None) -> pdarray:
    """
    Return a pdarray instance pointing to an array created by the arkouda server.
    The user should not call this function directly.

    Parameters
    ----------
    repMsg : str
        space-delimited string containing the pdarray name, datatype, size
        dimension, shape,and itemsize

    Returns
    -------
    pdarray
        A pdarray with the same attributes and data as the pdarray; on GPU

    Raises
    -----
    ValueError
        If there's an error in parsing the repMsg parameter into the six
        values needed to create the pdarray instance
    RuntimeError
        Raised if a server-side error is thrown in the process of creating
        the pdarray instance
    """
    try:
        fields = repMsg.split()
        name = fields[1]
        mydtype = fields[2]
        size = int(fields[3])
        ndim = int(fields[4])

        if fields[5] == "[]":
            shape = []
        else:
            trailing_comma_offset = -2 if fields[5][len(fields[5]) - 2] == "," else -1
            shape = [int(el) for el in fields[5][1:trailing_comma_offset].split(",")]

        itemsize = int(fields[6])
    except Exception as e:
        raise ValueError(e)
    logger.debug(
        f"created Chapel array with name: {name} dtype: {mydtype} size: {size} ndim: {ndim} "
        + f"shape: {shape} itemsize: {itemsize}"
    )
    return pdarray(name, dtype(mydtype), size, ndim, shape, itemsize, max_bits)


@typechecked
def create_pdarrays(repMsg: str) -> List[pdarray]:
    """
    Return a list of pdarray instances pointing to arrays created by the
    arkouda server.

    Parameters
    ----------
    repMsg : str
        A JSON list of space delimited strings, each containing the pdarray
        name, datatype, size,

    Returns
    -------
    List[pdarray]
        A list of pdarrays with the same attributes and data as the pdarrays

    Raises
    ------
    ValueError
        If there's an error in parsing the repMsg parameter into the six
        values needed to create the pdarray instance
    RuntimeError
        Raised if a server-side error is thrown in the process of creating
        the pdarray instance
    """

    # TODO: maybe add more robust json parsing here
    try:
        repMsg = repMsg.strip("[]")
        responses = [r.strip().strip('"') for r in repMsg.split('",')]
        return [create_pdarray(response) for response in responses]
    except Exception as e:
        raise ValueError(e)


def clear() -> None:
    """
    Send a clear message to clear all unregistered data from the server symbol table

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in executing clear request
    """
    generic_msg(cmd="clear")


def _make_reduction_func(
    op,
    function_descriptor="Return reduction of a pdarray by an operation along an axis.",
    return_descriptor="",
    return_dtype="numpy_scalars",
):
    if op not in SUPPORTED_REDUCTION_OPS:
        raise ValueError(f"value {op} not supported by _make_reduction_func.")

    @typechecked
    def op_func(
        pda: pdarray,
        axis: Optional[Union[int_scalars, Tuple[int_scalars, ...]]] = None,
        keepdims: bool = False,
    ) -> Union[numpy_scalars, pdarray]:
        return _common_reduction(op, pda, axis, keepdims=keepdims)

    op_func.__doc__ = f"""
    {function_descriptor}

    Parameters
    ----------
    pda : pdarray
        The pdarray instance to be evaluated.
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the sum. If None, the reduction of the entire array is
        computed (returning a scalar).
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.

    Returns
    -------
    pdarray or {return_dtype}
        {return_descriptor}

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
     """

    return op_func


def _make_index_reduction_func(
    op,
    function_descriptor="Return index reduction of a pdarray by an operation along an axis.",
    return_descriptor="",
    return_dtype="int64, uint64",
):
    if op not in SUPPORTED_INDEX_REDUCTION_OPS:
        raise ValueError(f"value {op} not supported by _make_index_reduction_func.")

    @typechecked
    def op_func(
        pda: pdarray,
        axis: Optional[Union[int_scalars, None]] = None,
        keepdims: bool = False,
    ) -> Union[akuint64, akint64, pdarray]:
        return _common_index_reduction(op, pda, axis, keepdims=keepdims)

    op_func.__doc__ = f"""
    {function_descriptor}

    Parameters
    ----------
    pda : pdarray
        The pdarray instance to be evaluated.
    axis : int, optional
        The axis along which to compute the index reduction.
        If None, the reduction of the entire array is
        computed (returning a scalar).
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.

    Returns
    -------
    pdarray or {return_dtype}
        {return_descriptor}

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance.
        Raised axis is not an int.
    RuntimeError
        Raised if there's a server-side error thrown.
     """

    return op_func


# check whether a reduction of the given axes on an 'ndim' dimensional array
# would result in a single scalar value
def _reduces_to_single_value(axis, ndim) -> bool:
    if len(axis) == 0 or ndim == 1:
        # if no axes are specified or the array is 1D, the result is a scalar
        return True
    elif len(axis) == ndim:
        # if all axes are specified, the result is a scalar
        axes = set(axis)
        for i in range(ndim):
            if i not in axes:
                return False
        return True
    else:
        return False


# helper function for sum, min, max, prod
@typechecked
def _common_reduction(
    kind: str,
    pda: pdarray,
    axis: Optional[Union[int_scalars, Tuple[int_scalars, ...], None]] = None,
    keepdims: bool = False,
) -> Union[numpy_scalars, pdarray]:
    """
    Return reduction of a pdarray by an operation along an axis.

    Parameters
    ----------
    kind : str
        The name of the reduction operation.  Must be a member of SUPPORTED_REDUCTION_OPS.
    pda : pdarray
        The pdarray instance to be evaluated.
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the reduction. If None, the sum of the entire array is
        computed (returning a scalar).
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.
    Returns
    -------
    numpy_scalars, pdarray

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance.
    RuntimeError
        Raised if there's a server-side error thrown.
    ValueError
        Raised op is not a supported reduction operation.
    """

    if kind not in SUPPORTED_REDUCTION_OPS:
        raise ValueError(f"Unsupported reduction type: {kind}")

    axis_ = (
        []
        if axis is None
        else (
            [
                axis,
            ]
            if isinstance(axis, ARKOUDA_SUPPORTED_INTS)
            else list(axis)
        )
    )

    if _reduces_to_single_value(axis_, pda.ndim):
        return parse_single_value(
            cast(
                str,
                generic_msg(
                    cmd=f"{kind}All<{pda.dtype.name},{pda.ndim}>",
                    args={"x": pda, "skipNan": False},
                ),
            )
        )
    else:
        result = create_pdarray(
            generic_msg(
                cmd=f"{kind}<{pda.dtype.name},{pda.ndim}>",
                args={"x": pda, "axis": axis_, "skipNan": False},
            )
        )
        if keepdims or axis is None or pda.ndim == 1:
            return result
        else:
            from arkouda.numpy import squeeze

            return squeeze(result, axis)


# helper function for argmin, argmax
@typechecked
def _common_index_reduction(
    kind: str,
    pda: pdarray,
    axis: Optional[Union[int_scalars, Tuple[int_scalars, ...], None]] = None,
    keepdims: bool = False,
) -> Union[akuint64, akint64, pdarray]:
    """
    Return reduction of a pdarray by an operation along an axis.

    Parameters
    ----------
    kind : str
        The name of the reduction operation.  Must be a member of SUPPORTED_INDEX_REDUCTION_OPS.
    pda : pdarray
        The pdarray instance to be evaluated.
    axis : int or Tuple[int, ...], optional
        The axis or axes along which to compute the reduction. If None, the sum of the entire array is
        computed (returning a scalar).
    keepdims : bool, optional
        Whether to keep the singleton dimension(s) along `axis` in the result.
    Returns
    -------
    int64

    Raises
    ------
    TypeError
        Raised if axis is not of type int.
    """
    if kind not in SUPPORTED_INDEX_REDUCTION_OPS:
        raise ValueError(f"Unsupported reduction type: {kind}")

    if pda.ndim == 1 or axis is None:
        return parse_single_value(
            generic_msg(
                cmd=f"{kind}All<{pda.dtype.name},{pda.ndim}>",
                args={"x": pda},
            )
        )
    elif isinstance(axis, int):
        result = create_pdarray(
            generic_msg(
                cmd=f"{kind}<{pda.dtype.name},{pda.ndim}>",
                args={"x": pda, "axis": axis},
            )
        )
        if keepdims is False:
            from arkouda.numpy import squeeze

            return squeeze(result, axis)
        else:
            return result
    else:
        raise TypeError("axis must by of type int.")


setattr(
    module,
    "any",
    _make_reduction_func(
        "any",
        function_descriptor="Return True iff any element of the array evaluates to True.",
        return_descriptor="Indicates if any pdarray element evaluates to True.",
        return_dtype="bool",
    ),
)

setattr(
    module,
    "all",
    _make_reduction_func(
        "all",
        function_descriptor="Return True iff all elements of the array evaluate to True.",
        return_descriptor="Indicates if all pdarray elements evaluate to True.",
        return_dtype="bool",
    ),
)

setattr(
    module,
    "is_sorted",
    _make_reduction_func(
        "isSorted",
        function_descriptor="Return True iff the array is monotonically non-decreasing.",
        return_descriptor="Indicates if the array is monotonically non-decreasing.",
        return_dtype="bool",
    ),
)

setattr(
    module,
    "is_locally_sorted",
    _make_reduction_func(
        "isSortedLocally",
        function_descriptor="Return True iff the array is monotonically non-decreasing "
        "on each locale where the data is stored.",
        return_descriptor="Indicates if the array is monotonically non-decreasing on each locale.",
        return_dtype="bool",
    ),
)

setattr(
    module,
    "sum",
    _make_reduction_func(
        "sum",
        function_descriptor="Return the sum of all elements in the array.",
        return_descriptor="The sum of all elements in the array.",
        return_dtype="float64",
    ),
)

setattr(
    module,
    "prod",
    _make_reduction_func(
        "prod",
        function_descriptor="Return the product of all elements in the array. "
        "Return value is always a np.float64 or np.int64",
        return_descriptor="The product calculated from the pda.",
        return_dtype="numpy_scalars",
    ),
)

setattr(
    module,
    "min",
    _make_reduction_func(
        "min",
        function_descriptor="Return the minimum value of the array.",
        return_descriptor="The min calculated from the pda.",
        return_dtype="numpy_scalars",
    ),
)


setattr(
    module,
    "max",
    _make_reduction_func(
        "max",
        function_descriptor="Return the maximum value of the array.",
        return_descriptor="The max calculated from the pda.",
        return_dtype="numpy_scalars",
    ),
)

setattr(
    module,
    "argmin",
    _make_index_reduction_func(
        "argmin",
        function_descriptor="Return the argmin of the array along the specified axis.  "
        "This is returned as the ordered index.",
        return_descriptor="This argmin of the array.",
        return_dtype="int64, uint64",
    ),
)

setattr(
    module,
    "argmax",
    _make_index_reduction_func(
        "argmax",
        function_descriptor="Return the argmax of the array along the specified axis.  "
        "This is returned as the ordered index.",
        return_descriptor="This argmax of the array.",
        return_dtype="int64, uint64",
    ),
)


@typechecked
def dot(
    pda1: Union[np.int64, np.float64, np.uint64, pdarray],
    pda2: Union[np.int64, np.float64, np.uint64, pdarray],
) -> Union[numpy_scalars, pdarray]:
    """
    Returns the sum of the elementwise product of two arrays of the same size (the dot product) or
    the product of a singleton element and an array.

    Parameters
    ----------
    pda1 : Union[numeric_scalars, pdarray]

    pda2 : Union[numeric_scalars, pdarray]


    Returns
    -------
    Union[numeric_scalars, pdarray]
        The sum of the elementwise product pda1 and pda2 or
        the product of a singleton element and an array.

    Raises
    ------
    ValueError
        Raised if the size of pda1 is not the same as pda2

    Examples
    --------
    >>> x = ak.array([2, 3])
    >>> y = ak.array([4, 5])
    >>> ak.dot(x,y)
    23
    >>> ak.dot(x,2)
    array([4 6])
    """
    if (
        not isinstance(pda1, pdarray)
        and not isinstance(pda2, pdarray)
        and ((pda1 is akuint64 and pda2 is akint64) or (pda1 is akint64 and pda2 is akuint64))
    ):
        raise TypeError(f"incompatible types {type(pda1)}, {type(pda2)}")
    if isinstance(pda1, pdarray) and isinstance(pda2, pdarray):
        if pda1.size != pda2.size:
            raise ValueError(f"Arrays must be same size, {pda1.size}, {pda2.size}")
        else:
            return sum(pda1 * pda2)
    else:
        return pda1 * pda2


@typechecked
def mean(pda: pdarray) -> np.float64:
    """
    Return the mean of the array.

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the mean

    Returns
    -------
    np.float64
        The mean calculated from the pda sum and size

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    return parse_single_value(
        generic_msg(
            cmd=f"mean<{pda.dtype},{pda.ndim}>",
            args={"x": pda, "skipNan": False},
        )
    )


@typechecked
def var(pda: pdarray, ddof: int_scalars = 0) -> np.float64:
    """
    Return the variance of values in the array.

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the variance
    ddof : int_scalars
        "Delta Degrees of Freedom" used in calculating var

    Returns
    -------
    np.float64
        The scalar variance of the array

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    ValueError
        Raised if the ddof >= pdarray size
    RuntimeError
        Raised if there's a server-side error thrown

    See Also
    --------
    mean, std

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean((x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.
    """
    if ddof >= pda.size:
        raise ValueError("var: ddof must be less than number of values")
    return parse_single_value(
        generic_msg(
            cmd=f"var<{pda.dtype},{pda.ndim}>",
            args={"x": pda, "ddof": ddof, "skipNan": False},
        )
    )


@typechecked
def std(pda: pdarray, ddof: int_scalars = 0) -> np.float64:
    """
    Return the standard deviation of values in the array. The standard
    deviation is implemented as the square root of the variance.

    Parameters
    ----------
    pda : pdarray
        values for which to calculate the standard deviation
    ddof : int_scalars
        "Delta Degrees of Freedom" used in calculating std

    Returns
    -------
    np.float64
        The scalar standard deviation of the array

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance or ddof is not an integer
    ValueError
        Raised if ddof is an integer < 0
    RuntimeError
        Raised if there's a server-side error thrown

    See Also
    --------
    mean, var

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean, i.e., ``std = sqrt(mean((x - x.mean())**2))``.

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
    the divisor ``N - ddof`` is used instead. In standard statistical
    practice, ``ddof=1`` provides an unbiased estimator of the variance
    of the infinite population. ``ddof=0`` provides a maximum likelihood
    estimate of the variance for normally distributed variables. The
    standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.
    """
    if ddof < 0:
        raise ValueError("ddof must be an integer 0 or greater")
    return parse_single_value(
        generic_msg(
            cmd=f"std<{pda.dtype},{pda.ndim}>",
            args={"x": pda, "ddof": ddof, "skipNan": False},
        )
    )


@typechecked
def cov(x: pdarray, y: pdarray) -> np.float64:
    """
    Return the covariance of x and y

    Parameters
    ----------
    x : pdarray
        One of the pdarrays used to calculate covariance
    y : pdarray
        One of the pdarrays used to calculate covariance

    Returns
    -------
    np.float64
        The scalar covariance of the two pdarrays

    Raises
    ------
    TypeError
        Raised if x or y is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown

    See Also
    --------
    mean, var

    Notes
    -----
    The covariance is calculated by
    ``cov = ((x - x.mean()) * (y - y.mean())).sum() / (x.size - 1)``.
    """
    return parse_single_value(
        generic_msg(cmd=f"cov<{x.dtype},{x.ndim},{y.dtype},{y.ndim}>", args={"x": x, "y": y})
    )


@typechecked
def corr(x: pdarray, y: pdarray) -> np.float64:
    """
    Return the correlation between x and y

    Parameters
    ----------
    x : pdarray
        One of the pdarrays used to calculate correlation
    y : pdarray
        One of the pdarrays used to calculate correlation

    Returns
    -------
    np.float64
        The scalar correlation of the two pdarrays

    Raises
    ------
    TypeError
        Raised if x or y is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown

    See Also
    --------
    std, cov

    Notes
    -----
    The correlation is calculated by
    cov(x, y) / (x.std(ddof=1) * y.std(ddof=1))
    """
    return parse_single_value(
        generic_msg(cmd=f"corr<{x.dtype},{x.ndim},{y.dtype},{y.ndim}>", args={"x": x, "y": y})
    )


@typechecked
def divmod(
    x: Union[numeric_scalars, pdarray],
    y: Union[numeric_scalars, pdarray],
    where: Union[bool_scalars, pdarray] = True,
) -> Tuple[pdarray, pdarray]:
    """
    Parameters
    ----------
    x : numeric_scalars(float_scalars, int_scalars) or pdarray
        The dividend array, the values that will be the numerator of the floordivision and will be
        acted on by the bases for modular division.
    y : numeric_scalars(float_scalars, int_scalars) or pdarray
        The divisor array, the values that will be the denominator of the division and will be the
        bases for the modular division.
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True, the
        corresponding value will be divided using floor and modular division. Elsewhere, it will retain
        its original value. Default set to True.

    Returns
    -------
    (pdarray, pdarray)
        Returns a tuple that contains quotient and remainder of the division

    Raises
    ------
    TypeError
        At least one entry must be a pdarray
    ValueError
        If both inputs are both pdarrays, their size must match
    ZeroDivisionError
        No entry in y is allowed to be 0, to prevent division by zero

    Notes
    -----
    The div is calculated by x // y
    The mod is calculated by x % y

    Examples
    --------
    >>> x = ak.arange(5, 10)
    >>> y = ak.array([2, 1, 4, 5, 8])
    >>> ak.divmod(x,y)
    (array([2 6 1 1 1]), array([1 0 3 3 1]))
    >>> ak.divmod(x,y, x % 2 == 0)
    (array([5 6 7 1 9]), array([5 0 7 3 9]))
    """
    from arkouda.numpy import cast as akcast
    from arkouda.numpy import where as akwhere
    from arkouda.pdarraycreation import full

    if not isinstance(x, pdarray) and not isinstance(y, pdarray):
        raise TypeError("At least one entry must be a pdarray.")

    if isinstance(x, pdarray) and isinstance(y, pdarray):
        if x.size != y.size:
            raise ValueError(f"size mismatch {x.size} {y.size}")

    equal_zero = y == 0
    if equal_zero if isinstance(equal_zero, bool) else any(equal_zero):
        raise ZeroDivisionError("Can not divide by zero")

    if where is True:
        return x // y, x % y  # type: ignore
    elif where is False:
        if not isinstance(x, pdarray) and isinstance(y, pdarray):
            x = full(y.size, x)
        return x, x  # type: ignore
    else:
        div = cast(pdarray, x // y)
        mod = cast(pdarray, x % y)
        return (akwhere(where, div, akcast(x, div.dtype)), akwhere(where, mod, akcast(x, mod.dtype)))


@typechecked
def mink(pda: pdarray, k: int_scalars) -> pdarray:
    """
    Find the `k` minimum values of an array.

    Returns the smallest `k` values of an array, sorted

    Parameters
    ----------
    pda : pdarray
        Input array.
    k : int_scalars
        The desired count of minimum values to be returned by the output.

    Returns
    -------
    pdarray
        The minimum `k` values from pda, sorted

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray
    ValueError
        Raised if the pda is empty or k < 1

    Notes
    -----
    This call is equivalent in value to:

        a[ak.argsort(a)[:k]]

    and generally outperforms this operation.

    This reduction will see a significant drop in performance as `k` grows
    beyond a certain value. This value is system dependent, but generally
    about a `k` of 5 million is where performance degredation has been observed.

    Examples
    --------
    >>> A = ak.array([10,5,1,3,7,2,9,0])
    >>> ak.mink(A, 3)
    array([0, 1, 2])
    >>> ak.mink(A, 4)
    array([0, 1, 2, 3])
    """
    if k < 1:
        raise ValueError("k must be 1 or greater")
    if pda.size == 0:
        raise ValueError("must be a non-empty pdarray of type int or float")

    repMsg = generic_msg(cmd="mink", args={"array": pda, "k": k, "rtnInd": False})
    return create_pdarray(cast(str, repMsg))


@typechecked
def maxk(pda: pdarray, k: int_scalars) -> pdarray:
    """
    Find the `k` maximum values of an array.

    Returns the largest `k` values of an array, sorted

    Parameters
    ----------
    pda : pdarray
        Input array.
    k : int_scalars
        The desired count of maximum values to be returned by the output.

    Returns
    -------
    pdarray, int
        The maximum `k` values from pda, sorted

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray or k is not an integer
    ValueError
        Raised if the pda is empty or k < 1

    Notes
    -----
    This call is equivalent in value to:

        a[ak.argsort(a)[k:]]

    and generally outperforms this operation.

    This reduction will see a significant drop in performance as `k` grows
    beyond a certain value. This value is system dependent, but generally
    about a `k` of 5 million is where performance degredation has been observed.


    Examples
    --------
    >>> A = ak.array([10,5,1,3,7,2,9,0])
    >>> ak.maxk(A, 3)
    array([7, 9, 10])
    >>> ak.maxk(A, 4)
    array([5, 7, 9, 10])
    """
    if k < 1:
        raise ValueError("k must be 1 or greater")
    if pda.size == 0:
        raise ValueError("must be a non-empty pdarray of type int or float")

    repMsg = generic_msg(cmd="maxk", args={"array": pda, "k": k, "rtnInd": False})
    return create_pdarray(repMsg)


@typechecked
def argmink(pda: pdarray, k: int_scalars) -> pdarray:
    """
    Finds the indices corresponding to the `k` minimum values of an array.

    Parameters
    ----------
    pda : pdarray
        Input array.
    k : int_scalars
        The desired count of indices corresponding to minimum array values

    Returns
    -------
    pdarray, int
        The indices of the minimum `k` values from the pda, sorted

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray or k is not an integer
    ValueError
        Raised if the pda is empty or k < 1

    Notes
    -----
    This call is equivalent in value to:

        ak.argsort(a)[:k]

    and generally outperforms this operation.

    This reduction will see a significant drop in performance as `k` grows
    beyond a certain value. This value is system dependent, but generally
    about a `k` of 5 million is where performance degradation has been observed.

    Examples
    --------
    >>> A = ak.array([10,5,1,3,7,2,9,0])
    >>> ak.argmink(A, 3)
    array([7, 2, 5])
    >>> ak.argmink(A, 4)
    array([7, 2, 5, 3])
    """
    if k < 1:
        raise ValueError("k must be 1 or greater")
    if pda.size == 0:
        raise ValueError("must be a non-empty pdarray of type int or float")

    repMsg = generic_msg(cmd="mink", args={"array": pda, "k": k, "rtnInd": True})
    return create_pdarray(repMsg)


@typechecked
def argmaxk(pda: pdarray, k: int_scalars) -> pdarray:
    """
    Find the indices corresponding to the `k` maximum values of an array.

    Returns the largest `k` values of an array, sorted

    Parameters
    ----------
    pda : pdarray
        Input array.
    k : int_scalars
        The desired count of indices corresponding to maxmum array values

    Returns
    -------
    pdarray, int
        The indices of the maximum `k` values from the pda, sorted

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray or k is not an integer
    ValueError
        Raised if the pda is empty or k < 1

    Notes
    -----
    This call is equivalent in value to:

        ak.argsort(a)[k:]

    and generally outperforms this operation.

    This reduction will see a significant drop in performance as `k` grows
    beyond a certain value. This value is system dependent, but generally
    about a `k` of 5 million is where performance degradation has been observed.


    Examples
    --------
    >>> A = ak.array([10,5,1,3,7,2,9,0])
    >>> ak.argmaxk(A, 3)
    array([4, 6, 0])
    >>> ak.argmaxk(A, 4)
    array([1, 4, 6, 0])
    """
    if k < 1:
        raise ValueError("k must be 1 or greater")
    if pda.size == 0:
        raise ValueError("must be a non-empty pdarray of type int or float")

    repMsg = generic_msg(cmd="maxk", args={"array": pda, "k": k, "rtnInd": True})
    return create_pdarray(repMsg)


def popcount(pda: pdarray) -> pdarray:
    """
    Find the population (number of bits set) for each integer in an array.

    Parameters
    ----------
    pda : pdarray, int64, uint64, bigint
        Input array (must be integral).

    Returns
    -------
    population : pdarray
        The number of bits set (1) in each element

    Raises
    ------
    TypeError
        If input array is not int64, uint64, or bigint

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.popcount(A)
    array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2])
    """
    if pda.dtype not in [akint64, akuint64, bigint]:
        raise TypeError("BitOps only supported on int64, uint64, and bigint arrays")
    if pda.dtype == bigint:
        from builtins import sum

        return sum(popcount(a) for a in pda.bigint_to_uint_arrays())  # type: ignore
    else:
        repMsg = generic_msg(
            cmd=f"popcount<{pda.dtype},{pda.ndim}>",
            args={
                "pda": pda,
            },
        )
        return create_pdarray(repMsg)


def parity(pda: pdarray) -> pdarray:
    """
    Find the bit parity (XOR of all bits) for each integer in an array.

    Parameters
    ----------
    pda : pdarray, int64, uint64, bigint
        Input array (must be integral).

    Returns
    -------
    parity : pdarray
        The parity of each element: 0 if even number of bits set, 1 if odd.

    Raises
    ------
    TypeError
        If input array is not int64, uint64, or bigint

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.parity(A)
    array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    """
    if pda.dtype not in [akint64, akuint64, bigint]:
        raise TypeError("BitOps only supported on int64, uint64, and bigint arrays")
    if pda.dtype == bigint:
        # XOR the parity of the underlying uint array to get the parity of the bigint array
        return reduce(lambda x, y: x ^ y, [parity(a) for a in pda.bigint_to_uint_arrays()])
    else:
        repMsg = generic_msg(
            cmd=f"parity<{pda.dtype},{pda.ndim}>",
            args={
                "pda": pda,
            },
        )
        return create_pdarray(repMsg)


def clz(pda: pdarray) -> pdarray:
    """
    Count leading zeros for each integer in an array.

    Parameters
    ----------
    pda : pdarray, int64, uint64, bigint
        Input array (must be integral).

    Returns
    -------
    lz : pdarray
        The number of leading zeros of each element.

    Raises
    ------
    TypeError
        If input array is not int64, uint64, or bigint

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.clz(A)
    array([64, 63, 62, 62, 61, 61, 61, 61, 60, 60])
    """
    if pda.dtype not in [akint64, akuint64, bigint]:
        raise TypeError("BitOps only supported on int64, uint64, and bigint arrays")
    if pda.dtype == bigint:
        if pda.max_bits == -1:
            raise ValueError("max_bits must be set to count leading zeros")
        from arkouda.numpy import where
        from arkouda.pdarraycreation import zeros

        uint_arrs = pda.bigint_to_uint_arrays()

        # we need to adjust the number of leading zeros to account for max_bits

        mod_max_bits, div_max_bits = pda.max_bits % 64, ceil(pda.max_bits / 64)

        # if we don't fall on a 64 bit boundary, we need to subtract off
        # leading zeros that aren't settable due to max_bits restrictions

        sub_off = 0 if mod_max_bits == 0 else 64 - mod_max_bits

        # we can have fewer uint arrays than max_bits allows if all the high bits are zero
        # i.e. ak.arange(10, dtype=ak.bigint, max_bits=256) will only store one uint64 array,
        # so we need to add on any leading zeros from empty higher bit arrays that were excluded
        add_on = 64 * (div_max_bits - len(uint_arrs))

        lz = zeros(pda.size, dtype=akuint64)
        previously_non_zero = zeros(pda.size, dtype=bool)
        for a in uint_arrs:

            # if a bit was set somewhere in the higher bits,
            # we don't want to add its clz to our leading zeros count
            # so only update positions where we've only seen zeros

            lz += where(previously_non_zero, 0, clz(a)).astype(akuint64)
            # note: the above cast is required or the += will fail on mixed types

            # OR in the places where the current bits have a bit set

            previously_non_zero |= a != 0
            if all(previously_non_zero):
                break

        lz += add_on - sub_off
        return lz

    else:
        repMsg = generic_msg(
            cmd=f"clz<{pda.dtype},{pda.ndim}>",
            args={
                "pda": pda,
            },
        )
        return create_pdarray(repMsg)


def ctz(pda: pdarray) -> pdarray:
    """
    Count trailing zeros for each integer in an array.

    Parameters
    ----------
    pda : pdarray, int64, uint64, bigint
        Input array (must be integral).

    Returns
    -------
    lz : pdarray
        The number of trailing zeros of each element.

    Notes
    -----
    ctz(0) is defined to be zero.

    Raises
    ------
    TypeError
        If input array is not int64, uint64, or bigint

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.ctz(A)
    array([0, 0, 1, 0, 2, 0, 1, 0, 3, 0])
    """
    if pda.dtype not in [akint64, akuint64, bigint]:
        raise TypeError("BitOps only supported on int64, uint64, and bigint arrays")
    if pda.dtype == bigint:

        # we don't need max_bits to be set because that only limits the high bits
        # which is only relevant when ctz(0) which is defined to be 0

        from arkouda.numpy import where
        from arkouda.pdarraycreation import zeros

        # reverse the list, so we visit low bits first

        reversed_uint_arrs = pda.bigint_to_uint_arrays()[::-1]
        tz = zeros(pda.size, dtype=akuint64)
        previously_non_zero = zeros(pda.size, dtype=bool)

        for a in reversed_uint_arrs:

            # if the lower bits are all zero, we want trailing zeros
            # to be 64 because the higher bits could still be set.
            # But ctz(0) is defined to be 0, so use 64 in that case

            a_is_zero = a == 0
            num_zeros = where(a_is_zero, 64, ctz(a))

            # if a bit was set somewhere in the lower bits,
            # we don't want to add its ctz to our trailing zeros count
            # so only update positions where we've only seen zeros

            tz += where(previously_non_zero, 0, num_zeros).astype(akuint64)
            # note: the above cast is required or the += will fail on mixed types

            # OR in the places where the current bits have a bit set

            previously_non_zero |= ~a_is_zero
            if all(previously_non_zero):
                break

        if not all(previously_non_zero):  # ctz(0) is defined to be 0
            tz[~previously_non_zero] = 0

        return tz

    else:

        repMsg = generic_msg(
            cmd=f"ctz<{pda.dtype},{pda.ndim}>",
            args={
                "pda": pda,
            },
        )
        return create_pdarray(repMsg)


def rotl(x, rot) -> pdarray:
    """
    Rotate bits of <x> to the left by <rot>.

    Parameters
    ----------
    x : pdarray(int64/uint64) or integer
        Value(s) to rotate left.
    rot : pdarray(int64/uint64) or integer
        Amount(s) to rotate by.

    Returns
    -------
    rotated : pdarray(int64/uint64)
        The rotated elements of x.

    Raises
    ------
    TypeError
        If input array is not int64 or uint64

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.rotl(A, A)
    array([0, 2, 8, 24, 64, 160, 384, 896, 2048, 4608])
    """
    if isinstance(x, pdarray) and x.dtype in [akint64, akuint64, bigint]:
        if (isinstance(rot, pdarray) and rot.dtype in [akint64, akuint64]) or isSupportedInt(rot):
            return x._binop(rot, "<<<")
        else:
            raise TypeError("Rotations only supported on integers")
    elif isSupportedInt(x) and isinstance(rot, pdarray) and rot.dtype in [akint64, akuint64]:
        return rot._r_binop(x, "<<<")
    else:
        raise TypeError("Rotations only supported on integers")


def rotr(x, rot) -> pdarray:
    """
    Rotate bits of <x> to the left by <rot>.

    Parameters
    ----------
    x : pdarray(int64/uint64) or integer
        Value(s) to rotate left.
    rot : pdarray(int64/uint64) or integer
        Amount(s) to rotate by.

    Returns
    -------
    rotated : pdarray(int64/uint64)
        The rotated elements of x.

    Raises
    ------
    TypeError
        If input array is not int64 or uint64

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.rotr(1024 * A, A)
    array([0, 512, 512, 384, 256, 160, 96, 56, 32, 18])
    """
    if isinstance(x, pdarray) and x.dtype in [akint64, akuint64, bigint]:
        if (isinstance(rot, pdarray) and rot.dtype in [akint64, akuint64]) or isSupportedInt(rot):
            return x._binop(rot, ">>>")
        else:
            raise TypeError("Rotations only supported on integers")
    elif isSupportedInt(x) and isinstance(rot, pdarray) and rot.dtype in [akint64, akuint64]:
        return rot._r_binop(x, ">>>")
    else:
        raise TypeError("Rotations only supported on integers")


@typechecked
def power(
    pda: pdarray, pwr: Union[int, float, pdarray], where: Union[bool_scalars, pdarray] = True
) -> pdarray:
    """
    Raises an array to a power. If where is given, the operation will only take place in the positions
    where the where condition is True.

    Note:
    Our implementation of the where argument deviates from numpy. The difference in behavior occurs
    at positions where the where argument contains a False. In numpy, these position will have
    uninitialized memory (which can contain anything and will vary between runs). We have chosen to
    instead return the value of the original array in these positions.

    Parameters
    ----------
    pda : pdarray
        A pdarray of values that will be raised to a power (pwr)
    pwr : integer, float, or pdarray
        The power(s) that pda is raised to
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True, the
        corresponding value will be raised to the respective power. Elsewhere, it will retain its
        original value. Default set to True.

    Returns
    -------
        pdarray
        Returns a pdarray of values raised to a power, under the boolean where condition.

    Examples
    --------
    >>> a = ak.arange(5)
    >>> ak.power(a, 3)
    array([0, 1, 8, 27, 64])
    >>> ak.power(a), 3, a % 2 == 0)
    array([0, 1, 8, 3, 64])
    """
    from arkouda.numpy import cast as akcast
    from arkouda.numpy import where as akwhere

    if where is True:
        return pda**pwr
    elif where is False:
        return pda
    else:
        exp = pda**pwr
        return akwhere(where, exp, akcast(pda, exp.dtype))


@typechecked
def sqrt(pda: pdarray, where: Union[bool_scalars, pdarray] = True) -> pdarray:
    """
    Takes the square root of array. If where is given, the operation will only take place in
    the positions where the where condition is True.

    Parameters
    ----------
    pda : pdarray
        A pdarray of values that will be square rooted
    where : Boolean or pdarray
        This condition is broadcast over the input. At locations where the condition is True, the
        corresponding value will be square rooted. Elsewhere, it will retain its original value.
        Default set to True.

    Returns
    -------
        pdarray
        Returns a pdarray of square rooted values, under the boolean where condition.

    Examples:
    >>> a = ak.arange(5)
    >>> ak.sqrt(a)
    array([0 1 1.4142135623730951 1.7320508075688772 2])
    >>> ak.sqrt(a, ak.sqrt([True, True, False, False, True]))
    array([0, 1, 2, 3, 2])
    """
    return power(pda, 0.5, where)


@typechecked
def skew(pda: pdarray, bias: bool = True) -> np.float64:
    """
    Computes the sample skewness of an array.
    Skewness > 0 means there's greater weight in the right tail of the distribution.
    Skewness < 0 means there's greater weight in the left tail of the distribution.
    Skewness == 0 means the data is normally distributed.
    Based on the `scipy.stats.skew` function.

    Parameters
    ----------
    pda : pdarray
        A pdarray of values that will be calculated to find the skew
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.

    Returns
    -------
        np.float64
            The skew of all elements in the array

    Examples:
    >>> a = ak.array([1, 1, 1, 5, 10])
    >>> ak.skew(a)
    0.9442193396379163
    """

    deviations = pda - pda.mean()
    cubed_deviations = deviations**3

    std_dev = pda.std()

    if std_dev != 0:
        skewness = cubed_deviations.mean() / (std_dev**3)
        # Apply bias correction using the Fisher-Pearson method
        if not bias:
            n = len(pda)
            correction = np.sqrt((n - 1) * n) / (n - 2)
            skewness = correction * skewness
    else:
        skewness = 0

    return skewness


# there's no need for typechecking, % can handle that
def mod(dividend, divisor) -> pdarray:
    """
    Returns the element-wise remainder of division.

    Computes the remainder complementary to the floor_divide function.
    It is equivalent to np.mod, the remainder has the same sign as the divisor.

    Parameters
    ----------
    dividend
        The array being acted on by the bases for the modular division.
    divisor
        The array that will be the bases for the modular division.

    Returns
    -------
    pdarray
        Returns an array that contains the element-wise remainder of division.
    """
    return dividend % divisor


@typechecked
def fmod(dividend: Union[pdarray, numeric_scalars], divisor: Union[pdarray, numeric_scalars]) -> pdarray:
    """
    Returns the element-wise remainder of division.

    It is equivalent to np.fmod, the remainder has the same sign as the dividend.

    Parameters
    ----------
    dividend : numeric scalars or pdarray
        The array being acted on by the bases for the modular division.
    divisor : numeric scalars or pdarray
        The array that will be the bases for the modular division.

    Returns
    -------
    pdarray
        Returns an array that contains the element-wise remainder of division.

    Raises
    ------
    TypeError
        Raised if neither dividend nor divisor is a pdarray (at least one must be)
        or if any scalar or pdarray element is not one of int, uint, float, bigint
    """
    if not builtins.all(
        isSupportedNumber(arg) or isinstance(arg, pdarray) for arg in [dividend, divisor]
    ):
        raise TypeError(
            f"Unsupported types {type(dividend)} and/or {type(divisor)}. Supported "
            "types are numeric scalars and pdarrays. At least one argument must be a pdarray."
        )
    if isSupportedNumber(dividend) and isSupportedNumber(divisor):
        raise TypeError(
            f"Unsupported types {type(dividend)} and/or {type(divisor)}. Supported "
            "types are numeric scalars and pdarrays. At least one argument must be a pdarray."
        )
    # TODO: handle shape broadcasting for multidimensional arrays


#   The code below creates a command string for fmod2vv, fmod2vs or fmod2sv.

    if isinstance(dividend, pdarray) and isinstance(divisor, pdarray) :
        cmdstring = f"fmod2vv<{dividend.dtype},{dividend.ndim},{divisor.dtype}>"

    elif isinstance(dividend, pdarray) and not (isinstance(divisor, pdarray)) :
        if resolve_scalar_dtype(divisor) in ['float64', 'int64', 'uint64', 'bool'] :
            acmd = 'fmod2vs_'+resolve_scalar_dtype(divisor)
        else :  # this condition *should* be impossible because of the isSupportedNumber check
            raise TypeError(f"Scalar divisor type {resolve_scalar_dtype(divisor)} not allowed in fmod")
        cmdstring = f"{acmd}<{dividend.dtype},{dividend.ndim}>"

    elif not (isinstance(dividend, pdarray) and isinstance(divisor, pdarray)) :
        if resolve_scalar_dtype(dividend) in ['float64', 'int64', 'uint64', 'bool'] :
            acmd = 'fmod2sv_'+resolve_scalar_dtype(dividend)
        else :  # this condition *should* be impossible because of the isSupportedNumber check
            raise TypeError(f"Scalar dividend type {resolve_scalar_dtype(dividend)} not allowed in fmod")
        cmdstring = f"{acmd}<{divisor.dtype},{divisor.ndim}>"  # type: ignore[union-attr]

    else:
        m = mod(dividend, divisor)
        return _create_scalar_array(m)

#   We reach here if this was any case other than scalar & scalar

    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=cmdstring,
                args={
                    "a": dividend,
                    "b": divisor,
                },
            ),
        )
    )


@typechecked
def broadcast_if_needed(x1: pdarray, x2: pdarray) -> Tuple[pdarray, pdarray, bool, bool]:
    from arkouda.util import broadcast_dims

    if x1.shape == x2.shape:
        return (x1, x2, False, False)
    else:
        tmp_x1 = False
        tmp_x2 = False
        try:
            # determine common shape for broadcasting
            bc_shape = broadcast_dims(x1.shape, x2.shape)
        except ValueError:
            raise ValueError(
                f"Incompatible array shapes for broadcasted operation: {x1.shape} and {x2.shape}"
            )

        # broadcast x1 if needed
        if bc_shape != x1.shape:
            x1b = broadcast_to_shape(x1, bc_shape)
            tmp_x1 = True
        else:
            x1b = x1

        # broadcast x2 if needed
        if bc_shape != x2.shape:
            x2b = broadcast_to_shape(x2, bc_shape)
            tmp_x2 = True
        else:
            x2b = x2
        return (x1b, x2b, tmp_x1, tmp_x2)


@typechecked
def broadcast_to_shape(pda: pdarray, shape: Tuple[int, ...]) -> pdarray:
    """
    expand an array's rank to the specified shape using broadcasting
    """

    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"broadcast<{pda.dtype},{pda.ndim},{len(shape)}>",
                args={
                    "name": pda,
                    "shape": shape,
                },
            ),
        )
    )


@typechecked
def attach_pdarray(user_defined_name: str) -> pdarray:
    """
    class method to return a pdarray attached to the registered name in the arkouda
    server which was registered using register()

    Parameters
    ----------
    user_defined_name : str
        user defined name which array was registered under

    Returns
    -------
    pdarray
        pdarray which is bound to the corresponding server side component which was registered
        with user_defined_name

    Raises
    ------
    TypeError
      Raised if user_defined_name is not a str

    See also
    --------
    attach, register, unregister, is_registered, unregister_pdarray_by_name, list_registry

    Notes
    -----
    Registered names/pdarrays in the server are immune to deletion
    until they are unregistered.

    Examples
    --------
    >>> a = zeros(100)
    >>> a.register("my_zeros")
    >>> # potentially disconnect from server and reconnect to server
    >>> b = ak.attach_pdarray("my_zeros")
    >>> # ...other work...
    >>> b.unregister()
    """
    import warnings

    from arkouda.util import attach

    warnings.warn(
        "ak.attach_pdarray() is deprecated. Please use ak.attach() instead.",
        DeprecationWarning,
    )
    return attach(user_defined_name)


@typechecked
def attach(user_defined_name: str) -> pdarray:
    """
    class method to return a pdarray attached to the registered name in the arkouda
    server which was registered using register()

    Parameters
    ----------
    user_defined_name : str
        user defined name which array was registered under

    Returns
    -------
    pdarray
        pdarray which is bound to the corresponding server side component which was registered
        with user_defined_name

    Raises
    ------
    TypeError
      Raised if user_defined_name is not a str

    See also
    --------
    register, unregister, is_registered, unregister_pdarray_by_name, list_registry

    Notes
    -----
    Registered names/pdarrays in the server are immune to deletion
    until they are unregistered.

    Examples
    --------
    >>> a = zeros(100)
    >>> a.register("my_zeros")
    >>> # potentially disconnect from server and reconnect to server
    >>> b = ak.pdarrayclass.attach("my_zeros")
    >>> # ...other work...
    >>> b.unregister()
    """
    import warnings

    from arkouda.util import attach

    warnings.warn(
        "ak.pdarrayclass.attach() is deprecated. Please use ak.attach() instead.",
        DeprecationWarning,
    )
    return attach(user_defined_name)


@typechecked
def unregister_pdarray_by_name(user_defined_name: str) -> None:
    """
    Unregister a named pdarray in the arkouda server which was previously
    registered using register() and/or attahced to using attach_pdarray()

    Parameters
    ----------
    user_defined_name : str
        user defined name which array was registered under

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Raised if the server could not find the internal name/symbol to remove

    See also
    --------
    register, unregister, is_registered, list_registry, attach

    Notes
    -----
    Registered names/pdarrays in the server are immune to deletion until
    they are unregistered.

    Examples
    --------
    >>> a = zeros(100)
    >>> a.register("my_zeros")
    >>> # potentially disconnect from server and reconnect to server
    >>> b = ak.attach_pdarray("my_zeros")
    >>> # ...other work...
    >>> ak.unregister_pdarray_by_name(b)
    """
    import warnings

    from arkouda.util import unregister

    warnings.warn(
        "ak.unregister_pdarray_by_name() is deprecated. Please use ak.unregister() instead.",
        DeprecationWarning,
    )
    return unregister(user_defined_name)


# TODO In the future move this to a specific errors file
class RegistrationError(Exception):
    """Error/Exception used when the Arkouda Server cannot register an object"""
