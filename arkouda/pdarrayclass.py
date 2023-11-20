from __future__ import annotations

import builtins
import json
from functools import reduce
from math import ceil
from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np  # type: ignore
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.dtypes import NUMBER_FORMAT_STRINGS, DTypes, bigint
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import bool as npbool
from arkouda.dtypes import dtype, get_server_byteorder
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import (
    int_scalars,
    isSupportedInt,
    isSupportedNumber,
    numeric_and_bool_scalars,
    numeric_scalars,
    numpy_scalars,
    resolve_scalar_dtype,
)
from arkouda.dtypes import str_ as akstr_
from arkouda.dtypes import translate_np_dtype
from arkouda.dtypes import uint64 as akuint64
from arkouda.infoclass import information, pretty_print_information
from arkouda.logger import getArkoudaLogger

__all__ = [
    "pdarray",
    "clear",
    "any",
    "all",
    "is_sorted",
    "sum",
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
]

logger = getArkoudaLogger(name="pdarrayclass")


@typechecked
def parse_single_value(msg: str) -> object:
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
    if mydtype == npbool:
        if value == "True":
            return mydtype.type(True)
        elif value == "False":
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
        self.shape = shape
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
        return self.shape[0]

    def __str__(self):
        from arkouda.client import pdarrayIterThresh

        return generic_msg(cmd="str", args={"array": self, "printThresh": pdarrayIterThresh})

    def __repr__(self):
        from arkouda.client import pdarrayIterThresh

        return generic_msg(cmd="repr", args={"array": self, "printThresh": pdarrayIterThresh})

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
        if self.dtype == bool:
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
            if self.size != other.size:
                raise ValueError(f"size mismatch {self.size} {other.size}")
            repMsg = generic_msg(cmd="binopvv", args={"op": op, "a": self, "b": other})
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
            cmd="binopvs",
            args={"op": op, "a": self, "dtype": dt, "value": other},
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
            cmd="binopsv",
            args={"op": op, "dtype": dt, "value": other, "a": self},
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

    def __eq__(self, other):
        if (self.dtype == bool) and (isinstance(other, pdarray) and (other.dtype == bool)):
            return ~(self ^ other)
        else:
            return self._binop(other, "==")

    def __ne__(self, other):
        if (self.dtype == bool) and (isinstance(other, pdarray) and (other.dtype == bool)):
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
        if self.dtype == bool:
            return self._binop(True, "^")
        raise TypeError(f"Unhandled dtype: {self} ({self.dtype})")

    # op= operators
    def opeq(self, other, op):
        if op not in self.OpEqOps:
            raise ValueError(f"bad operator {op}")
        # pdarray op= pdarray
        if isinstance(other, pdarray):
            if self.size != other.size:
                raise ValueError(f"size mismatch {self.size} {other.size}")
            generic_msg(cmd="opeqvv", args={"op": op, "a": self, "b": other})
            return self
        # pdarray binop scalar
        # opeq requires scalar to be cast as pdarray dtype
        try:
            other = self.dtype.type(other)
        except Exception:
            # Can't cast other as dtype of pdarray
            raise TypeError(f"Unhandled scalar type: {other} ({type(other)})")

        generic_msg(
            cmd="opeqvs",
            args={"op": op, "a": self, "dtype": self.dtype.name, "value": self.format_other(other)},
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
        if np.isscalar(key) and (resolve_scalar_dtype(key) in ["int64", "uint64"]):
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if key >= 0 and key < self.size:
                repMsg = generic_msg(
                    cmd="[int]",
                    args={
                        "array": self,
                        "idx": key,
                    },
                )
                fields = repMsg.split()
                # value = fields[2]
                return parse_single_value(" ".join(fields[1:]))
            else:
                raise IndexError(f"[int] {orig_key} is out of bounds with size {self.size}")
        if isinstance(key, slice):
            (start, stop, stride) = key.indices(self.size)
            logger.debug("start: {} stop: {} stride: {}".format(start, stop, stride))
            repMsg = generic_msg(
                cmd="[slice]",
                args={
                    "array": self,
                    "start": start,
                    "stop": stop,
                    "stride": stride,
                },
            )
            return create_pdarray(repMsg)
        if isinstance(key, pdarray):
            kind, _ = translate_np_dtype(key.dtype)
            if kind not in ("bool", "int", "uint"):
                raise TypeError(f"unsupported pdarray index type {key.dtype}")
            if kind == "bool" and self.size != key.size:
                raise ValueError(f"size mismatch {self.size} {key.size}")
            repMsg = generic_msg(
                cmd="[pdarray]",
                args={
                    "array": self,
                    "idx": key,
                },
            )
            return create_pdarray(repMsg)
        else:
            raise TypeError(f"Unhandled key type: {key} ({type(key)})")

    def __setitem__(self, key, value):
        if np.isscalar(key) and (resolve_scalar_dtype(key) in ["int64", "uint64"]):
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if key >= 0 and key < self.size:
                generic_msg(
                    cmd="[int]=val",
                    args={
                        "array": self,
                        "idx": key,
                        "dtype": self.dtype,
                        "value": self.format_other(value),
                    },
                )
            else:
                raise IndexError(f"index {orig_key} is out of bounds with size {self.size}")
        elif isinstance(key, pdarray):
            if isinstance(value, pdarray):
                generic_msg(cmd="[pdarray]=pdarray", args={"array": self, "idx": key, "value": value})
            else:
                generic_msg(
                    cmd="[pdarray]=val",
                    args={
                        "array": self,
                        "idx": key,
                        "dtype": self.dtype,
                        "value": self.format_other(value),
                    },
                )
        elif isinstance(key, slice):
            (start, stop, stride) = key.indices(self.size)
            logger.debug(f"start: {start} stop: {stop} stride: {stride}")
            if isinstance(value, pdarray):
                generic_msg(
                    cmd="[slice]=pdarray",
                    args={
                        "array": self,
                        "start": start,
                        "stop": stop,
                        "stride": stride,
                        "value": value,
                    },
                )
            else:
                generic_msg(
                    cmd="[slice]=val",
                    args={
                        "array": self,
                        "start": start,
                        "stop": stop,
                        "stride": stride,
                        "dtype": self.dtype,
                        "value": self.format_other(value),
                    },
                )
        else:
            raise TypeError(f"Unhandled key type: {key} ({type(key)})")

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
        cmd = f"set{self.ndim}D"
        generic_msg(
            cmd=cmd, args={"array": self, "dtype": self.dtype.name, "val": self.format_other(value)}
        )

    def any(self) -> np.bool_:
        """
        Return True iff any element of the array evaluates to True.
        """
        return any(self)

    def all(self) -> np.bool_:
        """
        Return True iff all elements of the array evaluate to True.
        """
        return all(self)

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

    def is_sorted(self) -> np.bool_:
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
        return is_sorted(self)

    def sum(self) -> numeric_and_bool_scalars:
        """
        Return the sum of all elements in the array.
        """
        return sum(self)

    def prod(self) -> np.float64:
        """
        Return the product of all elements in the array. Return value is
        always a np.float64 or np.int64.
        """
        return prod(self)

    def min(self) -> numpy_scalars:
        """
        Return the minimum value of the array.
        """
        return min(self)

    def max(self) -> numpy_scalars:
        """
        Return the maximum value of the array.
        """
        return max(self)

    def argmin(self) -> Union[np.int64, np.uint64]:
        """
        Return the index of the first occurrence of the array min value
        """
        return argmin(self)

    def argmax(self) -> Union[np.int64, np.uint64]:
        """
        Return the index of the first occurrence of the array max value.
        """
        return argmax(self)

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
        from arkouda.numeric import value_counts

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
        from arkouda.numeric import cast as akcast

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

    def reshape(self, *shape, order="row_major"):
        """
        Gives a new shape to an array without changing its data.

        Parameters
        ----------
        shape : int, tuple of ints, or pdarray
            The new shape should be compatible with the original shape.
        order : str {'row_major' | 'C' | 'column_major' | 'F'}
            Read the elements of the pdarray in this index order
            By default, read the elements in row_major or C-like order where the last index
            changes the fastest
            If 'column_major' or 'F', read the elements in column_major or Fortran-like order where the
            first index changes the fastest

        Returns
        -------
        ArrayView
            An arrayview object with the data from the array but with the new shape
        """
        from arkouda.array_view import ArrayView

        # allows the elements of the shape parameter to be passed in as separate arguments
        # For example, a.reshape(10, 11) is equivalent to a.reshape((10, 11))
        if len(shape) == 1:
            shape = shape[0]
        elif not isinstance(shape, pdarray):
            shape = [i for i in shape]
        return ArrayView(base=self, shape=shape, order=order)

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
            generic_msg(cmd="tondarray", args={"array": self}, recv_binary=True),
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
            return np.frombuffer(data, dt).copy()
        else:
            return np.frombuffer(data, dt)

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

    def _get_grouping_keys(self) -> List[pdarray]:
        """
        Private method for generating grouping keys used by GroupBy.

        API: this method must be defined by all groupable arrays, and it
        must return a list of arrays that can be (co)argsorted.
        """
        if self.dtype == akbool:
            from arkouda.numeric import cast as akcast

            return [akcast(self, akint64)]
        elif self.dtype in (akint64, akuint64):
            # Integral pdarrays are their own grouping keys
            return [self]
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


@typechecked
def any(pda: pdarray) -> np.bool_:
    """
    Return True iff any element of the array evaluates to True.

    Parameters
    ----------
    pda : pdarray
        The pdarray instance to be evaluated

    Returns
    -------
    bool
        Indicates if 1..n pdarray elements evaluate to True

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args={"op": "any", "array": pda})
    return parse_single_value(cast(str, repMsg))


@typechecked
def all(pda: pdarray) -> np.bool_:
    """
    Return True iff all elements of the array evaluate to True.

    Parameters
    ----------
    pda : pdarray
        The pdarray instance to be evaluated

    Returns
    -------
    bool
        Indicates if all pdarray elements evaluate to True

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args={"op": "all", "array": pda})
    return parse_single_value(cast(str, repMsg))


@typechecked
def is_sorted(pda: pdarray) -> np.bool_:
    """
    Return True iff the array is monotonically non-decreasing.

    Parameters
    ----------
    pda : pdarray
        The pdarray instance to be evaluated

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
    repMsg = generic_msg(cmd="reduction", args={"op": "is_sorted", "array": pda})
    return parse_single_value(cast(str, repMsg))


@typechecked
def sum(pda: pdarray) -> np.float64:
    """
    Return the sum of all elements in the array.

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the sum

    Returns
    -------
    np.float64
        The sum of all elements in the array

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args={"op": "sum", "array": pda})
    return parse_single_value(cast(str, repMsg))


@typechecked
def prod(pda: pdarray) -> np.float64:
    """
    Return the product of all elements in the array. Return value is
    always a np.float64 or np.int64

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the product

    Returns
    -------
    numpy_scalars
        The product calculated from the pda

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args={"op": "prod", "array": pda})
    return parse_single_value(cast(str, repMsg))


def min(pda: pdarray) -> numpy_scalars:
    """
    Return the minimum value of the array.

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the min

    Returns
    -------
    numpy_scalars
        The min calculated from the pda

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args={"op": "min", "array": pda})
    return parse_single_value(cast(str, repMsg))


@typechecked
def max(pda: pdarray) -> numpy_scalars:
    """
    Return the maximum value of the array.

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the max

    Returns
    -------
    numpy_scalars:
        The max calculated from the pda

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args={"op": "max", "array": pda})
    return parse_single_value(cast(str, repMsg))


@typechecked
def argmin(pda: pdarray) -> Union[np.int64, np.uint64]:
    """
    Return the index of the first occurrence of the array min value.

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the argmin

    Returns
    -------
    Union[np.int64, np.uint64]
        The index of the argmin calculated from the pda

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args={"op": "argmin", "array": pda})
    return parse_single_value(cast(str, repMsg))


@typechecked
def argmax(pda: pdarray) -> Union[np.int64, np.uint64]:
    """
    Return the index of the first occurrence of the array max value.

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the argmax

    Returns
    -------
    Union[np.int64, np.uint64]
        The index of the argmax calculated from the pda

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args={"op": "argmax", "array": pda})
    return parse_single_value(cast(str, repMsg))


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
    return parse_single_value(generic_msg(cmd="mean", args={"x": pda}))


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
    return parse_single_value(generic_msg(cmd="var", args={"x": pda, "ddof": ddof}))


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
    return parse_single_value(generic_msg(cmd="std", args={"x": pda, "ddof": ddof}))


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
    return parse_single_value(generic_msg(cmd="cov", args={"x": x, "y": y}))


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
    return parse_single_value(generic_msg(cmd="corr", args={"x": x, "y": y}))


@typechecked
def divmod(
    x: Union[numeric_scalars, pdarray],
    y: Union[numeric_scalars, pdarray],
    where: Union[bool, pdarray] = True,
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
    from arkouda.numeric import cast as akcast
    from arkouda.numeric import where as akwhere
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

        return sum(popcount(a) for a in pda.bigint_to_uint_arrays())
    else:
        repMsg = generic_msg(
            cmd="efunc",
            args={
                "func": "popcount",
                "array": pda,
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
            cmd="efunc",
            args={
                "func": "parity",
                "array": pda,
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
        from arkouda.numeric import where
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
            # we don't want to add it's clz to our leading zeros count
            # so only update positions where we've only seen zeros
            lz += where(previously_non_zero, 0, clz(a))
            # OR in the places where the current bits have a bit set
            previously_non_zero |= a != 0
            if all(previously_non_zero):
                break
        lz += add_on - sub_off
        return lz
    else:
        repMsg = generic_msg(
            cmd="efunc",
            args={
                "func": "clz",
                "array": pda,
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
        from arkouda.numeric import where
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
            # we don't want to add it's ctz to our trailing zeros count
            # so only update positions where we've only seen zeros
            tz += where(previously_non_zero, 0, num_zeros)
            # OR in the places where the current bits have a bit set
            previously_non_zero |= ~a_is_zero
            if all(previously_non_zero):
                break
        if not all(previously_non_zero):
            # ctz(0) is defined to be 0
            tz[~previously_non_zero] = 0
        return tz
    else:
        repMsg = generic_msg(
            cmd="efunc",
            args={
                "func": "ctz",
                "array": pda,
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
def power(pda: pdarray, pwr: Union[int, float, pdarray], where: Union[bool, pdarray] = True) -> pdarray:
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
    from arkouda.numeric import cast as akcast
    from arkouda.numeric import where as akwhere

    if where is True:
        return pda**pwr
    elif where is False:
        return pda
    else:
        exp = pda**pwr
        return akwhere(where, exp, akcast(pda, exp.dtype))


@typechecked
def sqrt(pda: pdarray, where: Union[bool, pdarray] = True) -> pdarray:
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
    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd="efunc2",
                args={
                    "func": "fmod",
                    "A": dividend,
                    "B": divisor,
                },
            ),
        )
    )

@typechecked
def broadcast_to_shape(shape: Tuple[int, ...]) -> pdarray:
    """
    expand an array's rank to the specified shape using broadcasting
    """

    return create_pdarray(
        cast(
            str,
            generic_msg(
                cmd=f"broadcast{self.ndims}Dx{len(shape)}D",
                args={
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
