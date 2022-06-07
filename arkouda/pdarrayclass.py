from __future__ import annotations

import builtins
import json
from typing import List, Sequence, cast

import numpy as np  # type: ignore
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.dtypes import NUMBER_FORMAT_STRINGS, DTypes
from arkouda.dtypes import bool as akbool
from arkouda.dtypes import bool as npbool
from arkouda.dtypes import dtype, get_server_byteorder
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import (
    int_scalars,
    isSupportedInt,
    numeric_and_bool_scalars,
    numeric_scalars,
    numpy_scalars,
    resolve_scalar_dtype,
)
from arkouda.dtypes import str_ as akstr_
from arkouda.dtypes import translate_np_dtype
from arkouda.dtypes import uint64 as akuint64
from arkouda.infoclass import information, list_registry, pretty_print_information
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
    OpEqOps = frozenset(["+=", "-=", "*=", "/=", "//=", "&=", "|=", "^=", "<<=", ">>=", "**="])
    objtype = "pdarray"

    __array_priority__ = 1000

    def __init__(
        self,
        name: str,
        mydtype: np.dtype,
        size: int_scalars,
        ndim: int_scalars,
        shape: Sequence[int],
        itemsize: int_scalars,
    ) -> None:
        self.name = name
        self.dtype = dtype(mydtype)
        self.size = size
        self.ndim = ndim
        self.shape = shape
        self.itemsize = itemsize

    def __del__(self):
        try:
            logger.debug(f"deleting pdarray with name {self.name}")
            generic_msg(cmd="delete", args="{}".format(self.name))
        except RuntimeError:
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

        return generic_msg(cmd="str", args="{} {}".format(self.name, pdarrayIterThresh))

    def __repr__(self):
        from arkouda.client import pdarrayIterThresh

        return generic_msg(cmd="repr", args="{} {}".format(self.name, pdarrayIterThresh))

    def format_other(self, other: object) -> str:
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
            other = self.dtype.type(other)
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
        if type(other) != pdarray and issubclass(type(other), pdarray):
            return NotImplemented
        if op not in self.BinOps:
            raise ValueError(f"bad operator {op}")
        # pdarray binop pdarray
        if isinstance(other, pdarray):
            if self.size != other.size:
                raise ValueError(f"size mismatch {self.size} {other.size}")
            cmd = "binopvv"
            args = "{} {} {}".format(op, self.name, other.name)
            repMsg = generic_msg(cmd=cmd, args=args)
            return create_pdarray(repMsg)
        # pdarray binop scalar
        if np.can_cast(other, self.dtype):
            # If scalar can be losslessly cast to array dtype,
            # do the cast so that return array will have same dtype
            dt = self.dtype.name
            other = self.dtype.type(other)
        else:
            # If scalar cannot be safely cast, server will infer the return dtype
            dt = resolve_scalar_dtype(other)
        if dt not in DTypes:
            raise TypeError(f"Unhandled scalar type: {other} ({type(other)})")
        cmd = "binopvs"
        args = "{} {} {} {}".format(op, self.name, dt, NUMBER_FORMAT_STRINGS[dt].format(other))
        repMsg = generic_msg(cmd=cmd, args=args)
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
        if np.can_cast(other, self.dtype):
            # If scalar can be losslessly cast to array dtype,
            # do the cast so that return array will have same dtype
            dt = self.dtype.name
            other = self.dtype.type(other)
        else:
            # If scalar cannot be safely cast, server will infer the return dtype
            dt = resolve_scalar_dtype(other)
        if dt not in DTypes:
            raise TypeError(f"Unhandled scalar type: {other} ({type(other)})")
        cmd = "binopsv"
        args = "{} {} {} {}".format(op, dt, NUMBER_FORMAT_STRINGS[dt].format(other), self.name)
        repMsg = generic_msg(cmd=cmd, args=args)
        return create_pdarray(repMsg)

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
            generic_msg(cmd="opeqvv", args="{} {} {}".format(op, self.name, other.name))
            return self
        # pdarray binop scalar
        # opeq requires scalar to be cast as pdarray dtype
        try:
            other = self.dtype.type(other)
        except Exception:
            # Can't cast other as dtype of pdarray
            raise TypeError(f"Unhandled scalar type: {other} ({type(other)})")

        cmd = "opeqvs"
        args = "{} {} {} {}".format(op, self.name, self.dtype.name, self.format_other(other))
        generic_msg(cmd=cmd, args=args)
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
        if np.isscalar(key) and (resolve_scalar_dtype(key) == "int64" or "uint64"):
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if key >= 0 and key < self.size:
                repMsg = generic_msg(cmd="[int]", args="{} {}".format(self.name, key))
                fields = repMsg.split()
                # value = fields[2]
                return parse_single_value(" ".join(fields[1:]))
            else:
                raise IndexError(f"[int] {orig_key} is out of bounds with size {self.size}")
        if isinstance(key, slice):
            (start, stop, stride) = key.indices(self.size)
            logger.debug("start: {} stop: {} stride: {}".format(start, stop, stride))
            repMsg = generic_msg(
                cmd="[slice]", args="{} {} {} {}".format(self.name, start, stop, stride)
            )
            return create_pdarray(repMsg)
        if isinstance(key, pdarray):
            kind, _ = translate_np_dtype(key.dtype)
            if kind not in ("bool", "int", "uint"):
                raise TypeError(f"unsupported pdarray index type {key.dtype}")
            if kind == "bool" and self.size != key.size:
                raise ValueError(f"size mismatch {self.size} {key.size}")
            repMsg = generic_msg(cmd="[pdarray]", args="{} {}".format(self.name, key.name))
            return create_pdarray(repMsg)
        else:
            raise TypeError(f"Unhandled key type: {key} ({type(key)})")

    def __setitem__(self, key, value):
        if np.isscalar(key) and (resolve_scalar_dtype(key) == "int64" or "uint64"):
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if key >= 0 and key < self.size:
                generic_msg(
                    cmd="[int]=val",
                    args="{} {} {} {}".format(self.name, key, self.dtype.name, self.format_other(value)),
                )
            else:
                raise IndexError(f"index {orig_key} is out of bounds with size {self.size}")
        elif isinstance(key, pdarray):
            if isinstance(value, pdarray):
                generic_msg(
                    cmd="[pdarray]=pdarray", args="{} {} {}".format(self.name, key.name, value.name)
                )
            else:
                generic_msg(
                    cmd="[pdarray]=val",
                    args="{} {} {} {}".format(
                        self.name, key.name, self.dtype.name, self.format_other(value)
                    ),
                )
        elif isinstance(key, slice):
            (start, stop, stride) = key.indices(self.size)
            logger.debug(f"start: {start} stop: {stop} stride: {stride}")
            if isinstance(value, pdarray):
                generic_msg(
                    cmd="[slice]=pdarray",
                    args="{} {} {} {} {}".format(self.name, start, stop, stride, value.name),
                )
            else:
                generic_msg(
                    cmd="[slice]=val",
                    args="{} {} {} {} {} {}".format(
                        self.name, start, stop, stride, self.dtype.name, self.format_other(value)
                    ),
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
        generic_msg(
            cmd="set", args="{} {} {}".format(self.name, self.dtype.name, self.format_other(value))
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
        """
        return np.bool_(self.name in list_registry())

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

    def argmin(self) -> np.int64:
        """
        Return the index of the first occurrence of the array min value
        """
        return argmin(self)

    def argmax(self) -> np.int64:
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
        array

        Examples
        --------
        >>> a = ak.arange(0, 5, 1)
        >>> a.to_ndarray()
        array([0, 1, 2, 3, 4])

        >>> type(a.to_ndarray())
        numpy.ndarray
        """
        from arkouda.client import maxTransferBytes

        # Total number of bytes in the array data
        arraybytes = self.size * self.dtype.itemsize
        # Guard against overflowing client memory
        if arraybytes > maxTransferBytes:
            raise RuntimeError(
                "Array exceeds allowed size for transfer. Increase client.maxTransferBytes to allow"
            )
        # The reply from the server will be binary data
        data = cast(
            memoryview, generic_msg(cmd="tondarray", args="{}".format(self.name), recv_binary=True)
        )
        # Make sure the received data has the expected length
        if len(data) != self.size * self.dtype.itemsize:
            raise RuntimeError(
                f"Expected {self.size * self.dtype.itemsize} bytes but received {len(data)}"
            )
        # The server sends us native-endian data so we need to account for
        # that. If the view is readonly, copy so the np array is mutable
        dt = np.dtype(self.dtype)
        if get_server_byteorder() == "big":
            dt = dt.newbyteorder(">")
        else:
            dt = dt.newbyteorder("<")

        if data.readonly:
            return np.frombuffer(data, dt).copy()
        else:
            return np.frombuffer(data, dt)

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
    def save(
        self,
        prefix_path: str,
        dataset: str = "array",
        mode: str = "truncate",
        compressed: bool = False,
        file_format: str = "HDF5",
    ) -> str:
        """
        Save the pdarray to HDF5 or Parquet. The result is a collection of files,
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
        compressed : bool
            Defaults to False. When True, files will be written with Snappy compression
            and RLE bit packing. This is currently only supported on Parquet files and will
            not impact the generated files when writing HDF5 files.
        file_format : str {'HDF5', 'Parquet'}
            By default, saved files will be written to the HDF5 file format. If
            'Parquet', the files will be written to the Parquet file format. This
            is case insensitive.

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
        save_all, load, read

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
        if mode.lower() in ["a", "app", "append"]:
            m = 1
        elif mode.lower() in ["t", "trunc", "truncate"]:
            m = 0
        else:
            raise ValueError("Allowed modes are 'truncate' and 'append'")

        if file_format.lower() == "hdf5":
            cmd = "tohdf"
        elif file_format.lower() == "parquet":
            cmd = "writeParquet"
        else:
            raise ValueError("Supported file formats are 'HDF5' and 'Parquet'")

        """
        If offsets are provided, add to the json_array as the offsets will be used to
        retrieve the array elements from the hdf5 files.
        """
        try:
            json_array = json.dumps([prefix_path])
        except Exception as e:
            raise ValueError(e)
        strings_placeholder = False

        return cast(
            str,
            generic_msg(
                cmd=cmd,
                args="{} {} {} {} {} {} {}".format(
                    self.name, dataset, m, json_array, self.dtype, strings_placeholder, compressed
                ),
            ),
        )

    @typechecked
    def save_parquet(
        self, prefix_path: str, dataset: str = "array", mode: str = "truncate", compressed: bool = False
    ) -> str:
        """
        Save the pdarray to Parquet. The result is a collection of Parquet files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in Parquet files (must not already exist)
        mode : str {'truncate', 'append'}
            By default, truncate (overwrite) output files, if they exist.
        compressed : bool
            By default, write without Snappy compression and RLE encoding.

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
        save, save_all, load, read

        Notes
        -----
        The prefix_path must be visible to the arkouda server and the user must
        have write permission.

        Output files have names of the form ``<prefix_path>_LOCALE<i>.parquet``, where ``<i>``
        ranges from 0 to ``numLocales``. If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.

        Examples
        --------
        >>> a = ak.arange(0, 100, 1)
        >>> a.save_parquet('arkouda_range')

        Array is saved in numLocales files with names like ``arkouda_range_LOCALE0000.parquet``

        The array can be read back in as follows

        >>> b = ak.read('arkouda_range')
        >>> (a == b).all()
        True
        """
        return self.save(
            prefix_path=prefix_path,
            dataset=dataset,
            mode=mode,
            compressed=compressed,
            file_format="Parquet",
        )

    @typechecked
    def save_hdf(self, prefix_path: str, dataset: str = "array", mode: str = "truncate") -> str:
        """
        Save the pdarray to HDF5. The result is a collection of HDF5 files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in HDF5 files (must not already exist)
        mode : str {'truncate', 'append'}
            By default, truncate (overwrite) output files, if they exist.
        compressed : bool
            By default, write without Snappy compression and RLE encoding.

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
        save, save_all, load, read

        Notes
        -----
        The prefix_path must be visible to the arkouda server and the user must
        have write permission.

        Output files have names of the form ``<prefix_path>_LOCALE<i>``, where ``<i>``
        ranges from 0 to ``numLocales``. If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.

        Examples
        --------
        >>> a = ak.arange(0, 100, 1)
        >>> a.save_hdf('arkouda_range')

        Array is saved in numLocales files with names like ``arkouda_range_LOCALE0000``

        The array can be read back in as follows

        >>> b = ak.read('arkouda_range')
        >>> (a == b).all()
        True
        """
        return self.save(
            prefix_path=prefix_path, dataset=dataset, mode=mode, compressed=False, file_format="HDF5"
        )

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
        try:
            rep_msg = generic_msg(cmd="register", args=f"{self.name} {user_defined_name}")
            if isinstance(rep_msg, bytes):
                rep_msg = str(rep_msg, "UTF-8")
            if rep_msg != "success":
                raise RegistrationError
        except (
            RuntimeError,
            RegistrationError,
        ):  # Registering two objects with the same name is not allowed
            raise RegistrationError(f"Server was unable to register {user_defined_name}")

        self.name = user_defined_name
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
        unregister_pdarray_by_name(self.name)

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
        return attach_pdarray(user_defined_name)

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
        else:
            raise TypeError("Grouping is only supported on numeric data (integral types) and bools.")


# end pdarray class def

# creates pdarray object
#   only after:
#       all values have been checked by python module and...
#       server has created pdarray already before this is called
#       server has created pdarray already befroe this is called
@typechecked
def create_pdarray(repMsg: str) -> pdarray:
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
        shape = [int(el) for el in fields[5][1:-1].split(",")]
        itemsize = int(fields[6])
    except Exception as e:
        raise ValueError(e)
    logger.debug(
        f"created Chapel array with name: {name} dtype: {mydtype} size: {size} ndim: {ndim} "
        + f"shape: {shape} itemsize: {itemsize}"
    )
    return pdarray(name, dtype(mydtype), size, ndim, shape, itemsize)


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
    repMsg = generic_msg(cmd="reduction", args="{} {}".format("any", pda.name))
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
    repMsg = generic_msg(cmd="reduction", args="{} {}".format("all", pda.name))
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
    repMsg = generic_msg(cmd="reduction", args="{} {}".format("is_sorted", pda.name))
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
    repMsg = generic_msg(cmd="reduction", args="{} {}".format("sum", pda.name))
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
    repMsg = generic_msg(cmd="reduction", args="{} {}".format("prod", pda.name))

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
    repMsg = generic_msg(cmd="reduction", args="{} {}".format("min", pda.name))
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
    repMsg = generic_msg(cmd="reduction", args="{} {}".format("max", pda.name))
    return parse_single_value(cast(str, repMsg))


@typechecked
def argmin(pda: pdarray) -> np.int64:
    """
    Return the index of the first occurrence of the array min value.

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the argmin

    Returns
    -------
    np.int64
        The index of the argmin calculated from the pda

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args="{} {}".format("argmin", pda.name))
    return parse_single_value(cast(str, repMsg))


@typechecked
def argmax(pda: pdarray) -> np.int64:
    """
    Return the index of the first occurrence of the array max value.

    Parameters
    ----------
    pda : pdarray
        Values for which to calculate the argmax

    Returns
    -------
    np.int64
        The index of the argmax calculated from the pda

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray instance
    RuntimeError
        Raised if there's a server-side error thrown
    """
    repMsg = generic_msg(cmd="reduction", args="{} {}".format("argmax", pda.name))
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
    return np.float64(pda.sum()) / pda.size


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
    m = mean(pda)
    return ((pda - m) ** 2).sum() / (pda.size - ddof)


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

    return np.sqrt(var(pda, ddof=ddof))


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

    repMsg = generic_msg(cmd="mink", args="{} {} {}".format(pda.name, k, False))
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

    repMsg = generic_msg(cmd="maxk", args="{} {} {}".format(pda.name, k, False))
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

    repMsg = generic_msg(cmd="mink", args="{} {} {}".format(pda.name, k, True))
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

    repMsg = generic_msg(cmd="maxk", args="{} {} {}".format(pda.name, k, True))
    return create_pdarray(repMsg)


def popcount(pda: pdarray) -> pdarray:
    """
    Find the population (number of bits set) for each integer in an array.

    Parameters
    ----------
    pda : pdarray, int64, uint64
        Input array (must be integral).

    Returns
    -------
    population : pdarray
        The number of bits set (1) in each element

    Raises
    ------
    TypeError
        If input array is not int64 or uint64

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.popcount(A)
    array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2])
    """
    if pda.dtype != akint64 and pda.dtype != akuint64:
        raise TypeError("BitOps only supported on int64 and uint64 arrays")
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("popcount", pda.name))
    return create_pdarray(repMsg)


def parity(pda: pdarray) -> pdarray:
    """
    Find the bit parity (XOR of all bits) for each integer in an array.

    Parameters
    ----------
    pda : pdarray, int64, uint64
        Input array (must be integral).

    Returns
    -------
    parity : pdarray
        The parity of each element: 0 if even number of bits set, 1 if odd.

    Raises
    ------
    TypeError
        If input array is not int64 or uint64

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.parity(A)
    array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    """
    if pda.dtype != akint64 and pda.dtype != akuint64:
        raise TypeError("BitOps only supported on int64 and uint64 arrays")
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("parity", pda.name))
    return create_pdarray(repMsg)


def clz(pda: pdarray) -> pdarray:
    """
    Count leading zeros for each integer in an array.

    Parameters
    ----------
    pda : pdarray, int64, uint64
        Input array (must be integral).

    Returns
    -------
    lz : pdarray
        The number of leading zeros of each element.

    Raises
    ------
    TypeError
        If input array is not int64 or uint64

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.clz(A)
    array([64, 63, 62, 62, 61, 61, 61, 61, 60, 60])
    """
    if pda.dtype != akint64 and pda.dtype != akuint64:
        raise TypeError("BitOps only supported on int64 and uint64 arrays")
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("clz", pda.name))
    return create_pdarray(repMsg)


def ctz(pda: pdarray) -> pdarray:
    """
    Count trailing zeros for each integer in an array.

    Parameters
    ----------
    pda : pdarray, int64, uint64
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
        If input array is not int64 or uint64

    Examples
    --------
    >>> A = ak.arange(10)
    >>> ak.ctz(A)
    array([0, 0, 1, 0, 2, 0, 1, 0, 3, 0])
    """
    if pda.dtype != akint64 and pda.dtype != akuint64:
        raise TypeError("BitOps only supported on int64 and uint64 arrays")
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("ctz", pda.name))
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
    if isinstance(x, pdarray) and (x.dtype == akint64 or x.dtype == akuint64):
        if (
            isinstance(rot, pdarray) and (rot.dtype == akint64 or rot.dtype == akuint64)
        ) or isSupportedInt(rot):
            return x._binop(rot, "<<<")
        else:
            raise TypeError("Rotations only supported on integers")
    elif (
        isSupportedInt(x)
        and isinstance(rot, pdarray)
        and (rot.dtype == akint64 or rot.dtype == akuint64)
    ):
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
    if isinstance(x, pdarray) and (x.dtype == akint64 or x.dtype == akuint64):
        if (
            isinstance(rot, pdarray) and (rot.dtype == akint64 or rot.dtype == akuint64)
        ) or isSupportedInt(rot):
            return x._binop(rot, ">>>")
        else:
            raise TypeError("Rotations only supported on integers")
    elif (
        isSupportedInt(x)
        and isinstance(rot, pdarray)
        and (rot.dtype == akint64 or rot.dtype == akuint64)
    ):
        return rot._r_binop(x, ">>>")
    else:
        raise TypeError("Rotations only supported on integers")


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
    >>> b = ak.attach_pdarray("my_zeros")
    >>> # ...other work...
    >>> b.unregister()
    """
    repMsg = generic_msg(cmd="attach", args="{}".format(user_defined_name))
    return create_pdarray(repMsg)


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
    generic_msg(cmd="unregister", args=user_defined_name)


# TODO In the future move this to a specific errors file
class RegistrationError(Exception):
    """Error/Exception used when the Arkouda Server cannot register an object"""
