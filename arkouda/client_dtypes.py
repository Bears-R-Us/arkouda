"""
Arkouda client-defined dtypes and helper utilities for structured or specialized array semantics.

This module introduces specialized subclasses of `pdarray` for handling and representing
data with specific interpretations or domain semantics. These include:

- `BitVector`: For representing integers as sets of binary flags.
- `Fields`: For displaying and interacting with named binary flags.
- `IPv4`: For storing and displaying 32-bit integers as IPv4 addresses.

These classes enhance usability and improve readability when working with encoded or
domain-specific data while preserving Arkouda’s performance model and distributed data structures.

Functions
---------
- `bit_vectorizer`: Creates a partially applied BitVector constructor.
- `ip_address`: Converts various formats to an Arkouda IPv4 object.
- `is_ipv4`: Returns a boolean array indicating IPv4 addresses.
- `is_ipv6`: Returns a boolean array indicating IPv6 addresses.

Examples
--------
>>> import arkouda as ak
>>> from arkouda.client_dtypes import BitVector, Fields, IPv4, ip_address, is_ipv4

Create and use BitVectors:
>>> a = ak.array([3, 5, 7])
>>> bv = BitVector(a, width=4)
>>> print(bv)
BitVector([..||,
           .|.|,
           .|||],
          width=4, reverse=False)

Create Fields with named binary flags:
>>> f = Fields(ak.array([1, 2, 3]), names=['read', 'write', 'exec'], separator=':')
>>> print(f[0])  # doctest: +SKIP
--:--:read (1)

Convert and work with IP addresses:
>>> ips = ip_address(['192.168.0.1', '10.0.0.1'])
>>> print(ips)
IPv4([192.168.0.1,
      10.0.0.1],
     )

>>> is_ipv4(ips)
array([True True])

"""

import warnings

from functools import partial
from ipaddress import ip_address as _ip_address
from typing import TYPE_CHECKING, Literal, Optional, TypeVar, Union

import numpy as np

from typeguard import typechecked

from arkouda.numpy.dtypes import bitType, intTypes, is_supported_int
from arkouda.numpy.dtypes import uint64 as akuint64
from arkouda.numpy.pdarrayclass import RegistrationError, pdarray
from arkouda.pandas.groupbyclass import GroupBy, broadcast


if TYPE_CHECKING:
    from arkouda.numpy import cast as akcast
    from arkouda.numpy import where as akwhere
    from arkouda.numpy.pdarraycreation import arange, zeros
    from arkouda.numpy.strings import Strings
else:
    Strings = TypeVar("Strings")

__all__ = [
    "BitVector",
    "bit_vectorizer",
    "Fields",
    "IPv4",
    "ip_address",
    "is_ipv4",
    "is_ipv6",
]


def bit_vectorizer(width=64, reverse=False):
    """
    Make a callback (i.e. function) that can be called on an array to create a BitVector.

    Parameters
    ----------
    width : int
        The number of bit fields in the vector
    reverse : bool
        If True, display bits from least significant (left) to most
        significant (right). By default, the most significant bit
        is the left-most bit.

    Returns
    -------
    bitvectorizer : callable
        A function that takes an array and returns a BitVector instance

    """
    return partial(BitVector, width=width, reverse=reverse)


def BitVectorizer(*args, **kwargs):
    """
    Deprecated alias for :func:`bit_vectorizer`.

    This function exists for backward compatibility only. Use
    :func:`bit_vectorizer` instead.

    Parameters
    ----------
    *args : tuple
        Positional arguments forwarded to :func:`bit_vectorizer`.
    **kwargs : dict
        Keyword arguments forwarded to :func:`bit_vectorizer`.

    Returns
    -------
    bitvectorizer : callable
        A function that takes an array and returns a BitVector instance.

    See Also
    --------
    bit_vectorizer : Preferred replacement.
    """
    warnings.warn(
        "BitVectorizer is deprecated; use bit_vectorizer",
        DeprecationWarning,
        stacklevel=2,
    )
    return bit_vectorizer(*args, **kwargs)


class BitVector(pdarray):
    """
    Represent integers as bit vectors, e.g. a set of flags.

    Parameters
    ----------
    values : pdarray, int64
        The integers to represent as bit vectors
    width : int
        The number of bit fields in the vector
    reverse : bool
        If True, display bits from least significant (left) to most
        significant (right). By default, the most significant bit
        is the left-most bit.

    Returns
    -------
    bitvectors : BitVector
        The array of binary vectors

    Notes
    -----
    This class is a thin wrapper around pdarray that mostly affects
    how values are displayed to the user. Operators and methods will
    typically treat this class like a uint64 pdarray.

    """

    conserves = frozenset(("+", "-", "|", "&", "^", ">>", "<<"))
    special_objType = "BitVector"

    def __init__(self, values, width=64, reverse=False):
        from arkouda.numpy import cast as akcast

        self.registered_name = None
        if not isinstance(values, pdarray) or values.dtype not in intTypes:
            self.name = None  # This is needed to silence warnings of missing name during failed creation
            raise TypeError("Argument must be integer pdarray")
        self.width = width
        self.reverse = reverse
        # If values is already bitType, this will make a copy
        # A copy should be made in order to get a new server-side name,
        # to avoid undefinedSymbol errors if/when values gets deleted
        self.values = akcast(values, bitType)
        super().__init__(
            self.values.name,
            self.values.dtype.name,
            self.values.size,
            self.values.ndim,
            self.values.shape,
            self.values.itemsize,
        )

    def format(self, x):
        """Format a single binary vector as a string."""
        # Start with a fixed-width, zero-padded binary value,
        # and replace 0/1 with ./| for better visibility
        fmt = "{{:0{}b}}".format(self.width).format(x).replace("0", ".").replace("1", "|")
        if self.reverse:
            return fmt[::-1]
        else:
            return fmt

    def __str__(self):
        """
        Return a string representation of the BitVector.

        Returns
        -------
        str
            Human-readable string showing formatted bit vectors.

        """
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            vals = [self.format(self.values[i]) for i in range(self.size)]
        else:
            vals = [self.format(self.values[i]) for i in range(3)]
            vals.append("...")
            vals.extend([self.format(self.values[i]) for i in range(self.size - 3, self.size)])
        # Print values as a single, aligned column for easier viewing
        # Also show "width" and "reverse" parameters
        spaces = " " * (len(self.__class__.__name__) + 1)
        return "{}([{}],\n{}width={}, reverse={})".format(
            self.__class__.__name__, ",\n{} ".format(spaces).join(vals), spaces, self.width, self.reverse
        )

    def __repr__(self):
        """
        Return the official string representation of the BitVector.

        Returns
        -------
        str
            Same as __str__.

        """
        return self.__str__()

    def to_ndarray(self):
        """Export data to a numpy array of string-formatted bit vectors."""
        return np.array([self.format(x) for x in self.values.to_ndarray()])

    def tolist(self):
        """Export data to a list of string-formatted bit vectors."""
        return self.to_ndarray().tolist()

    def _cast(self, values):
        return self.__class__(values, width=self.width, reverse=self.reverse)

    def __getitem__(self, key):
        """
        Retrieve a formatted bit vector or a subarray.

        Parameters
        ----------
        key : int or slice
            Index or range to access.

        Returns
        -------
        str or BitVector
            A single formatted string or a BitVector slice.

        """
        if is_supported_int(key):
            # Return single value as a formatted string
            return self.format(self.values[key])
        else:
            # Forward all other indexing ops to integer array
            return self._cast(self.values[key])

    def __setitem__(self, key, value):
        """
        Set value(s) in the BitVector.

        Parameters
        ----------
        key : int or slice
            Index or range to set.
        value : int or BitVector
            New value(s) to assign.

        """
        if isinstance(value, self.__class__):
            # Set a slice or selection of values directly
            self.values[key] = value.values
        elif is_supported_int(key) and is_supported_int(value):
            # Set a single value
            self.values[key] = value
        else:
            return NotImplemented

    def _binop(self, other, op):
        # Based on other type, select which data to pass
        # to _binop of pdarray
        if is_supported_int(other):
            otherdata = other
        elif isinstance(other, self.__class__):
            otherdata = other.values
        elif isinstance(other, pdarray) and other.dtype in intTypes:
            otherdata = other
        else:
            return NotImplemented
        # Some operators return a BitVector, but others don't
        if op in self.conserves:
            # These ops are defined as a class attribute
            return self._cast(self.values._binop(otherdata, op))
        else:
            return self.values._binop(otherdata, op)

    def _r_binop(self, other, op):
        # Cases where left operand is BitVector are handled above by _binop
        # Only two cases to handle here: scalar int and int64 pdarray
        if is_supported_int(other):
            if op in self.conserves:
                return self._cast(self.values._r_binop(other, op))
            else:
                return self.values._r_binop(other, op)
        elif isinstance(other, pdarray) and other.dtype in intTypes:
            # Some operators return a BitVector, but others don't
            if op in self.conserves:
                return self._cast(other._binop(self.values, op))
            else:
                return other._binop(self.values, op)
        else:
            return NotImplemented

    def opeq(self, other, op):
        """
        In-place binary operation on the BitVector.

        Parameters
        ----------
        other : int, BitVector, or pdarray
            Right-hand operand.
        op : str
            Binary operator as a string (e.g., '+', '&').

        """
        # Based on other type, select data to pass to pdarray opeq
        if is_supported_int(other):
            otherdata = other
        elif isinstance(other, self.__class__):
            otherdata = other.values
        elif isinstance(other, pdarray) and other.dtype in intTypes:
            otherdata = other
        else:
            return NotImplemented
        self.values.opeq(otherdata, op)

    def register(self, user_defined_name):
        """
        Register this BitVector object and underlying components with the Arkouda server.

        Parameters
        ----------
        user_defined_name : str
            user defined name the BitVector is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        BitVector
            The same BitVector which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different BitVectors with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the BitVector with the user_defined_name

        See Also
        --------
        unregister, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        """
        from arkouda.client import generic_msg

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")
        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.special_objType,
                "values": self.values.name,
                "width": self.width,
                "reverse": self.reverse,
            },
        )
        self.registered_name = user_defined_name
        return self

    @classmethod
    def from_return_msg(cls, rep_msg):
        """
        Reconstruct a BitVector from a server return message.

        Parameters
        ----------
        rep_msg : str
            JSON-encoded response from Arkouda server.

        Returns
        -------
        BitVector
            Reconstructed BitVector instance.

        """
        import json

        from arkouda.numpy.pdarrayclass import create_pdarray

        data = json.loads(rep_msg)

        return cls(
            create_pdarray(data["values"]),
            width=data["width"],
            reverse=json.loads(data["reverse"].lower()),
        )


class Fields(BitVector):
    """
    An integer-backed representation of a set of named binary fields, e.g. flags.

    Parameters
    ----------
    values : pdarray or Strings
        The array of field values. If (u)int64, the values are used as-is for the
        binary representation of fields. If Strings, the values are converted
        to binary according to the mapping defined by the names and MSB_left
        arguments.
    names : str or sequence of str
        The names of the fields, in order. A string will be treated as a list
        of single-character field names. Multi-character field names are allowed,
        but must be passed as a list or tuple and user must specify a separator.
    MSB_left : bool
        Controls how field names are mapped to binary values. If True (default),
        the left-most field name corresponds to the most significant bit in the
        binary representation. If False, the left-most field name corresponds to
        the least significant bit.
    pad : str
        Character to display when field is not present. Use empty string if no
        padding is desired.
    separator : str
        Substring that separates fields. Used to parse input values (if ak.Strings)
        and to display output.
    show_int : bool
        If True (default), display the integer value of the binary fields in output.

    Returns
    -------
    fields : Fields
        The array of field values

    Notes
    -----
    This class is a thin wrapper around pdarray that mostly affects
    how values are displayed to the user. Operators and methods will
    typically treat this class like an int64 pdarray.

    """

    def __init__(self, values, names, MSB_left=True, pad="-", separator="", show_int=True):
        from arkouda.numpy.strings import Strings

        # Argument validation
        # Normalize names, which can be string or sequence
        self.names = tuple(names)
        self.name = None  # This is needed to silence warnings of missing name during failed creation
        if len(self.names) > 63:
            raise ValueError("Cannot represent more than 63 fields")
        # Ensure no duplicate names
        if len(self.names) != len(set(self.names)):
            raise ValueError("Field names must be unique")
        # Ensure no empty names
        if any(name == "" for name in self.names):
            raise ValueError("Names cannot be empty strings")
        # If separator is non-empty, it cannot be a field name
        if (separator != "") and (separator in self.names):
            raise ValueError("Separator cannot be a field name")
        # Field names can have multiple characters, but if so, there must be a separator
        self.namewidth = max(len(n) for n in self.names)
        if (self.namewidth > 1) and (separator == ""):
            raise ValueError(
                "A non-empty separator must be specified when field names have more than one character"
            )
        if len(pad) > 1:
            raise ValueError("Pad must be single character or empty string")

        self.padchar = pad
        # string to display when a field is not set
        self.pad = self.namewidth * self.padchar
        # separator to display between fields
        self.separator = separator
        width = len(self.names)
        # whether the left-most field name corresponds to the most significant bit
        self.MSB_left = MSB_left
        if self.MSB_left:
            self.shifts = range(width - 1, -1, -1)
        else:
            self.shifts = range(width)
        # whether to display the integer value of the field bit vector
        self.show_int = show_int
        # values arg can be either int64 or Strings. If Strings, convert to binary
        if isinstance(values, Strings):
            values = self._convert_strings(values)
        # super init will type-check values and make copy
        super().__init__(values, width=width, reverse=not MSB_left)

    def _convert_strings(self, s):
        """Convert string field names to binary vectors."""
        # Initialize to zero
        values = zeros(s.size, dtype=bitType)
        if self.separator == "":
            # When separator is empty, field names are guaranteed to be single characters
            for name, shift in zip(self.names, self.shifts):
                # Check if name exists in each string
                bit = s.contains(name)
                values = values | akcast(akwhere(bit, 1 << shift, 0), bitType)
        else:
            # When separator is non-empty, split on it
            sf, segs = s.flatten(self.separator, return_segments=True)
            # Create a grouping to map split fields back to originating string
            orig = broadcast(segs, arange(segs.size), sf.size)
            g = GroupBy(orig)
            for name, shift in zip(self.names, self.shifts):
                # Check if name matches one of the split fields from originating string
                bit = g.any(sf == name)[1]
                values = values | akcast(akwhere(bit, 1 << shift, 0), bitType)
        return values

    def _parse_scalar(self, s):
        """Convert a string of named fields to a binary value."""
        val = 0
        if self.separator == "":
            # Arg validation guarantees single-character field names if here
            for name, shift in zip(self.names, self.shifts):
                if name in s:
                    val |= 1 << shift
        else:
            fields = s.split(self.separator)
            for name, shift in zip(self.names, self.shifts):
                if name in fields:
                    val |= 1 << shift
        return val

    def format(self, x):
        """Format a single binary value as a string of named fields."""
        # Start with a fixed-width, zero-padded binary value,
        # and replace 0/1 with ./| for better visibility
        s = ""
        for i, shift in enumerate(self.shifts):
            bitset = ((int(x) >> shift) & 1) == 1
            if bitset:
                s += "{{:^{}s}}".format(self.namewidth).format(self.names[i])
            else:
                s += self.pad
            s += self.separator
        if self.show_int:
            s += " ({:n})".format(x)
        return s

    def __setitem__(self, key, value):
        """
        Set value(s) in the Fields object.

        Parameters
        ----------
        key : int or slice
            Index or range to set.
        value : str or int
            Value(s) to assign, which may be parsed field strings.

        """
        if isinstance(value, str):
            v = self._parse_scalar(value)
            return super().__setitem__(key, v)
        else:
            return super().__setitem__(key, value)

    def _cast(self, values):
        return self.__class__(
            values,
            self.names,
            MSB_left=self.MSB_left,
            pad=self.padchar,
            separator=self.separator,
            show_int=self.show_int,
        )

    def _binop(self, other, op):
        if isinstance(other, str):
            o = self._parse_scalar(other)
            return super()._binop(o, op)
        else:
            return super()._binop(other, op)

    def _r_binop(self, other, op):
        if isinstance(other, str):
            o = self._parse_scalar(other)
            return super()._r_binop(o, op)
        else:
            return super()._r_binop(other, op)

    def opeq(self, other, op):
        """
        Perform in-place binary operation on the Fields object.

        Parameters
        ----------
        other : str or BitVector
            Operand to apply the binary operation with.
        op : str
            Binary operation to apply (e.g., '|', '&').

        """
        if isinstance(other, str):
            o = self._parse_scalar(other)
            return super().opeq(o, op)
        else:
            return super().opeq(other, op)


def ip_address(values):
    """
    Convert values to an Arkouda array of IP addresses.

    Parameters
    ----------
    values : list-like, integer pdarray, or IPv4
        The integer IP addresses or IPv4 object.

    Returns
    -------
    IPv4
        The same IP addresses as an Arkouda array

    Notes
    -----
    This helper is intended to help future proof changes made to
    accomodate IPv6 and to prevent errors if a user inadvertently
    casts a IPv4 instead of a int64 pdarray. It can also be used
    for importing Python lists of IP addresses into Arkouda.

    """
    from arkouda.numpy.pdarraycreation import array
    from arkouda.numpy.strings import Strings

    if isinstance(values, IPv4):
        return values

    if isinstance(values, pdarray):
        return IPv4(values)

    if isinstance(values, Strings):
        raise NotImplementedError("Strings to IP address not yet implemented")

    # Assume values is a python sequence of IP addresses in some format
    try:
        return IPv4(array([int(_ip_address(x)) for x in values]))
    except Exception as e:
        raise RuntimeError("Error converting non-arkouda object to list of IP addresses") from e


class IPv4(pdarray):
    """
    Represent integers as IPv4 addresses.

    Parameters
    ----------
    values : pdarray, int64
        The integer IP addresses

    Returns
    -------
    IPv4
        The same IP addresses

    Notes
    -----
    This class is a thin wrapper around pdarray that mostly affects
    how values are displayed to the user. Operators and methods will
    typically treat this class like an int64 pdarray.

    """

    special_objType = "IPv4"

    def __init__(self, values):
        from arkouda.numpy import cast as akcast

        if not isinstance(values, pdarray) or values.dtype not in intTypes:
            self.name = None  # This is needed to silence warnings of missing name during failed creation
            raise TypeError("Argument must be int64 pdarray")
        # Casting always creates a copy with new server-side name,
        # which will avoid unknown symbol errors
        self.values = akcast(values, bitType)
        super().__init__(
            self.values.name,
            self.values.dtype.name,
            self.values.size,
            self.values.ndim,
            self.values.shape,
            self.values.itemsize,
        )

    def export_uint(self):
        """
        Export the internal values as unsigned 64-bit integers.

        Returns
        -------
        pdarray
            Array of uint64 values representing the data.

        """
        from arkouda.numpy import cast as akcast

        return akcast(self.values, akuint64)

    def format(self, x):
        """Format a single integer IP address as a string."""
        if not is_supported_int(x):
            raise TypeError("Argument must be an integer scalar")
        return str(_ip_address(int(x)))

    def normalize(self, x):
        """
        Normalize IP adress.

        Take in an IP address as a string, integer, or IPAddress object,
        and convert it to an integer.

        """
        if not is_supported_int(x):
            x = int(_ip_address(x))
        if x < 0:
            raise ValueError(f"Not an IP address: {_ip_address(x)}")
        return x

    def _is_supported_scalar(self, x):
        try:
            return True, self.normalize(x)
        except ValueError:
            return False, None

    def __str__(self):
        """
        Return a string representation of the IPv4 object.

        Returns
        -------
        str
            Human-readable string showing IP addresses.

        """
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            vals = [self.format(self.values[i]) for i in range(self.size)]
        else:
            vals = [self.format(self.values[i]) for i in range(3)]
            vals.append("...")
            vals.extend([self.format(self.values[i]) for i in range(self.size - 3, self.size)])
        # Display values as single, aligned column for ease of viewing
        spaces = " " * (len(self.__class__.__name__) + 1)
        return "{}([{}],\n{})".format(
            self.__class__.__name__, ",\n{} ".format(spaces).join(vals), spaces
        )

    def __repr__(self):
        """
        Return the official string representation of the IPv4 object.

        Returns
        -------
        str
            Same as __str__.

        """
        return self.__str__()

    def to_ndarray(self):
        """Export array as a numpy array of integers."""
        return np.array([self.format(x) for x in self.values.to_ndarray()])

    def tolist(self):
        """Export array as a list of integers."""
        return self.to_ndarray().tolist()

    def __getitem__(self, key):
        """
        Retrieve an IP address or a slice.

        Parameters
        ----------
        key : int or slice
            Index to retrieve.

        Returns
        -------
        str or IPv4
            Formatted IP address string or a sliced IPv4 instance.

        """
        if is_supported_int(key):
            # Display single item as string
            return self.format(self.values[key])
        else:
            # Forward other accesses to pdarray class
            return self.__class__(self.values[key])

    def __setitem__(self, key, value):
        """
        Set IP address value(s).

        Parameters
        ----------
        key : int or slice
            Index or range to set.
        value : str or IPv4
            IP address string or IPv4 instance.

        """
        # If scalar, convert to integer and set
        isscalar, scalarval = self._is_supported_scalar(value)
        if isscalar:
            self.values[key] = scalarval
        elif isinstance(value, self.__class__):
            self.values[key] = value.values
        else:
            return NotImplemented

    def _binop(self, other, op):
        # Based on other type, select data to pass to pdarray _binop
        isscalar, scalarval = self._is_supported_scalar(other)
        if isscalar:
            otherdata = scalarval
        elif isinstance(other, self.__class__):
            otherdata = other.values
        elif isinstance(other, pdarray) and other.dtype in intTypes:
            otherdata = other
        else:
            return NotImplemented
        return self.values._binop(otherdata, op)

    def _r_binop(self, other, op):
        # _binop handles everything except left types: scalar and pdarray int64
        isscalar, scalarval = self._is_supported_scalar(other)
        if isscalar:
            return self.values._r_binop(scalarval, op)
        elif isinstance(other, pdarray) and other.dtype in intTypes:
            return other._binop(self.values, op)
        else:
            return NotImplemented

    def opeq(self, other, op):
        """
        Perform an in-place binary operation on the IPv4 object.

        Parameters
        ----------
        other : int, str, pdarray, or IPv4
            Operand for the operation.
        op : str
            Binary operator as a string.

        """
        # Based on other type, select data to pass to pdarray opeq
        isscalar, scalarval = self._is_supported_scalar(other)
        if isscalar:
            otherdata = scalarval
        elif isinstance(other, self.__class__):
            otherdata = other.values
        elif isinstance(other, pdarray) and other.dtype in intTypes:
            otherdata = other
        else:
            return NotImplemented
        self.values.opeq(otherdata, op)

    def register(self, user_defined_name):
        """
        Register this IPv4 object and underlying components with the Arkouda server.

        Parameters
        ----------
        user_defined_name : str
            user defined name the IPv4 is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        IPv4
            The same IPv4 which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different IPv4s with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the IPv4 with the user_defined_name

        See Also
        --------
        unregister, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        """
        from arkouda.client import generic_msg

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")
        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.special_objType,
                "array": self.values,
            },
        )
        self.registered_name = user_defined_name
        return self

    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "array",
        mode: Literal["truncate", "append"] = "truncate",
        file_type: Literal["single", "distribute"] = "distribute",
    ):
        """Override of the pdarray to_hdf to store the special object type."""
        from arkouda.client import generic_msg
        from arkouda.pandas.io import _file_type_to_int, _mode_str_to_int

        return (
            generic_msg(
                cmd="tohdf",
                args={
                    "values": self,
                    "dset": dataset,
                    "write_mode": _mode_str_to_int(mode),
                    "filename": prefix_path,
                    "dtype": self.dtype,
                    "objType": self.special_objType,
                    "file_format": _file_type_to_int(file_type),
                },
            ),
        )

    def update_hdf(self, prefix_path: str, dataset: str = "array", repack: bool = True):
        """Override the pdarray implementation so that the special object type will be used."""
        from arkouda.client import generic_msg
        from arkouda.pandas.io import (
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
                "objType": self.special_objType,
                "file_format": _file_type_to_int(file_type),
                "overwrite": True,
            },
        )

        if repack:
            _repack_hdf(prefix_path)


@typechecked
def is_ipv4(ip: Union[pdarray, IPv4], ip2: Optional[pdarray] = None) -> pdarray:
    """
    Indicate which values are ipv4 when passed data containing IPv4 and IPv6 values.

    Parameters
    ----------
    ip: pdarray (int64) or ak.IPv4
        IPv4 value. High Bits of IPv6 if IPv6 is passed in.
    ip2: pdarray (int64), Optional
        Low Bits of IPv6. This is added for support when dealing with data that contains IPv6 as well.

    Returns
    -------
    pdarray of bools indicating which indexes are IPv4.

    See Also
    --------
    ak.is_ipv6

    """
    # grab the ipv4 pdarray of values
    if isinstance(ip, IPv4):
        ip = ip.values

    if ip.dtype not in intTypes or (ip2 is not None and ip2.dtype not in intTypes):
        raise TypeError("ip and ip2 must be int64 pdarrays. ip2 is Optional.")

    if ip2 is not None and ip.size != ip2.size:
        raise RuntimeError("When supplying a value for ip2, ip and ip2 must be the same size.")

    if ip2 is not None:
        ans = ip < 2**32
        ans2 = ip2 == 0
        return ans & ans2
    else:
        return ip < 2**32


@typechecked
def is_ipv6(ip: Union[pdarray, IPv4], ip2: Optional[pdarray] = None) -> pdarray:
    """
    Indicate which values are ipv6 when passed data containing IPv4 and IPv6 values.

    Parameters
    ----------
    ip: pdarray (int64) or ak.IPv4
        High Bits of IPv6.
    ip2: pdarray (int64), Optional
        Low Bits of IPv6

    Returns
    -------
    pdarray of bools indicating which indexes are IPv6.

    See Also
    --------
    ak.is_ipv4

    """
    # grab the ipv4 pdarray of values
    if isinstance(ip, IPv4):
        ip = ip.values
    if ip.dtype not in intTypes or (ip2 is not None and ip2.dtype not in intTypes):
        raise TypeError("ip and ip2 must be int64 pdarrays. ip2 is Optional.")
    if ip2 is not None and ip.size != ip2.size:
        raise RuntimeError("When supplying a value for ip2, ip and ip2 must be the same size.")

    if ip2 is not None:
        ans = ip >= 2**32
        ans2 = ip2 != 0
        return ans | ans2
    else:
        return ip >= 2**32
