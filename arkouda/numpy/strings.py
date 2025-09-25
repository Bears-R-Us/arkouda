from __future__ import annotations

import codecs
import itertools
import re
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from numpy import dtype as npdtype
from typeguard import typechecked

from arkouda.infoclass import information, list_symbol_table
from arkouda.logger import ArkoudaLogger, getArkoudaLogger
import arkouda.numpy.dtypes
from arkouda.numpy.dtypes import NUMBER_FORMAT_STRINGS, bool_scalars
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import int_scalars, resolve_scalar_dtype, str_scalars
from arkouda.numpy.pdarrayclass import RegistrationError
from arkouda.numpy.pdarrayclass import all as akall
from arkouda.numpy.pdarrayclass import create_pdarray, parse_single_value, pdarray
from arkouda.pandas.match import Match, MatchType


if TYPE_CHECKING:
    from arkouda.client import generic_msg
else:
    generic_msg = TypeVar("generic_msg")

if TYPE_CHECKING:
    from arkouda.numpy.sorting import SortingAlgorithm
else:
    from enum import Enum

    class SortingAlgorithm(Enum):
        RadixSortLSD = "RadixSortLSD"


__all__ = ["Strings"]


# Command strings for message passing to arkouda server, specific to Strings
CMD_ASSEMBLE = "segStr-assemble"
CMD_TO_NDARRAY = "segStr-tondarray"


class Strings:
    """
    Represents an array of strings whose data resides on the
    arkouda server. The user should not call this class directly;
    rather its instances are created by other arkouda functions.

    Attributes
    ----------
    entry : pdarray
        Encapsulation of a Segmented Strings array contained on
        the arkouda server.  This is a composite of
         - offsets array: starting indices for each string
         - bytes array: raw bytes of all strings joined by nulls
    size : int_scalars
        The number of strings in the array
    nbytes : int_scalars
        The total number of bytes in all strings
    ndim : int_scalars
        The rank of the array (currently only rank 1 arrays supported)
    shape : tuple
        The sizes of each dimension of the array
    dtype : type
        The dtype is ak.str_
    logger : ArkoudaLogger
        Used for all logging operations

    Notes
    -----
    Strings is composed of two pdarrays: (1) offsets, which contains the
    starting indices for each string and (2) bytes, which contains the
    raw bytes of all strings, delimited by nulls.
    """

    entry: pdarray
    size: int_scalars
    nbytes: int_scalars
    ndim: int_scalars
    shape: Tuple[int]
    logger: ArkoudaLogger

    BinOps = frozenset(["==", "!="])
    objType = "Strings"

    @staticmethod
    def from_return_msg(rep_msg: str) -> Strings:
        """
        Create a Strings object from an Arkouda server response message.

        Parse the server’s response descriptor and construct a `Strings` array
        with its underlying pdarray and total byte size.

        Parameters
        ----------
        rep_msg : str
            Server response message of the form:
            ```
            created <name> <type> <size> <ndim> <shape> <itemsize>+... bytes.size <total_bytes>
            ```
            For example:
            ```
            "created foo Strings 3 1 (3,) 8+created bytes.size 24"
            ```

        Returns
        -------
        Strings
            A `Strings` object representing the segmented strings array on the server,
            initialized with the returned pdarray and byte-size metadata.

        Raises
        ------
        RuntimeError
            If the response message cannot be parsed or does not match the expected format.

        Examples
        --------
        >>> import arkouda as ak

        # Example response message (typically from `generic_msg`)
        >>> rep_msg = "created foo Strings 3 1 (3,) 8+created bytes.size 24"
        >>> s = ak.Strings.from_return_msg(rep_msg)
        >>> isinstance(s, ak.Strings)
        True
        """
        left, right = cast(str, rep_msg).split("+")
        try:
            bytes_size = int(right.split()[-1])
        except Exception as e:
            raise RuntimeError(f"Cannot parse byte size from response: {rep_msg}") from e
        return Strings(create_pdarray(left), bytes_size)

    @staticmethod
    def from_parts(offset_attrib: Union[pdarray, str], bytes_attrib: Union[pdarray, str]) -> Strings:
        """
        Assemble a Strings object from separate offset and bytes arrays.

        This factory method constructs a segmented `Strings` array by sending two
        separate components—offsets and values—to the Arkouda server and instructing
        it to assemble them into a single `Strings` object. Use this when offsets
        and byte data are created or transported independently.

        Parameters
        ----------
        offset_attrib : pdarray or str
            The array of starting positions for each string, or a string
            expression that can be passed to `create_pdarray` to build it.
        bytes_attrib : pdarray or str
            The array of raw byte values (e.g., uint8 character codes), or a string
            expression that can be passed to `create_pdarray` to build it.

        Returns
        -------
        Strings
            A `Strings` object representing the assembled segmented strings array
            on the Arkouda server.

        Raises
        ------
        RuntimeError
            If conversion of `offset_attrib` or `bytes_attrib` to `pdarray` fails,
            or if the server is unable to assemble the parts into a `Strings`.

        Notes
        -----
        - Both inputs can be existing `pdarray` instances or arguments suitable
          for `create_pdarray`.
        - Internally uses the `CMD_ASSEMBLE` command to merge offsets and values.

        """
        from arkouda.client import generic_msg

        if not isinstance(offset_attrib, pdarray):
            try:
                offset_attrib = create_pdarray(offset_attrib)
            except Exception as e:
                raise RuntimeError(f"Failed to convert offsets: {e}") from e
        if not isinstance(bytes_attrib, pdarray):
            try:
                bytes_attrib = create_pdarray(bytes_attrib)
            except Exception as e:
                raise RuntimeError(f"Failed to convert values: {e}") from e

        response = cast(
            str,
            generic_msg(
                cmd=CMD_ASSEMBLE,
                args={"offsets": offset_attrib, "values": bytes_attrib},
            ),
        )
        return Strings.from_return_msg(response)

    def __init__(self, strings_pdarray: pdarray, bytes_size: int_scalars) -> None:
        """
        Initialize the Strings instance by setting all instance
        attributes, some of which are derived from the array parameters.

        Parameters
        ----------
        strings_pdarray : pdarray
            the array containing the meta-info on a server side strings object
        bytes_size : int_scalars
            length of the bytes array contained on the server aka total bytes

        Raises
        ------
        RuntimeError
            Raised if there's an error converting a server-returned str-descriptor
            or pdarray to either the offset_attrib or bytes_attrib
        ValueError
            Raised if there's an error in generating instance attributes
            from either the offset_attrib or bytes_attrib parameter
        """
        self.entry: pdarray = strings_pdarray
        self.registered_name: Optional[str] = None
        try:
            self.size = self.entry.size
            self.nbytes = bytes_size  # This is a deficiency of server GenSymEntry right now
            self.ndim = self.entry.ndim
            self.shape = self.entry.shape
            self.name: Optional[str] = self.entry.name
        except Exception as e:
            raise ValueError(e)

        self._bytes: Optional[pdarray] = None
        self._offsets: Optional[pdarray] = None
        self._regex_dict: Dict = dict()
        self.logger = getArkoudaLogger(name=__class__.__name__)  # type: ignore

    """
    NOTE:
         The Strings.__del__() method should NOT be implemented.
         Python will invoke the __del__() of any components by default.
         Overriding this default behavior with an explicitly specified Strings.__del__() method may
         introduce unknown symbol errors.
         By allowing Python's garbage collecting to handle this automatically, we avoid extra maintenance
    """

    def __iter__(self):
        raise NotImplementedError(
            "Strings does not support iteration. To force data transfer from server, use to_ndarray."
        )

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self) -> str:
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            vals = [f"'{self[i]}'" for i in range(self.size)]
        else:
            vals = [f"'{self[i]}'" for i in range(3)]
            vals.append("... ")
            vals.extend([f"'{self[i]}'" for i in range(self.size - 3, self.size)])
        return "[{}]".format(", ".join(vals))

    def __repr__(self) -> str:
        return f"array({self.__str__()})"

    @typechecked
    def _binop(self, other: Union[Strings, str_scalars], op: str) -> pdarray:
        """
        Execute the requested binop on this Strings instance and the
        parameter Strings object and returns the results within
        a pdarray object.

        Parameters
        ----------
        other : Strings or str_scalars
            the other object is a Strings object
        op : str
            name of the binary operation to be performed

        Returns
        -------
        pdarray
            encapsulating the results of the requested binop

        Raises
        ------
        ValueError
            Raised if (1) the op is not in the self.BinOps set, or (2) if the
            sizes of this and the other instance don't match, or (3) the other
            object is not a Strings object
        RuntimeError
            Raised if a server-side error is thrown while executing the
            binary operation
        """
        from arkouda.client import generic_msg

        if op not in self.BinOps:
            raise ValueError(f"Strings: unsupported operator: {op}")
        if isinstance(other, Strings):
            if self.size != other.size:
                raise ValueError(f"Strings: size mismatch {self.size} {other.size}")
            cmd = "segmentedBinopvv"
            args = {
                "op": op,
                "objType": self.objType,
                "obj": self.entry,
                "otherType": other.objType,
                "other": other.entry,
                "left": False,  # placeholder for stick
                "delim": "",  # placeholder for stick
            }
        elif resolve_scalar_dtype(other) == "str":
            cmd = "segmentedBinopvs"
            args = {
                "op": op,
                "objType": self.objType,
                "obj": self.entry,
                "otherType": "str",
                "other": other,
            }
        else:
            raise ValueError(
                f"Strings: {op} not supported between Strings and {other.__class__.__name__}"
            )
        return create_pdarray(generic_msg(cmd=cmd, args=args))

    def __eq__(self, other):  # type: ignore
        if self.size > 0:
            return self._binop(other, "==")
        else:
            from arkouda import array as ak_array

            return ak_array([], dtype="bool")

    def __ne__(self, other):
        if self.size > 0:  # type: ignore
            return self._binop(cast(Strings, other), "!=")
        else:
            from arkouda import array as ak_array

            return ak_array([], dtype="bool")

    def __getitem__(self, key):
        from arkouda.client import generic_msg

        if np.isscalar(key) and (resolve_scalar_dtype(key) in ["int64", "uint64"]):
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if key >= 0 and key < self.size:
                repMsg = generic_msg(
                    cmd="segmentedIndex",
                    args={
                        "subcmd": "intIndex",
                        "objType": self.objType,
                        "dtype": self.entry.dtype,
                        "obj": self.entry,
                        "key": key,
                    },
                )
                _, value = repMsg.split(maxsplit=1)
                return parse_single_value(value)
            else:
                raise IndexError(f"[int] {orig_key} is out of bounds with size {self.size}")
        elif isinstance(key, slice):
            (start, stop, stride) = key.indices(self.size)
            self.logger.debug(f"start: {start}; stop: {stop}; stride: {stride}")
            repMsg = generic_msg(
                cmd="segmentedIndex",
                args={
                    "subcmd": "sliceIndex",
                    "objType": self.objType,
                    "obj": self.entry,
                    "dtype": self.entry.dtype,
                    "key": [start, stop, stride],
                },
            )
            return Strings.from_return_msg(repMsg)
        elif isinstance(key, pdarray):
            if key.dtype not in ("bool", "int", "uint"):
                raise TypeError(f"unsupported pdarray index type {key.dtype}")
            if key.dtype == "bool" and self.size != key.size:
                raise ValueError(f"size mismatch {self.size} {key.size}")
            repMsg = generic_msg(
                cmd="segmentedIndex",
                args={
                    "subcmd": "pdarrayIndex",
                    "objType": self.objType,
                    "dtype": self.entry.dtype,
                    "obj": self.entry,
                    "key": key,
                },
            )
            return Strings.from_return_msg(repMsg)
        elif isinstance(key, np.ndarray):
            # convert numpy array to pdarray
            from arkouda.numpy.pdarraycreation import array as ak_array

            return self[ak_array(key)]
        else:
            raise TypeError(f"unsupported pdarray index type {key.__class__.__name__}")

    @property
    def dtype(self) -> npdtype:
        """
        Return the dtype object of the underlying data.
        """
        return npdtype("<U")

    @property
    def inferred_type(self) -> str:
        """
        Return a string of the type inferred from the values.
        """
        return "string"

    def copy(self) -> Strings:
        """
        Return a deep copy of the Strings object.

        Returns
        -------
        Strings
            A deep copy of the Strings.
        """
        from arkouda.pdarraycreation import array

        ret = array(self, copy=True)
        if isinstance(ret, Strings):
            return ret
        else:
            raise RuntimeError("Could not copy Strings object.")

    def equals(self, other) -> bool_scalars:
        """
        Whether Strings are the same size and all entries are equal.

        Parameters
        ----------
        other : Any
            object to compare.

        Returns
        -------
        bool_scalars
            True if the Strings are the same, o.w. False.

        Examples
        --------
        >>> import arkouda as ak
        >>> s = ak.array(["a", "b", "c"])
        >>> s_cpy = ak.array(["a", "b", "c"])
        >>> s.equals(s_cpy)
        np.True_
        >>> s2 = ak.array(["a", "x", "c"])
        >>> s.equals(s2)
        np.False_
        """
        if isinstance(other, Strings):
            if other.size != self.size:
                return False
            else:
                result = akall(self == other)
                if isinstance(result, (bool, np.bool_)):
                    return result
        return False

    def get_lengths(self) -> pdarray:
        """
        Return the length of each string in the array.

        Returns
        -------
        pdarray
            The length of each string

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(cmd="segmentLengths", args={"objType": self.objType, "obj": self.entry})
        )

    def get_bytes(self) -> pdarray:
        """
        Getter for the bytes component (uint8 pdarray) of this Strings.

        Returns
        -------
        pdarray
            Pdarray of bytes of the string accessed

        Example
        -------
        >>> import arkouda as ak
        >>> x = ak.array(['one', 'two', 'three'])
        >>> x.get_bytes()
        array([111 110 101 0 116 119 111 0 116 104 114 101 101 0])
        """
        from arkouda.client import generic_msg

        if self._bytes is None or self._bytes.name not in list_symbol_table():
            self._bytes = create_pdarray(
                generic_msg(
                    cmd="getSegStringProperty", args={"property": "get_bytes", "obj": self.entry}
                )
            )
        if self._bytes is None:
            raise RuntimeError("Failed to initialize the bytes property.")
        return self._bytes

    def get_offsets(self) -> pdarray:
        """
        Getter for the offsets component (int64 pdarray) of this Strings.

        Returns
        -------
        pdarray
            Pdarray of offsets of the string accessed

        Example
        -------
        >>> import arkouda as ak
        >>> x = ak.array(['one', 'two', 'three'])
        >>> x.get_offsets()
        array([0 4 8])
        """
        from arkouda.client import generic_msg

        if self._offsets is None or self._offsets.name not in list_symbol_table():
            self._offsets = create_pdarray(
                generic_msg(
                    cmd="getSegStringProperty", args={"property": "get_offsets", "obj": self.entry}
                )
            )
        if self._offsets is None:
            raise RuntimeError("Failed to initialize the offsets property.")
        return self._offsets

    def encode(self, toEncoding: str, fromEncoding: str = "UTF-8") -> Strings:
        """
        Return a new strings object in `toEncoding`, expecting that the
        current Strings is encoded in `fromEncoding`.

        Parameters
        ----------
        toEncoding: str
            The encoding that the strings will be converted to
        fromEncoding : str, default="UTF-8"
            The current encoding of the strings object, default to
            UTF-8

        Returns
        -------
        Strings
            A new Strings object in `toEncoding`

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown
        """
        from arkouda.client import generic_msg

        if (toEncoding.upper() == "IDNA" and fromEncoding.upper() != "UTF-8") or (
            toEncoding.upper() != "UTF-8" and fromEncoding.upper() == "IDNA"
        ):
            # first convert to UTF-8
            rep_msg = generic_msg(
                cmd="encode",
                args={
                    "toEncoding": "UTF-8",
                    "fromEncoding": fromEncoding,
                    "obj": self.entry,
                },
            )
            intermediate = Strings.from_return_msg(cast(str, rep_msg))
            # now go to idna
            rep_msg = generic_msg(
                cmd="encode",
                args={
                    "toEncoding": toEncoding,
                    "fromEncoding": "UTF-8",
                    "obj": intermediate,
                },
            )
            return Strings.from_return_msg(cast(str, rep_msg))

        rep_msg = generic_msg(
            cmd="encode",
            args={
                "toEncoding": toEncoding,
                "fromEncoding": fromEncoding,
                "obj": self.entry,
            },
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    def decode(self, fromEncoding: str, toEncoding: str = "UTF-8") -> Strings:
        """
        Return a new strings object in `fromEncoding`, expecting that the
        current Strings is encoded in `toEncoding`.

        Parameters
        ----------
        fromEncoding: str
            The current encoding of the strings object
        toEncoding : str, default="UTF-8"
            The encoding that the strings will be converted to,
            default to UTF-8

        Returns
        -------
        Strings
            A new Strings object in `toEncoding`

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown
        """
        return self.encode(toEncoding, fromEncoding)

    @typechecked
    def lower(self) -> Strings:
        """
        Return a new Strings with all uppercase characters from the original replaced with
        their lowercase equivalent.

        Returns
        -------
        Strings
            Strings with all uppercase characters from the original replaced with
            their lowercase equivalent

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.upper

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array([f'StrINgS {i}' for i in range(5)])
        >>> strings
        array(['StrINgS 0', 'StrINgS 1', 'StrINgS 2', 'StrINgS 3', 'StrINgS 4'])
        >>> strings.lower()
        array(['strings 0', 'strings 1', 'strings 2', 'strings 3', 'strings 4'])
        """
        from arkouda.client import generic_msg

        rep_msg = generic_msg(
            cmd="caseChange", args={"subcmd": "toLower", "objType": self.objType, "obj": self.entry}
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    @typechecked
    def upper(self) -> Strings:
        """
        Return a new Strings with all lowercase characters from the original replaced with
        their uppercase equivalent.

        Returns
        -------
        Strings
            Strings with all lowercase characters from the original replaced with
            their uppercase equivalent

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.lower

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array([f'StrINgS {i}' for i in range(5)])
        >>> strings
        array(['StrINgS 0', 'StrINgS 1', 'StrINgS 2', 'StrINgS 3', 'StrINgS 4'])
        >>> strings.upper()
        array(['STRINGS 0', 'STRINGS 1', 'STRINGS 2', 'STRINGS 3', 'STRINGS 4'])
        """
        from arkouda.client import generic_msg

        rep_msg = generic_msg(
            cmd="caseChange", args={"subcmd": "toUpper", "objType": self.objType, "obj": self.entry}
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    @typechecked
    def title(self) -> Strings:
        """
        Return a new Strings from the original replaced with their titlecase equivalent.

        Returns
        -------
        Strings
            Strings from the original replaced with their titlecase equivalent.

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown.

        See Also
        --------
        Strings.lower
        String.upper

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array([f'StrINgS {i}' for i in range(5)])
        >>> strings
        array(['StrINgS 0', 'StrINgS 1', 'StrINgS 2', 'StrINgS 3', 'StrINgS 4'])
        >>> strings.title()
        array(['Strings 0', 'Strings 1', 'Strings 2', 'Strings 3', 'Strings 4'])
        """
        from arkouda.client import generic_msg

        rep_msg = generic_msg(
            cmd="caseChange", args={"subcmd": "toTitle", "objType": self.objType, "obj": self.entry}
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    @typechecked
    def isdecimal(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i of the
        Strings has all decimal characters.

        Returns
        -------
        pdarray
            True for elements that are decimals, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.isdigit

        Examples
        --------
        >>> import arkouda as ak
        >>> not_decimal = ak.array([f'Strings {i}' for i in range(3)])
        >>> decimal = ak.array([f'12{i}' for i in range(3)])
        >>> strings = ak.concatenate([not_decimal, decimal])
        >>> strings
        array(['Strings 0', 'Strings 1', 'Strings 2', '120', '121', '122'])
        >>> strings.isdecimal()
        array([False False False True True True])

        Special Character Examples

        >>> special_strings = ak.array(["3.14", "\u0030", "\u00b2", "2³₇", "2³x₇"])
        >>> special_strings
        array(['3.14', '0', '²', '2³₇', '2³x₇'])
        >>> special_strings.isdecimal()
        array([False True False False False])
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars",
                args={"subcmd": "isDecimal", "objType": self.objType, "obj": self.entry},
            )
        )

    @typechecked
    def isnumeric(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i of the
        Strings has all numeric characters. There are 1922 unicode characters that
        qualify as numeric, including the digits 0 through 9, superscripts and
        subscripted digits, special characters with the digits encircled or
        enclosed in parens, "vulgar fractions," and more.

        Returns
        -------
        pdarray
            True for elements that are numerics, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.isdecimal

        Examples
        --------
        >>> import arkouda as ak
        >>> not_numeric = ak.array([f'Strings {i}' for i in range(3)])
        >>> numeric = ak.array([f'12{i}' for i in range(3)])
        >>> strings = ak.concatenate([not_numeric, numeric])
        >>> strings
        array(['Strings 0', 'Strings 1', 'Strings 2', '120', '121', '122'])
        >>> strings.isnumeric()
        array([False False False True True True])

        Special Character Examples

        >>> special_strings = ak.array(["3.14", "\u0030", "\u00b2", "2³₇", "2³x₇"])
        >>> special_strings
        array(['3.14', '0', '²', '2³₇', '2³x₇'])
        >>> special_strings.isnumeric()
        array([False True True True False])
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars",
                args={"subcmd": "isNumeric", "objType": self.objType, "obj": self.entry},
            )
        )

    @typechecked
    def capitalize(self) -> Strings:
        """
        Return a new Strings from the original replaced with the first letter capitilzed
        and the remaining letters lowercase.

        Returns
        -------
        Strings
            Strings from the original replaced with the capitalized equivalent.

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown.

        See Also
        --------
        Strings.lower
        String.upper
        String.title

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array([f'StrINgS aRe Here {i}' for i in range(5)])
        >>> strings
        array(['StrINgS aRe Here 0', 'StrINgS aRe Here 1', 'StrINgS aRe Here 2', \
'StrINgS aRe Here 3', 'StrINgS aRe Here 4'])
        >>> strings.title()
        array(['Strings Are Here 0', 'Strings Are Here 1', 'Strings Are Here 2', \
'Strings Are Here 3', 'Strings Are Here 4'])
        """
        from arkouda.client import generic_msg

        rep_msg = generic_msg(
            cmd="caseChange", args={"subcmd": "capitalize", "objType": self.objType, "obj": self.entry}
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    @typechecked
    def islower(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i of the
        Strings is entirely lowercase.

        Returns
        -------
        pdarray
            True for elements that are entirely lowercase, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.isupper

        Examples
        --------
        >>> import arkouda as ak
        >>> lower = ak.array([f'strings {i}' for i in range(3)])
        >>> upper = ak.array([f'STRINGS {i}' for i in range(3)])
        >>> strings = ak.concatenate([lower, upper])
        >>> strings
        array(['strings 0', 'strings 1', 'strings 2', 'STRINGS 0', 'STRINGS 1', 'STRINGS 2'])
        >>> strings.islower()
        array([True True True False False False])
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isLower", "objType": self.objType, "obj": self.entry}
            )
        )

    @typechecked
    def isupper(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i of the
        Strings is entirely uppercase.

        Returns
        -------
        pdarray
            True for elements that are entirely uppercase, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.islower

        Examples
        --------
        >>> import arkouda as ak
        >>> lower = ak.array([f'strings {i}' for i in range(3)])
        >>> upper = ak.array([f'STRINGS {i}' for i in range(3)])
        >>> strings = ak.concatenate([lower, upper])
        >>> strings
        array(['strings 0', 'strings 1', 'strings 2', 'STRINGS 0', 'STRINGS 1', 'STRINGS 2'])
        >>> strings.isupper()
        array([False False False True True True])
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isUpper", "objType": self.objType, "obj": self.entry}
            )
        )

    @typechecked
    def istitle(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i of the
        Strings is titlecase.

        Returns
        -------
        pdarray
            True for elements that are titlecase, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.islower
        Strings.isupper

        Examples
        --------
        >>> import arkouda as ak
        >>> mixed = ak.array([f'sTrINgs {i}' for i in range(3)])
        >>> title = ak.array([f'Strings {i}' for i in range(3)])
        >>> strings = ak.concatenate([mixed, title])
        >>> strings
        array(['sTrINgs 0', 'sTrINgs 1', 'sTrINgs 2', 'Strings 0', 'Strings 1', 'Strings 2'])
        >>> strings.istitle()
        array([False False False True True True])
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isTitle", "objType": self.objType, "obj": self.entry}
            )
        )

    @typechecked
    def isalnum(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i of the
        Strings is alphanumeric.

        Returns
        -------
        pdarray
            True for elements that are alphanumeric, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.islower
        Strings.isupper
        Strings.istitle

        Examples
        --------
        >>> import arkouda as ak
        >>> not_alnum = ak.array([f'%Strings {i}' for i in range(3)])
        >>> alnum = ak.array([f'Strings{i}' for i in range(3)])
        >>> strings = ak.concatenate([not_alnum, alnum])
        >>> strings
        array(['%Strings 0', '%Strings 1', '%Strings 2', 'Strings0', 'Strings1', 'Strings2'])
        >>> strings.isalnum()
        array([False False False True True True])
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isalnum", "objType": self.objType, "obj": self.entry}
            )
        )

    @typechecked
    def isalpha(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i of the
        Strings is alphabetic.  This means there is at least one character,
        and all the characters are alphabetic.

        Returns
        -------
        pdarray
            True for elements that are alphabetic, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.islower
        Strings.isupper
        Strings.istitle
        Strings.isalnum

        Examples
        --------
        >>> import arkouda as ak
        >>> not_alpha = ak.array([f'%Strings {i}' for i in range(3)])
        >>> alpha = ak.array(['StringA','StringB','StringC'])
        >>> strings = ak.concatenate([not_alpha, alpha])
        >>> strings
        array(['%Strings 0', '%Strings 1', '%Strings 2', 'StringA', 'StringB', 'StringC'])
        >>> strings.isalpha()
        array([False False False True True True])
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isalpha", "objType": self.objType, "obj": self.entry}
            )
        )

    @typechecked
    def isdigit(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i of the
        Strings has all digit characters.

        Returns
        -------
        pdarray
            True for elements that are digits, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.islower
        Strings.isupper
        Strings.istitle

        Examples
        --------
        >>> import arkouda as ak
        >>> not_digit = ak.array([f'Strings {i}' for i in range(3)])
        >>> digit = ak.array([f'12{i}' for i in range(3)])
        >>> strings = ak.concatenate([not_digit, digit])
        >>> strings
        array(['Strings 0', 'Strings 1', 'Strings 2', '120', '121', '122'])
        >>> strings.isdigit()
        array([False False False True True True])

        Special Character Examples

        >>> special_strings = ak.array(["3.14", "\u0030", "\u00b2", "2³₇", "2³x₇"])
        >>> special_strings
        array(['3.14', '0', '²', '2³₇', '2³x₇'])
        >>> special_strings.isdigit()
        array([False True True True False])
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isdigit", "objType": self.objType, "obj": self.entry}
            )
        )

    @typechecked
    def isempty(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i of the
        Strings is empty.


        True for elements that are the empty string, False otherwise

        Returns
        -------
        pdarray
            True for elements that are digits, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.islower
        Strings.isupper
        Strings.istitle

        Examples
        --------
        >>> import arkouda as ak
        >>> not_empty = ak.array([f'Strings {i}' for i in range(3)])
        >>> empty = ak.array(['' for i in range(3)])
        >>> strings = ak.concatenate([not_empty, empty])
        >>> strings
        array(['Strings 0', 'Strings 1', 'Strings 2', '', '', ''])
        >>> strings.isempty()
        array([False False False True True True])
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isempty", "objType": self.objType, "obj": self.entry}
            )
        )

    @typechecked
    def isspace(self) -> pdarray:
        """
        Return a boolean pdarray where index i indicates whether string i has all
        whitespace characters (‘ ’, ‘\t’, ‘\n’, ‘\v’, ‘\f’, ‘\r’).

        Returns
        -------
        pdarray
            True for elements that are whitespace, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.islower
        Strings.isupper
        Strings.istitle

        Examples
        --------
        >>> import arkouda as ak
        >>> not_space = ak.array([f'Strings {i}' for i in range(3)])
        >>> space = ak.array([' ', '\\t', '\\n', '\\v', '\\f', '\\r', ' \\t\\n\\v\\f\\r'])
        >>> strings = ak.concatenate([not_space, space])
        >>> strings
        array(['Strings 0', 'Strings 1', 'Strings 2', ' ', 'u0009', 'n', \
'u000B', 'u000C', 'u000D', ' u0009nu000Bu000Cu000D'])
        >>> strings.isspace()
        array([False False False True True True True True True True])

        """  # noqa: D301
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isspace", "objType": self.objType, "obj": self.entry}
            )
        )

    @typechecked
    def strip(self, chars: Optional[Union[bytes, str_scalars]] = "") -> Strings:
        """
        Return a new Strings object with all leading and trailing occurrences of characters contained
        in chars removed. The chars argument is a string specifying the set of characters to be removed.
        If omitted, the chars argument defaults to removing whitespace. The chars argument is not a
        prefix or suffix; rather, all combinations of its values are stripped.

        Parameters
        ----------
        chars : bytes or str_scalars, optional
            the set of characters to be removed

        Returns
        -------
        Strings
            Strings object with the leading and trailing characters matching the set of characters in
            the chars argument removed

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['Strings ', '  StringS  ', 'StringS   '])
        >>> s = strings.strip()
        >>> s
        array(['Strings', 'StringS', 'StringS'])

        >>> strings = ak.array(['Strings 1', '1 StringS  ', '  1StringS  12 '])
        >>> s = strings.strip(' 12')
        >>> s
        array(['Strings', 'StringS', 'StringS'])
        """
        from arkouda.client import generic_msg

        if isinstance(chars, bytes):
            chars = chars.decode()
        rep_msg = generic_msg(
            cmd="segmentedStrip", args={"objType": self.objType, "name": self.entry, "chars": chars}
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    @typechecked
    def cached_regex_patterns(self) -> List:
        """
        Returns the regex patterns for which Match objects have been cached.
        """
        return list(self._regex_dict.keys())

    @typechecked
    def purge_cached_regex_patterns(self) -> None:
        """
        Purges cached regex patterns.
        """
        self._regex_dict = dict()

    def _empty_pattern_verification(self, pattern):
        if pattern == "$" or (re.search(pattern, "") and (self == "").any()):  # type: ignore
            # TODO remove once changes from chapel issue #20431 and #20441 are in arkouda
            raise ValueError(
                "regex operations not currently supported with a pattern='$' or pattern='' when "
                "the empty string is contained in Strings"
            )

    def _get_matcher(self, pattern: Union[bytes, str_scalars], create: bool = True):
        """
        Internal function to fetch cached Matcher objects.
        """
        from arkouda.pandas.matcher import Matcher

        if isinstance(pattern, bytes):
            pattern = pattern.decode()
        try:
            re.compile(pattern)
        except Exception as e:
            raise ValueError(e)
        self._empty_pattern_verification(pattern)
        matcher = None
        if pattern in self._regex_dict:
            matcher = self._regex_dict[pattern]
        elif create:
            self._regex_dict[pattern] = Matcher(pattern=pattern, parent_entry_name=self.entry.name)
            matcher = self._regex_dict[pattern]
        return matcher

    @typechecked
    def find_locations(self, pattern: Union[bytes, str_scalars]) -> Tuple[pdarray, pdarray, pdarray]:
        r"""
        Finds pattern matches and returns pdarrays containing the number, start postitions,
        and lengths of matches.

        Parameters
        ----------
        pattern : bytes or str_scalars
            The regex pattern used to find matches

        Returns
        -------
        Tuple[pdarray, pdarray, pdarray]
            pdarray, int64
                For each original string, the number of pattern matches
            pdarray, int64
                The start positons of pattern matches
            pdarray, int64
                The lengths of pattern matches

        Raises
        ------
        TypeError
            Raised if the pattern parameter is not bytes or str_scalars
        ValueError
            Raised if pattern is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.findall, Strings.match

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array([f'{i} string {i}' for i in range(1, 6)])
        >>> num_matches, starts, lens = strings.find_locations('\\d')
        >>> num_matches
        array([2 2 2 2 2])
        >>> starts
        array([0 9 0 9 0 9 0 9 0 9])
        >>> lens
        array([1 1 1 1 1 1 1 1 1 1])
        """
        matcher = self._get_matcher(pattern)
        matcher.find_locations()
        return matcher.num_matches, matcher.starts, matcher.lengths

    @typechecked
    def search(self, pattern: Union[bytes, str_scalars]) -> Match:
        """
        Return a match object with the first location in each element where pattern produces a match.
        Elements match if any part of the string matches the regular expression pattern.

        Parameters
        ----------
        pattern : bytes or str_scalars
            Regex used to find matches

        Returns
        -------
        Match
            Match object where elements match if any part of the string matches the
            regular expression pattern

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.search('_+')
        <ak.Match object: matched=True, span=(1, 2); matched=True, span=(0, 4);
        matched=False; matched=True, span=(0, 2); matched=False>
        """
        return self._get_matcher(pattern).get_match(MatchType.SEARCH, self)

    @typechecked
    def match(self, pattern: Union[bytes, str_scalars]) -> Match:
        """
        Return a match object where elements match only if the beginning of the string matches the
        regular expression pattern.

        Parameters
        ----------
        pattern : bytes or str_scalars
            Regex used to find matches

        Returns
        -------
        Match
            Match object where elements match only if the beginning of the string matches the
            regular expression pattern

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.match('_+')
        <ak.Match object: matched=False; matched=True, span=(0, 4); matched=False;
        matched=True, span=(0, 2); matched=False>
        """
        return self._get_matcher(pattern).get_match(MatchType.MATCH, self)

    @typechecked()
    def fullmatch(self, pattern: Union[bytes, str_scalars]) -> Match:
        """
        Return a match object where elements match only if the whole string matches the
        regular expression pattern.

        Parameters
        ----------
        pattern : bytes or str_scalars
            Regex used to find matches

        Returns
        -------
        Match
            Match object where elements match only if the whole string matches the
            regular expression pattern

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.fullmatch('_+')
        <ak.Match object: matched=False; matched=True, span=(0, 4); matched=False;
        matched=False; matched=False>
        """
        return self._get_matcher(pattern).get_match(MatchType.FULLMATCH, self)

    @typechecked()
    def regex_split(
        self, pattern: Union[bytes, str_scalars], maxsplit: int = 0, return_segments: bool = False
    ) -> Union[Strings, Tuple]:
        """
        Return a new Strings split by the occurrences of pattern.

        If maxsplit is nonzero, at most maxsplit splits occur.

        Parameters
        ----------
        pattern : bytes or str_scalars
            Regex used to split strings into substrings
        maxsplit : int, default=0
            The max number of pattern match occurences in each element to split.
            The default maxsplit=0 splits on all occurences
        return_segments : bool, default=False
            If True, return mapping of original strings to first substring
            in return array.

        Returns
        -------
        Union[Strings, Tuple]
            Strings
                Substrings with pattern matches removed
            pdarray, int64 (optional)
                For each original string, the index of first corresponding substring
                in the return array

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.regex_split('_+', maxsplit=2, return_segments=True)
        (array(['1', '2', '', '', '', '3', '', '4', '5____6___7', '']), array([0 3 5 6 9]))
        """
        return self._get_matcher(pattern).split(maxsplit, return_segments)

    @typechecked
    def findall(
        self, pattern: Union[bytes, str_scalars], return_match_origins: bool = False
    ) -> Union[Strings, Tuple]:
        """
        Return a new Strings containg all non-overlapping matches of pattern.

        Parameters
        ----------
        pattern : bytes or str_scalars
            Regex used to find matches
        return_match_origins : bool, default=False
            If True, return a pdarray containing the index of the original string each
            pattern match is from

        Returns
        -------
        Union[Strings, Tuple]
            Strings
                Strings object containing only pattern matches
            pdarray, int64 (optional)
                The index of the original string each pattern match is from

        Raises
        ------
        TypeError
            Raised if the pattern parameter is not bytes or str_scalars
        ValueError
            Raised if pattern is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.find_locations

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.findall('_+', return_match_origins=True)
        (array(['_', '___', '____', '__', '___', '____', '___']), array([0 0 1 3 3 3 3]))

        """
        return self._get_matcher(pattern).findall(return_match_origins)

    @typechecked()
    def sub(
        self, pattern: Union[bytes, str_scalars], repl: Union[bytes, str_scalars], count: int = 0
    ) -> Strings:
        """
        Return new Strings obtained by replacing non-overlapping occurrences of pattern with the
        replacement repl.

        If count is nonzero, at most count substitutions occur.

        Parameters
        ----------
        pattern : bytes or str_scalars
            The regex to substitue
        repl : bytes or str_scalars
            The substring to replace pattern matches with
        count : int, default=0
            The max number of pattern match occurences in each element to replace.
            The default count=0 replaces all occurences of pattern with repl

        Returns
        -------
        Strings
            Strings with pattern matches replaced

        Raises
        ------
        TypeError
            Raised if pattern or repl are not bytes or str_scalars
        ValueError
            Raised if pattern is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.subn

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.sub(pattern='_+', repl='-', count=2)
        array(['1-2-', '-', '3', '-4-5____6___7', ''])

        """
        if isinstance(repl, bytes):
            repl = repl.decode()
        return self._get_matcher(pattern).sub(repl, count)

    @typechecked()
    def subn(
        self, pattern: Union[bytes, str_scalars], repl: Union[bytes, str_scalars], count: int = 0
    ) -> Tuple[Strings, pdarray]:
        """
        Perform the same operation as sub(), but return a tuple (new_Strings, number_of_substitions).

        Parameters
        ----------
        pattern : bytes or str_scalars
            The regex to substitue
        repl : bytes or str_scalars
            The substring to replace pattern matches with
        count : int, default=0
            The max number of pattern match occurences in each element to replace.
            The default count=0 replaces all occurences of pattern with repl

        Returns
        -------
        Tuple[Strings, pdarray]
            Strings
                Strings with pattern matches replaced
            pdarray, int64
                The number of substitutions made for each element of Strings

        Raises
        ------
        TypeError
            Raised if pattern or repl are not bytes or str_scalars
        ValueError
            Raised if pattern is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.sub

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.subn(pattern='_+', repl='-', count=2)
        (array(['1-2-', '-', '3', '-4-5____6___7', '']), array([2 1 0 2 0]))

        """
        if isinstance(repl, bytes):
            repl = repl.decode()
        return self._get_matcher(pattern).sub(repl, count, return_num_subs=True)

    @typechecked
    def contains(self, substr: Union[bytes, str_scalars], regex: bool = False) -> pdarray:
        r"""
        Check whether each element contains the given substring.

        Parameters
        ----------
        substr : bytes or str_scalars
            The substring in the form of string or byte array to search for
        regex : bool, default=False
            Indicates whether substr is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        pdarray
            True for elements that contain substr, False otherwise

        Raises
        ------
        TypeError
            Raised if the substr parameter is not bytes or str_scalars
        ValueError
            Rasied if substr is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.startswith, Strings.endswith

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array([f'{i} string {i}' for i in range(1, 6)])
        >>> strings
        array(['1 string 1', '2 string 2', '3 string 3', '4 string 4', '5 string 5'])
        >>> strings.contains('string')
        array([True True True True True])
        >>> strings.contains('string \\d', regex=True)
        array([True True True True True])

        """
        from arkouda.client import generic_msg

        if isinstance(substr, bytes):
            substr = substr.decode()
        if not regex:
            substr = re.escape(substr)
        self._empty_pattern_verification(substr)
        matcher = self._get_matcher(substr, create=False)
        if matcher is not None:
            return matcher.get_match(MatchType.SEARCH, self).matched()
        return create_pdarray(
            generic_msg(
                cmd="segmentedSearch",
                args={"objType": self.objType, "obj": self.entry, "valType": "str", "val": substr},
            )
        )

    @typechecked
    def startswith(self, substr: Union[bytes, str_scalars], regex: bool = False) -> pdarray:
        r"""
        Check whether each element starts with the given substring.

        Parameters
        ----------
        substr : bytes or str_scalars
            The prefix to search for
        regex : bool, default=False
            Indicates whether substr is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        pdarray
            True for elements that start with substr, False otherwise

        Raises
        ------
        TypeError
            Raised if the substr parameter is not a bytes ior str_scalars
        ValueError
            Rasied if substr is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.contains, Strings.endswith

        Examples
        --------
        >>> import arkouda as ak
        >>> strings_end = ak.array([f'string {i}' for i in range(1, 6)])
        >>> strings_end
        array(['string 1', 'string 2', 'string 3', 'string 4', 'string 5'])
        >>> strings_end.startswith('string')
        array([True True True True True])
        >>> strings_start = ak.array([f'{i} string' for i in range(1,6)])
        >>> strings_start
        array(['1 string', '2 string', '3 string', '4 string', '5 string'])
        >>> strings_start.startswith('\\d str', regex = True)
        array([True True True True True])

        """
        if isinstance(substr, bytes):
            substr = substr.decode()
        if not regex:
            substr = re.escape(substr)
        self._empty_pattern_verification(substr)
        matcher = self._get_matcher(substr, create=False)
        if matcher is not None:
            return matcher.get_match(MatchType.MATCH, self).matched()
        else:
            return self.contains("^" + substr, regex=True)

    @typechecked
    def endswith(self, substr: Union[bytes, str_scalars], regex: bool = False) -> pdarray:
        r"""
        Check whether each element ends with the given substring.

        Parameters
        ----------
        substr : bytes or str_scalars
            The suffix to search for
        regex : bool, default=False
            Indicates whether substr is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        pdarray
            True for elements that end with substr, False otherwise

        Raises
        ------
        TypeError
            Raised if the substr parameter is not bytes or str_scalars
        ValueError
            Rasied if substr is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.contains, Strings.startswith

        Examples
        --------
        >>> import arkouda as ak
        >>> strings_start = ak.array([f'{i} string' for i in range(1,6)])
        >>> strings_start
        array(['1 string', '2 string', '3 string', '4 string', '5 string'])
        >>> strings_start.endswith('ing')
        array([True True True True True])
        >>> strings_end = ak.array([f'string {i}' for i in range(1, 6)])
        >>> strings_end
        array(['string 1', 'string 2', 'string 3', 'string 4', 'string 5'])
        >>> strings_end.endswith('ing \\d', regex = True)
        array([True True True True True])

        """
        if isinstance(substr, bytes):
            substr = substr.decode()
        if not regex:
            substr = re.escape(substr)
        self._empty_pattern_verification(substr)
        return self.contains(substr + "$", regex=True)

    def split(
        self, delimiter: str, return_segments: bool = False, regex: bool = False
    ) -> Union[Strings, Tuple]:
        """Unpack delimiter-joined substrings into a flat array.

        Parameters
        ----------
        delimiter: str
            Characters used to split strings into substrings
        return_segments : bool, default=False
            If True, also return mapping of original strings to first substring
            in return array.
        regex : bool, default=False
            Indicates whether delimiter is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        Union[Strings, Tuple]
            Strings
                Flattened substrings with delimiters removed
            pdarray, int64 (optional)
                For each original string, the index of first corresponding substring
                in the return array

        See Also
        --------
        peel, rpeel

        Examples
        --------
        >>> import arkouda as ak
        >>> orig = ak.array(['one|two', 'three|four|five', 'six'])
        >>> orig.split('|')
        array(['one', 'two', 'three', 'four', 'five', 'six'])
        >>> flat, mapping = orig.split('|', return_segments=True)
        >>> mapping
        array([0 2 5])
        >>> under = ak.array(['one_two', 'three_____four____five', 'six'])
        >>> under_split, under_map = under.split('_+', return_segments=True, regex=True)
        >>> under_split
        array(['one', 'two', 'three', 'four', 'five', 'six'])
        >>> under_map
        array([0 2 5])
        """
        from arkouda.client import generic_msg

        if regex:
            try:
                re.compile(delimiter)
            except Exception as e:
                raise ValueError(e)
            return self.regex_split(delimiter, return_segments=return_segments)
        else:
            cmd = "segmentedFlatten"
            repMsg = cast(
                str,
                generic_msg(
                    cmd=cmd,
                    args={
                        "values": self.entry,
                        "objtype": self.objType,
                        "return_segs": return_segments,
                        "regex": regex,
                        "delim": delimiter,
                    },
                ),
            )
            if return_segments:
                arrays = repMsg.split("+", maxsplit=2)
                return Strings.from_return_msg("+".join(arrays[0:2])), create_pdarray(arrays[2])
            else:
                return Strings.from_return_msg(repMsg)

    @typechecked
    def peel(
        self,
        delimiter: Union[bytes, str_scalars],
        times: int_scalars = 1,
        includeDelimiter: bool = False,
        keepPartial: bool = False,
        fromRight: bool = False,
        regex: bool = False,
    ) -> Tuple[Strings, Strings]:
        """
        Peel off one or more delimited fields from each string (similar
        to string.partition), returning two new arrays of strings.
        *Warning*: This function is experimental and not guaranteed to work.

        Parameters
        ----------
        delimiter : bytes or str_scalars
            The separator where the split will occur
        times : int_scalars, default=1
            The number of times the delimiter is sought, i.e. skip over
            the first (times-1) delimiters
        includeDelimiter : bool, default=False
            If true, append the delimiter to the end of the first return
            array. By default, it is prepended to the beginning of the
            second return array.
        keepPartial : bool, default=False
            If true, a string that does not contain <times> instances of
            the delimiter will be returned in the first array. By default,
            such strings are returned in the second array.
        fromRight : bool, default=False
            If true, peel from the right instead of the left (see also rpeel)
        regex : bool, default=False
            Indicates whether delimiter is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        Tuple[Strings, Strings]
            left: Strings
                The field(s) peeled from the end of each string (unless
                fromRight is true)
            right: Strings
                The remainder of each string after peeling (unless fromRight
                is true)

        Raises
        ------
        TypeError
            Raised if the delimiter parameter is not byte or str_scalars, if
            times is not int64, or if includeDelimiter, keepPartial, or
            fromRight is not bool
        ValueError
            Raised if times is < 1 or if delimiter is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        rpeel, stick, lstick

        Examples
        --------
        >>> import arkouda as ak
        >>> s = ak.array(['a.b', 'c.d', 'e.f.g'])
        >>> s.peel('.')
        (array(['a', 'c', 'e']), array(['b', 'd', 'f.g']))
        >>> s.peel('.', includeDelimiter=True)
        (array(['a.', 'c.', 'e.']), array(['b', 'd', 'f.g']))
        >>> s.peel('.', times=2)
        (array(['', '', 'e.f']), array(['a.b', 'c.d', 'g']))
        >>> s.peel('.', times=2, keepPartial=True)
        (array(['a.b', 'c.d', 'e.f']), array(['', '', 'g']))
        """
        from arkouda.client import generic_msg

        if isinstance(delimiter, bytes):
            delimiter = delimiter.decode()
        if regex:
            try:
                re.compile(delimiter)
            except Exception as e:
                raise ValueError(e)
            if re.search(delimiter, ""):
                raise ValueError(
                    "peel with a pattern that matches the empty string are not currently supported"
                )
        if times < 1:
            raise ValueError("times must be >= 1")
        repMsg = generic_msg(
            cmd="segmentedPeel",
            args={
                "subcmd": "peel",
                "objType": self.objType,
                "obj": self.entry,
                "valType": "str",
                "times": NUMBER_FORMAT_STRINGS["int64"].format(times),
                "id": NUMBER_FORMAT_STRINGS["bool"].format(includeDelimiter),
                "keepPartial": NUMBER_FORMAT_STRINGS["bool"].format(keepPartial),
                "lStr": NUMBER_FORMAT_STRINGS["bool"].format(not fromRight),
                "regex": NUMBER_FORMAT_STRINGS["bool"].format(regex),
                "delim": delimiter,
            },
        )
        arrays = cast(str, repMsg).split("+", maxsplit=3)
        # first two created are left Strings, last two are right strings
        left_str = Strings.from_return_msg("+".join(arrays[0:2]))
        right_str = Strings.from_return_msg("+".join(arrays[2:4]))
        return left_str, right_str

    def rpeel(
        self,
        delimiter: Union[bytes, str_scalars],
        times: int_scalars = 1,
        includeDelimiter: bool = False,
        keepPartial: bool = False,
        regex: bool = False,
    ) -> Tuple[Strings, Strings]:
        """
        Peel off one or more delimited fields from the end of each string
        (similar to string.rpartition), returning two new arrays of strings.
        *Warning*: This function is experimental and not guaranteed to work.

        Parameters
        ----------
        delimiter : bytes or str_scalars
            The separator where the split will occur
        times : int_scalars, default=1
            The number of times the delimiter is sought, i.e. skip over
            the last (times-1) delimiters
        includeDelimiter : bool, default=False
            If true, prepend the delimiter to the start of the first return
            array. By default, it is appended to the end of the
            second return array.
        keepPartial : bool, default=False
            If true, a string that does not contain <times> instances of
            the delimiter will be returned in the second array. By default,
            such strings are returned in the first array.
        regex : bool, default=False
            Indicates whether delimiter is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        Tuple[Strings, Strings]
            left: Strings
                The remainder of the string after peeling
            right: Strings
                The field(s) that were peeled from the right of each string

        Raises
        ------
        TypeError
            Raised if the delimiter parameter is not bytes or str_scalars or
            if times is not int64
        ValueError
            Raised if times is < 1 or if delimiter is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        peel, stick, lstick

        Examples
        --------
        >>> import arkouda as ak
        >>> s = ak.array(['a.b', 'c.d', 'e.f.g'])
        >>> s.rpeel('.')
        (array(['a', 'c', 'e.f']), array(['b', 'd', 'g']))

        Compared against peel

        >>> s.peel('.')
        (array(['a', 'c', 'e']), array(['b', 'd', 'f.g']))
        """
        return self.peel(
            delimiter,
            times=times,
            includeDelimiter=includeDelimiter,
            keepPartial=keepPartial,
            fromRight=True,
            regex=regex,
        )

    @typechecked
    def stick(
        self, other: Strings, delimiter: Union[bytes, str_scalars] = "", toLeft: bool = False
    ) -> Strings:
        """
        Join the strings from another array onto one end of the strings
        of this array, optionally inserting a delimiter.
        *Warning*: This function is experimental and not guaranteed to work.

        Parameters
        ----------
        other : Strings
            The strings to join onto self's strings
        delimiter : bytes or str_scalars, default=""
            String inserted between self and other
        toLeft : bool, default=False
            If true, join other strings to the left of self. By default,
            other is joined to the right of self.

        Returns
        -------
        Strings
            The array of joined strings

        Raises
        ------
        TypeError
            Raised if the delimiter parameter is not bytes or str_scalars
            or if the other parameter is not a Strings instance
        ValueError
            Raised if times is < 1
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        lstick, peel, rpeel

        Examples
        --------
        >>> import arkouda as ak
        >>> s = ak.array(['a', 'c', 'e'])
        >>> t = ak.array(['b', 'd', 'f'])
        >>> s.stick(t, delimiter='.')
        array(['a.b', 'c.d', 'e.f'])
        """
        from arkouda.client import generic_msg

        if isinstance(delimiter, bytes):
            delimiter = delimiter.decode()
        rep_msg = generic_msg(
            cmd="segmentedBinopvv",
            args={
                "op": "stick",
                "objType": self.objType,
                "obj": self.entry,
                "otherType": other.objType,
                "other": other.entry,
                "left": NUMBER_FORMAT_STRINGS["bool"].format(toLeft),
                "delim": delimiter,
            },
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    def __add__(self, other: Strings) -> Strings:
        return self.stick(other)

    def lstick(self, other: Strings, delimiter: Union[bytes, str_scalars] = "") -> Strings:
        """
        Join the strings from another array onto the left of the strings
        of this array, optionally inserting a delimiter.
        *Warning*: This function is experimental and not guaranteed to work.

        Parameters
        ----------
        other : Strings
            The strings to join onto self's strings
        delimiter : bytes or str_scalars, default=""
            String inserted between self and other

        Returns
        -------
        Strings
            The array of joined strings, as other + self

        Raises
        ------
        TypeError
            Raised if the delimiter parameter is neither bytes nor a str
            or if the other parameter is not a Strings instance

        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        stick, peel, rpeel

        Examples
        --------
        >>> import arkouda as ak
        >>> s = ak.array(['a', 'c', 'e'])
        >>> t = ak.array(['b', 'd', 'f'])
        >>> s.lstick(t, delimiter='.')
        array(['b.a', 'd.c', 'f.e'])
        """
        return self.stick(other, delimiter=delimiter, toLeft=True)

    def __radd__(self, other: Strings) -> Strings:
        return self.lstick(other)

    def get_prefixes(
        self, n: int_scalars, return_origins: bool = True, proper: bool = True
    ) -> Union[Strings, Tuple[Strings, pdarray]]:
        """
        Return the n-long prefix of each string, where possible.

        Parameters
        ----------
        n : int_scalars
            Length of prefix
        return_origins : bool, default=True
            If True, return a logical index indicating which strings
            were long enough to return an n-prefix
        proper : bool, default=True
            If True, only return proper prefixes, i.e. from strings
            that are at least n+1 long. If False, allow the entire
            string to be returned as a prefix.

        Returns
        -------
        Union[Strings, Tuple[Strings, pdarray]]
            prefixes : Strings
                The array of n-character prefixes; the number of elements is the number of
                True values in the returned mask.
            origin_indices : pdarray, bool
                Boolean array that is True where the string was long enough to return
                an n-character prefix, False otherwise.
        """
        from arkouda.client import generic_msg

        repMsg = cast(
            str,
            generic_msg(
                cmd="segmentedSubstring",
                args={
                    "objType": self.objType,
                    "name": self,
                    "nChars": n,
                    "returnOrigins": return_origins,
                    "kind": "prefixes",
                    "proper": proper,
                },
            ),
        )
        if return_origins:
            parts = repMsg.split("+")
            prefixes = Strings.from_return_msg("+".join(parts[:2]))
            longenough = create_pdarray(parts[2])
            return prefixes, cast(pdarray, longenough)
        else:
            return Strings.from_return_msg(repMsg)

    def get_suffixes(
        self, n: int_scalars, return_origins: bool = True, proper: bool = True
    ) -> Union[Strings, Tuple[Strings, pdarray]]:
        """
        Return the n-long suffix of each string, where possible.

        Parameters
        ----------
        n : int_scalars
            Length of suffix
        return_origins : bool, default=True
            If True, return a logical index indicating which strings
            were long enough to return an n-suffix
        proper : bool, default=True
            If True, only return proper suffixes, i.e. from strings
            that are at least n+1 long. If False, allow the entire
            string to be returned as a suffix.

        Returns
        -------
        Union[Strings, Tuple[Strings, pdarray]]
            suffixes : Strings
                The array of n-character suffixes; the number of elements is the number of
                True values in the returned mask.
            origin_indices : pdarray, bool
                Boolean array that is True where the string was long enough to return
                an n-character suffix, False otherwise.
        """
        from arkouda.client import generic_msg

        repMsg = cast(
            str,
            generic_msg(
                cmd="segmentedSubstring",
                args={
                    "objType": self.objType,
                    "name": self,
                    "nChars": n,
                    "returnOrigins": return_origins,
                    "kind": "suffixes",
                    "proper": proper,
                },
            ),
        )
        if return_origins:
            parts = repMsg.split("+")
            suffixes = Strings.from_return_msg("+".join(parts[:2]))
            longenough = create_pdarray(parts[2])
            return suffixes, cast(pdarray, longenough)
        else:
            return Strings.from_return_msg(repMsg)

    def hash(self) -> Tuple[pdarray, pdarray]:
        """
        Compute a 128-bit hash of each string.

        Returns
        -------
        Tuple[pdarray,pdarray]
            A tuple of two int64 pdarrays. The ith hash value is the concatenation
            of the ith values from each array.

        Notes
        -----
        The implementation uses SipHash128, a fast and balanced hash function (used
        by Python for dictionaries and sets). For realistic numbers of strings (up
        to about 10**15), the probability of a collision between two 128-bit hash
        values is negligible.
        """
        from arkouda.client import generic_msg

        # TODO fix this to return a single pdarray of hashes
        repMsg = generic_msg(cmd="segmentedHash", args={"objType": self.objType, "obj": self.entry})
        h1, h2 = cast(str, repMsg).split("+")
        return create_pdarray(h1), create_pdarray(h2)

    def group(self) -> pdarray:
        """
        Return the permutation that groups the array, placing equivalent
        strings together. All instances of the same string are guaranteed to lie
        in one contiguous block of the permuted array, but the blocks are not
        necessarily ordered.

        Returns
        -------
        pdarray
            The permutation that groups the array by value

        See Also
        --------
        GroupBy, unique

        Notes
        -----
        If the arkouda server is compiled with "-sSegmentedString.useHash=true",
        then arkouda uses 128-bit hash values to group strings, rather than sorting
        the strings directly. This method is fast, but the resulting permutation
        merely groups equivalent strings and does not sort them. If the "useHash"
        parameter is false, then a full sort is performed.

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error in executing group request or
            creating the pdarray encapsulating the return message
        """
        from arkouda.client import generic_msg

        return create_pdarray(
            generic_msg(cmd="segmentedGroup", args={"objType": self.objType, "obj": self.entry})
        )

    def _get_grouping_keys(self) -> List[Strings]:
        """
        Private method for generating grouping keys used by GroupBy.

        API: this method must be defined by all groupable arrays, and it
        must return a list of arrays that can be (co)argsorted.
        """
        return [self]

    def flatten(self) -> Strings:
        """
        Return a copy of the array collapsed into one dimension.

        Returns
        -------
        A copy of the input array, flattened to one dimension.

        Note
        ----
        As multidimensional Strings are currently supported,
        flatten on a Strings object will always return itself.
        """
        return self

    def to_ndarray(self) -> np.ndarray:
        """
        Convert the array to a np.ndarray, transferring array data from the
        arkouda server to Python. If the array exceeds a built-in size limit,
        a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            A numpy ndarray with the same strings as this array

        Notes
        -----
        The number of bytes in the array cannot exceed ``ak.client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.client.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        array()
        tolist()

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.array(["hello", "my", "world"])
        >>> a.to_ndarray()
        array(['hello', 'my', 'world'], dtype='<U5')
        >>> type(a.to_ndarray())
        <class 'numpy.ndarray'>
        """
        # Get offsets and append total bytes for length calculation
        npoffsets = np.hstack((self._comp_to_ndarray("offsets"), np.array([self.nbytes])))
        # Get contents of strings (will error if too large)
        npvalues = self._comp_to_ndarray("values")
        # Compute lengths, discounting null terminators
        lengths = np.diff(npoffsets) - 1
        # Numpy dtype is based on max string length
        dt = f"<U{lengths.max() if len(lengths) > 0 else 1}"
        res = np.empty(self.size, dtype=dt)
        # Form a string from each segment and store in numpy array
        for i, (o, ln) in enumerate(zip(npoffsets, lengths)):
            res[i] = np.str_(codecs.decode(b"".join(npvalues[o : o + ln])))
        return res

    def tolist(self) -> List[str]:
        """
        Convert the SegString to a list, transferring data from the
        arkouda server to Python. If the SegString exceeds a built-in size limit,
        a RuntimeError is raised.

        Returns
        -------
        List[str]
            A list with the same strings as this SegString

        Notes
        -----
        The number of bytes in the array cannot exceed ``ak.client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.client.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        to_ndarray()

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.array(["hello", "my", "world"])
        >>> a.tolist()
        ['hello', 'my', 'world']
        >>> type(a.tolist())
        <class 'list'>
        """
        return cast(List[str], self.to_ndarray().tolist())

    def _comp_to_ndarray(self, comp: str) -> np.ndarray:
        """
        Return a NumPy ndarray representing one component of the string structure.

        Parameters
        ----------
        comp : str
            The strings component to request

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
        """
        from arkouda.client import generic_msg, maxTransferBytes

        # Total number of bytes in the array data
        array_bytes = (
            self.size * arkouda.numpy.dtypes.dtype(arkouda.numpy.dtypes.int64).itemsize
            if comp == "offsets"
            else self.nbytes * arkouda.numpy.dtypes.dtype(arkouda.numpy.dtypes.uint8).itemsize
        )

        # Guard against overflowing client memory
        if array_bytes > maxTransferBytes:
            raise RuntimeError(
                "Array exceeds allowed size for transfer. Increase ak.client.maxTransferBytes to allow"
            )
        # The reply from the server will be a bytes object
        rep_msg = generic_msg(
            cmd=CMD_TO_NDARRAY, args={"obj": self.entry, "comp": comp}, recv_binary=True
        )

        # Make sure the received data has the expected length
        if len(rep_msg) != array_bytes:
            raise RuntimeError(f"Expected {array_bytes} bytes but received {len(rep_msg)}")

        # The server sends us native-endian bytes so we need to account for that.
        # Since bytes are immutable, we need to copy the np array to be mutable
        dt: np.dtype = np.dtype(np.int64) if comp == "offsets" else np.dtype(np.uint8)
        if arkouda.numpy.dtypes.get_server_byteorder() == "big":
            dt = dt.newbyteorder(">")
        else:
            dt = dt.newbyteorder("<")
        return (
            np.frombuffer(rep_msg.encode("utf_8"), dt).copy()
            if isinstance(rep_msg, str)
            else np.frombuffer(rep_msg, dt).copy()
        )

    def astype(self, dtype: Union[np.dtype, str]) -> pdarray:
        """
        Cast values of Strings object to provided dtype.

        Parameters
        ----------
        dtype: np.dtype or str
            Dtype to cast to

        Returns
        -------
        pdarray
            An arkouda pdarray with values converted to the specified data type

        Notes
        -----
        This is essentially shorthand for ak.cast(x, '<dtype>') where x is a pdarray.
        """
        from arkouda.numpy import cast as akcast

        return akcast(self, dtype)

    def to_parquet(
        self,
        prefix_path: str,
        dataset: str = "strings_array",
        mode: Literal["truncate", "append"] = "truncate",
        compression: Optional[Literal["snappy", "gzip", "brotli", "zstd", "lz4"]] = None,
    ) -> str:
        """
        Save the Strings object to Parquet. The result is a collection of files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str, default="strings_array"
            Name of the dataset to create in files (must not already exist)
        mode : {"truncate", "append"}, default = "truncate"
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.
        compression : {"snappy", "gzip", "brotli", "zstd", "lz4"}, optional
            Sets the compression type used with Parquet files

        Returns
        -------
        str
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
        """
        from arkouda.client import generic_msg
        from arkouda.pandas.io import _mode_str_to_int

        return cast(
            str,
            generic_msg(
                "writeParquet",
                {
                    "values": self.entry,
                    "dset": dataset,
                    "mode": _mode_str_to_int(mode),
                    "prefix": prefix_path,
                    "objType": "strings",
                    "dtype": self.dtype,
                    "compression": compression,
                },
            ),
        )

    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "strings_array",
        mode: Literal["truncate", "append"] = "truncate",
        save_offsets: bool = True,
        file_type: Literal["single", "distribute"] = "distribute",
    ) -> str:
        """
        Save the Strings object to HDF5.
        The object can be saved to a collection of files or single file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str, default="strings_array"
            The name of the Strings dataset to be written, defaults to strings_array
        mode : {"truncate", "append"}, default = "truncate"
            By default, truncate (overwrite) output files, if they exist.
            If 'append', create a new Strings dataset within existing files.
        save_offsets : bool, default=True
            Defaults to True which will instruct the server to save the offsets array to HDF5
            If False the offsets array will not be save and will be derived from the string values
            upon load/read.
        file_type : {"single", "distribute"}, default = "distribute"
            Default: Distribute
            Distribute the dataset over a file per locale.
            Single file will save the dataset to one file

        Returns
        -------
        str
            String message indicating result of save operation

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray

        Notes
        -----
        - Parquet files do not store the segments, only the values.
        - Strings state is saved as two datasets within an hdf5 group:
          one for the string characters and one for the
          segments corresponding to the start of each string
        - the hdf5 group is named via the dataset parameter.
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

        See Also
        --------
        to_hdf
        """
        from arkouda.client import generic_msg
        from arkouda.pandas.io import _file_type_to_int, _mode_str_to_int

        return cast(
            str,
            generic_msg(
                "tohdf",
                {
                    "values": self.entry,
                    "dset": dataset,
                    "write_mode": _mode_str_to_int(mode),
                    "filename": prefix_path,
                    "dtype": self.dtype,
                    "save_offsets": save_offsets,
                    "objType": "strings",
                    "file_format": _file_type_to_int(file_type),
                },
            ),
        )

    def update_hdf(
        self,
        prefix_path: str,
        dataset: str = "strings_array",
        save_offsets: bool = True,
        repack: bool = True,
    ) -> str:
        """
        Overwrite the dataset with the name provided with this Strings object.

        If the dataset does not exist it is added.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str, default="strings_array"
            Name of the dataset to create in files
        save_offsets : bool, default=True
            Defaults to True which will instruct the server to save the offsets array to HDF5
            If False the offsets array will not be save and will be derived from the string values
            upon load/read.
        repack : bool, default=True
            Default: True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Returns
        -------
        str
            success message if successful

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the Strings object

        Notes
        -----
        - If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        - If the dataset provided does not exist, it will be added
        """
        from arkouda.client import generic_msg
        from arkouda.pandas.io import _file_type_to_int, _get_hdf_filetype, _mode_str_to_int, _repack_hdf

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        msg = generic_msg(
            cmd="tohdf",
            args={
                "values": self,
                "dset": dataset,
                "write_mode": _mode_str_to_int("append"),
                "filename": prefix_path,
                "dtype": self.dtype,
                "save_offsets": save_offsets,
                "objType": "strings",
                "file_format": _file_type_to_int(file_type),
                "overwrite": True,
            },
        )

        if repack:
            _repack_hdf(prefix_path)

        return cast(str, msg)

    @typechecked
    def to_csv(
        self,
        prefix_path: str,
        dataset: str = "strings_array",
        col_delim: str = ",",
        overwrite: bool = False,
    ) -> str:
        r"""
        Write Strings to CSV file(s). File will contain a single column with the Strings data.
        All CSV Files written by Arkouda include a header denoting data types of the columns.
        Unlike other file formats, CSV files store Strings as their UTF-8 format instead of storing
        bytes as uint(8).

        Parameters
        ----------
        prefix_path: str
            The filename prefix to be used for saving files. Files will have _LOCALE#### appended
            when they are written to disk.
        dataset : str, default="strings_array"
            Column name to save the Strings under. Defaults to "strings_array".
        col_delim : str, default=","
            Defaults to ",". Value to be used to separate columns within the file.
            Please be sure that the value used DOES NOT appear in your dataset.
        overwrite : bool, default=False
            Defaults to False. If True, any existing files matching your provided prefix_path will
            be overwritten. If False, an error will be returned if existing files are found.

        Returns
        -------
        str
            response message

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
        -----
        - CSV format is not currently supported by load/load_all operations
        - The column delimiter is expected to be the same for column names and data
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline (``\\n``) at this time.
        """
        from arkouda.client import generic_msg

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

    def _list_component_names(self) -> List[str]:
        """
        Return a list of all component names.

        Returns
        -------
        List[str]
            List of all component names
        """
        return list(itertools.chain.from_iterable([self.entry._list_component_names()]))

    def info(self) -> str:
        """
        Return a JSON formatted string containing information about all components of self.

        Returns
        -------
        str
            JSON string containing information about all components of self
        """
        return information(self._list_component_names())

    def pretty_print_info(self) -> None:
        """Print information about all components of self in a human readable format."""
        self.entry.pretty_print_info()

    @typechecked
    def register(self, user_defined_name: str) -> Strings:
        """
        Register this Strings object with a user defined name in the arkouda server
        so it can be attached to later using Strings.attach().

        This is an in-place operation, registering a Strings object more than once will
        update the name in the registry and remove the previously registered name.
        A name can only be registered to one object at a time.

        Parameters
        ----------
        user_defined_name : str
            user defined name which the Strings object is to be registered under

        Returns
        -------
        Strings
            The same Strings object which is now registered with the arkouda server and
            has an updated name.
            This is an in-place modification, the original is returned to support a
            fluid programming style.
            Please note you cannot register two different objects with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the Strings object with the user_defined_name
            If the user is attempting to register more than one object with the same name,
            the former should be unregistered first to free up the registration name.

        See Also
        --------
        attach, unregister

        Notes
        -----
        Registered names/Strings objects in the server are immune to deletion
        until they are unregistered.

        """
        from arkouda.client import generic_msg

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
        Unregister a Strings object in the arkouda server which was previously
        registered using register() and/or attached to using attach().

        Raises
        ------
        RuntimeError
            Raised if the server could not find the internal name/symbol to remove

        See Also
        --------
        register, attach

        Notes
        -----
        Registered names/Strings objects in the server are immune to deletion until
        they are unregistered.

        """
        from arkouda.numpy.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    def is_registered(self) -> np.bool_:
        """
        Return True iff the object is contained in the registry.

        Returns
        -------
        bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RuntimeError
            Raised if there's a server-side error thrown
        """
        from arkouda.numpy.util import is_registered

        if self.registered_name is None:
            return np.bool_(is_registered(self.name, as_component=True))
        else:
            return np.bool_(is_registered(self.registered_name))

    def transfer(self, hostname: str, port: int_scalars) -> Union[str, memoryview]:
        """
        Send a Strings object to a different Arkouda server.

        Parameters
        ----------
        hostname : str
            The hostname where the Arkouda server intended to
            receive the Strings object is running.
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
        str
            A message indicating a complete transfer

        Raises
        ------
        ValueError
            Raised if the op is not within the pdarray.BinOps set
        TypeError
            Raised if other is not a pdarray or the pdarray.dtype is not
            a supported dtype
        """
        from arkouda.client import generic_msg

        # hostname is the hostname to send to
        return generic_msg(
            cmd="sendArray",
            args={"values": self.entry, "hostname": hostname, "port": port, "objType": "strings"},
        )

    @staticmethod
    def concatenate_uniquely(strings: List[Strings]) -> Strings:
        """
        Concatenates a list of Strings into a single Strings object
        containing only unique strings. Order may not be preserved.

        Parameters
        ----------
        strings : List[Strings]
            List of segmented string objects to concatenate.

        Returns
        -------
        Strings
            A new Strings object containing the unique values.
        """
        from arkouda.client import generic_msg

        if not strings:
            raise ValueError("Must provide at least one Strings object")

        # Extract name of each SegmentedString
        names = [s.name for s in strings]

        # Send the command to the server
        rep_msg = generic_msg(
            cmd="concatenateUniquely",
            args={
                "names": names,
            },
        )

        return Strings.from_return_msg(cast(str, rep_msg))

    def argsort(
        self,
        algorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD,
        ascending: bool = True,
    ) -> pdarray:
        """
        Return the permutation that sorts the Strings.

        Parameters
        ----------
        algorithm : SortingAlgorithm, default SortingAlgorithm.RadixSortLSD
            The algorithm to use for sorting.
        ascending : bool, default True
            Whether to sort in ascending order.

        Returns
        -------
        pdarray
            The indices that sort the Strings.

        """
        from arkouda.client import generic_msg
        from arkouda.numpy.manipulation_functions import flip
        from arkouda.numpy.pdarraycreation import zeros

        if self.size == 0:
            return zeros(0, dtype=akint64)  # Strings always maps to int64 indices

        repMsg = generic_msg(
            cmd="argsortStrings",
            args={
                "name": self.entry.name,
                "algoName": algorithm.name,
            },
        )

        sorted_array = create_pdarray(cast(str, repMsg))
        return sorted_array if ascending else flip(sorted_array)
