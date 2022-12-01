from __future__ import annotations

import codecs
import itertools
import re
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np  # type: ignore
from typeguard import typechecked
from warnings import warn

import arkouda.dtypes
from arkouda.client import generic_msg
from arkouda.dtypes import (
    NUMBER_FORMAT_STRINGS,
    int_scalars,
    npstr,
    resolve_scalar_dtype,
    str_scalars,
    translate_np_dtype,
)
from arkouda.infoclass import information
from arkouda.logger import getArkoudaLogger
from arkouda.match import Match, MatchType
from arkouda.pdarrayclass import (
    create_pdarray,
    parse_single_value,
    pdarray,
    unregister_pdarray_by_name,
)

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
    dtype : dtype
        The dtype is ak.str
    logger : ArkoudaLogger
        Used for all logging operations

    Notes
    -----
    Strings is composed of two pdarrays: (1) offsets, which contains the
    starting indices for each string and (2) bytes, which contains the
    raw bytes of all strings, delimited by nulls.
    """

    BinOps = frozenset(["==", "!="])
    objtype = "str"

    @staticmethod
    def from_return_msg(rep_msg: str) -> Strings:
        """
        Factory method for creating a Strings object from an Arkouda server
        response message

        Parameters
        ----------
        rep_msg : str
            Server response message currently of form
            `created name type size ndim shape itemsize+created bytes.size 1234`

        Returns
        -------
        Strings
            object representing a segmented strings array on the server

        Raises
        ------
        RuntimeError
            Raised if there's an error converting a server-returned str-descriptor

        Notes
        -----
        We really don't have an itemsize because these are variable length strings.
        In the future we could probably use this position to store the total bytes.
        """
        left, right = cast(str, rep_msg).split("+")
        bytes_size: int_scalars = int(right.split()[-1])
        return Strings(create_pdarray(left), bytes_size)

    @staticmethod
    def from_parts(offset_attrib: Union[pdarray, str], bytes_attrib: Union[pdarray, str]) -> Strings:
        """
        Factory method for creating a Strings object from an Arkouda server
        response where the arrays are separate components.

        Parameters
        ----------
        offset_attrib : Union[pdarray, str]
            the array containing the offsets
        bytes_attrib : Union[pdarray, str]
            the array containing the string values

        Returns
        -------
        Strings
            object representing a segmented strings array on the server

        Raises
        ------
        RuntimeError
            Raised if there's an error converting a server-returned str-descriptor

        Notes
        -----
        This factory method is used when we construct the parts of a Strings
        object on the client side and transfer the offsets & bytes separately
        to the server.  This results in two entries in the symbol table and we
        need to instruct the server to assemble the into a composite entity.
        """
        if not isinstance(offset_attrib, pdarray):
            try:
                offset_attrib = create_pdarray(offset_attrib)
            except Exception as e:
                raise RuntimeError(e)
        if not isinstance(bytes_attrib, pdarray):
            try:
                bytes_attrib = create_pdarray(bytes_attrib)
            except Exception as e:
                raise RuntimeError(e)
        # Now we have two pdarray objects
        response = cast(
            str, generic_msg(cmd=CMD_ASSEMBLE, args={"offsets": offset_attrib, "values": bytes_attrib})
        )
        return Strings.from_return_msg(response)

    def __init__(self, strings_pdarray: pdarray, bytes_size: int_scalars) -> None:
        """
        Initializes the Strings instance by setting all instance
        attributes, some of which are derived from the array parameters.

        Parameters
        ----------
        strings_pdarray : pdarray
            the array containing the meta-info on a server side strings object
        bytes_size : int_scalars
            length of the bytes array contained on the server aka total bytes

        Returns
        -------
        None

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
        try:
            self.size = self.entry.size
            self.nbytes = bytes_size  # This is a deficiency of server GenSymEntry right now
            self.ndim = self.entry.ndim
            self.shape = self.entry.shape
            self.name: Optional[str] = self.entry.name
        except Exception as e:
            raise ValueError(e)

        self.dtype = npstr
        self._regex_dict: Dict = dict()
        self.logger = getArkoudaLogger(name=__class__.__name__)  # type: ignore

    """
    NOTE:
         The Strings.__del__() method should NOT be implemented.
         Python will invoke the __del__() of any components by default.
         Overriding this default behavior with an explicitly specified Strings.__del__() method may
         introduce errors in the event that additional components are added to Strings
         and the method is not updated.
         By allowing Python's garbage collecting to handle this automatically, we avoid extra maintenance
    """

    def __iter__(self):
        raise NotImplementedError(
            "Strings does not support iteration. To force data transfer from server, use to_ndarray"
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
        Executes the requested binop on this Strings instance and the
        parameter Strings object and returns the results within
        a pdarray object.

        Parameters
        ----------
        other : Strings, str_scalars
            the other object is a Strings object
        op : str
            name of the binary operation to be performed

        Returns
        -------
        pdarray
            encapsulating the results of the requested binop

        Raises
        -----
        ValueError
            Raised if (1) the op is not in the self.BinOps set, or (2) if the
            sizes of this and the other instance don't match, or (3) the other
            object is not a Strings object
        RuntimeError
            Raised if a server-side error is thrown while executing the
            binary operation
        """
        if op not in self.BinOps:
            raise ValueError(f"Strings: unsupported operator: {op}")
        if isinstance(other, Strings):
            if self.size != other.size:
                raise ValueError(f"Strings: size mismatch {self.size} {other.size}")
            cmd = "segmentedBinopvv"
            args = {
                "op": op,
                "objType": self.objtype,
                "obj": self.entry,
                "otherType": other.objtype,
                "other": other.entry,
                "left": False,  # placeholder for stick
                "delim": "",  # placeholder for stick
            }
        elif resolve_scalar_dtype(other) == "str":
            cmd = "segmentedBinopvs"
            args = {
                "op": op,
                "objType": self.objtype,
                "obj": self.entry,
                "otherType": self.objtype,
                "other": other,
            }
        else:
            raise ValueError(
                f"Strings: {op} not supported between Strings and {other.__class__.__name__}"
            )
        return create_pdarray(generic_msg(cmd=cmd, args=args))

    def __eq__(self, other) -> bool:
        return self._binop(other, "==")

    def __ne__(self, other) -> bool:
        return self._binop(cast(Strings, other), "!=")

    def __getitem__(self, key):
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
                        "objType": self.objtype,
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
                    "objType": self.objtype,
                    "obj": self.entry,
                    "dtype": self.entry.dtype,
                    "key": [start, stop, stride],
                },
            )
            return Strings.from_return_msg(repMsg)
        elif isinstance(key, pdarray):
            kind, _ = translate_np_dtype(key.dtype)
            if kind not in ("bool", "int", "uint"):
                raise TypeError(f"unsupported pdarray index type {key.dtype}")
            if kind == "bool" and self.size != key.size:
                raise ValueError(f"size mismatch {self.size} {key.size}")
            repMsg = generic_msg(
                cmd="segmentedIndex",
                args={
                    "subcmd": "pdarrayIndex",
                    "objType": self.objtype,
                    "dtype": self.entry.dtype,
                    "obj": self.entry,
                    "key": key,
                },
            )
            return Strings.from_return_msg(repMsg)
        else:
            raise TypeError(f"unsupported pdarray index type {key.__class__.__name__}")

    def get_lengths(self) -> pdarray:
        """
        Return the length of each string in the array.

        Returns
        -------
        pdarray, int
            The length of each string

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown
        """
        return create_pdarray(
            generic_msg(cmd="segmentLengths", args={"objType": self.objtype, "obj": self.entry})
        )

    # TODO: Store the current encoding for fromEncoding in strings object?
    def encode(self, toEncoding, fromEncoding="UTF-8"):
        """
        Return a new strings object in `toEncoding`, expecting that the
        current Strings is encoded in `fromEncoding`

        Returns
        -------
        Strings
            A new Strings object in `toEncoding`

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown
        """
        rep_msg = generic_msg(
            cmd="encode",
            args={
                "toEncoding": toEncoding,
                "fromEncoding": fromEncoding,
                "obj": self.entry,
            },
        )
        return Strings.from_return_msg(rep_msg)

    def decode(self, fromEncoding, toEncoding="UTF-8"):
        """
        Return a new strings object in `fromEncoding`, expecting that the
        current Strings is encoded in `toEncoding`

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
    def to_lower(self) -> Strings:
        """
        Returns a new Strings with all uppercase characters from the original replaced with
        their lowercase equivalent

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
        Strings.to_upper

        Examples
        --------
        >>> strings = ak.array([f'StrINgS {i}' for i in range(5)])
        >>> strings
        array(['StrINgS 0', 'StrINgS 1', 'StrINgS 2', 'StrINgS 3', 'StrINgS 4'])
        >>> strings.to_lower()
        array(['strings 0', 'strings 1', 'strings 2', 'strings 3', 'strings 4'])
        """
        rep_msg = generic_msg(
            cmd="caseChange", args={"subcmd": "toLower", "objType": self.objtype, "obj": self.entry}
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    @typechecked
    def to_upper(self) -> Strings:
        """
        Returns a new Strings with all lowercase characters from the original replaced with
        their uppercase equivalent

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
        Strings.to_lower

        Examples
        --------
        >>> strings = ak.array([f'StrINgS {i}' for i in range(5)])
        >>> strings
        array(['StrINgS 0', 'StrINgS 1', 'StrINgS 2', 'StrINgS 3', 'StrINgS 4'])
        >>> strings.to_upper()
        array(['STRINGS 0', 'STRINGS 1', 'STRINGS 2', 'STRINGS 3', 'STRINGS 4'])
        """
        rep_msg = generic_msg(
            cmd="caseChange", args={"subcmd": "toUpper", "objType": self.objtype, "obj": self.entry}
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    @typechecked
    def to_title(self) -> Strings:
        """
        Returns a new Strings from the original replaced with their titlecase equivalent

        Returns
        -------
        Strings
            Strings from the original replaced with their titlecase equivalent

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.to_lower
        String.to_upper

        Examples
        --------
        >>> strings = ak.array([f'StrINgS {i}' for i in range(5)])
        >>> strings
        array(['StrINgS 0', 'StrINgS 1', 'StrINgS 2', 'StrINgS 3', 'StrINgS 4'])
        >>> strings.to_title()
        array(['Strings 0', 'Strings 1', 'Strings 2', 'Strings 3', 'Strings 4'])
        """
        rep_msg = generic_msg(
            cmd="caseChange", args={"subcmd": "toTitle", "objType": self.objtype, "obj": self.entry}
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    @typechecked
    def is_lower(self) -> pdarray:
        """
        Returns a boolean pdarray where index i indicates whether string i of the
        Strings is entirely lowercase

        Returns
        -------
        pdarray, bool
            True for elements that are entirely lowercase, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.is_upper

        Examples
        --------
        >>> lower = ak.array([f'strings {i}' for i in range(3)])
        >>> upper = ak.array([f'STRINGS {i}' for i in range(3)])
        >>> strings = ak.concatenate([lower, upper])
        >>> strings
        array(['strings 0', 'strings 1', 'strings 2', 'STRINGS 0', 'STRINGS 1', 'STRINGS 2'])
        >>> strings.is_lower()
        array([True True True False False False])
        """
        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isLower", "objType": self.objtype, "obj": self.entry}
            )
        )

    @typechecked
    def is_upper(self) -> pdarray:
        """
        Returns a boolean pdarray where index i indicates whether string i of the
        Strings is entirely uppercase

        Returns
        -------
        pdarray, bool
            True for elements that are entirely uppercase, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.is_lower

        Examples
        --------
        >>> lower = ak.array([f'strings {i}' for i in range(3)])
        >>> upper = ak.array([f'STRINGS {i}' for i in range(3)])
        >>> strings = ak.concatenate([lower, upper])
        >>> strings
        array(['strings 0', 'strings 1', 'strings 2', 'STRINGS 0', 'STRINGS 1', 'STRINGS 2'])
        >>> strings.is_upper()
        array([False False False True True True])
        """
        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isUpper", "objType": self.objtype, "obj": self.entry}
            )
        )

    @typechecked
    def is_title(self) -> pdarray:
        """
        Returns a boolean pdarray where index i indicates whether string i of the
        Strings is titlecase

        Returns
        -------
        pdarray, bool
            True for elements that are titlecase, False otherwise

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Strings.is_lower
        Strings.is_upper

        Examples
        --------
        >>> mixed = ak.array([f'sTrINgs {i}' for i in range(3)])
        >>> title = ak.array([f'Strings {i}' for i in range(3)])
        >>> strings = ak.concatenate([mixed, title])
        >>> strings
        array(['sTrINgs 0', 'sTrINgs 1', 'sTrINgs 2', 'Strings 0', 'Strings 1', 'Strings 2'])
        >>> strings.is_title()
        array([False False False True True True])
        """
        return create_pdarray(
            generic_msg(
                cmd="checkChars", args={"subcmd": "isTitle", "objType": self.objtype, "obj": self.entry}
            )
        )

    @typechecked
    def strip(self, chars: Optional[Union[bytes, str_scalars]] = "") -> Strings:
        """
        Returns a new Strings object with all leading and trailing occurrences of characters contained
        in chars removed. The chars argument is a string specifying the set of characters to be removed.
        If omitted, the chars argument defaults to removing whitespace. The chars argument is not a
        prefix or suffix; rather, all combinations of its values are stripped.

        Parameters
        ----------
        chars : the set of characters to be removed

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
        >>> strings = ak.array(['Strings ', '  StringS  ', 'StringS   '])
        >>> s = strings.strip()
        >>> s
        array(['Strings', 'StringS', 'StringS'])

        >>> strings = ak.array(['Strings 1', '1 StringS  ', '  1StringS  12 '])
        >>> s = strings.strip(' 12')
        >>> s
        array(['Strings', 'StringS', 'StringS'])
        """
        if isinstance(chars, bytes):
            chars = chars.decode()
        rep_msg = generic_msg(
            cmd="segmentedStrip", args={"objType": self.objtype, "name": self.entry, "chars": chars}
        )
        return Strings.from_return_msg(cast(str, rep_msg))

    @typechecked
    def cached_regex_patterns(self) -> List:
        """
        Returns the regex patterns for which Match objects have been cached
        """
        return list(self._regex_dict.keys())

    @typechecked
    def purge_cached_regex_patterns(self) -> None:
        """
        purges cached regex patterns
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
        internal function to fetch cached Matcher objects
        """
        from arkouda.matcher import Matcher

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
        """
        Finds pattern matches and returns pdarrays containing the number, start postitions,
        and lengths of matches

        Parameters
        ----------
        pattern: str_scalars
            The regex pattern used to find matches

        Returns
        -------
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
        >>> strings = ak.array([f'{i} string {i}' for i in range(1, 6)])
        >>> num_matches, starts, lens = strings.find_locations('\\d')
        >>> num_matches
        array([2, 2, 2, 2, 2])
        >>> starts
        array([0, 9, 0, 9, 0, 9, 0, 9, 0, 9])
        >>> lens
        array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        """
        matcher = self._get_matcher(pattern)
        matcher.find_locations()
        return matcher.num_matches, matcher.starts, matcher.lengths

    @typechecked
    def search(self, pattern: Union[bytes, str_scalars]) -> Match:
        """
        Returns a match object with the first location in each element where pattern produces a match.
        Elements match if any part of the string matches the regular expression pattern

        Parameters
        ----------
        pattern: str
            Regex used to find matches

        Returns
        -------
        Match
            Match object where elements match if any part of the string matches the
            regular expression pattern

        Examples
        --------
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.search('_+')
        <ak.Match object: matched=True, span=(1, 2); matched=True, span=(0, 4);
        matched=False; matched=True, span=(0, 2); matched=False>
        """
        return self._get_matcher(pattern).get_match(MatchType.SEARCH, self)

    @typechecked
    def match(self, pattern: Union[bytes, str_scalars]) -> Match:
        """
        Returns a match object where elements match only if the beginning of the string matches the
        regular expression pattern

        Parameters
        ----------
        pattern: str
            Regex used to find matches

        Returns
        -------
        Match
            Match object where elements match only if the beginning of the string matches the
            regular expression pattern

        Examples
        --------
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.match('_+')
        <ak.Match object: matched=False; matched=True, span=(0, 4); matched=False;
        matched=True, span=(0, 2); matched=False>
        """
        return self._get_matcher(pattern).get_match(MatchType.MATCH, self)

    @typechecked()
    def fullmatch(self, pattern: Union[bytes, str_scalars]) -> Match:
        """
        Returns a match object where elements match only if the whole string matches the
        regular expression pattern

        Parameters
        ----------
        pattern: str
            Regex used to find matches

        Returns
        -------
        Match
            Match object where elements match only if the whole string matches the
            regular expression pattern

        Examples
        --------
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.fullmatch('_+')
        <ak.Match object: matched=False; matched=True, span=(0, 4); matched=False;
        matched=False; matched=False>
        """
        return self._get_matcher(pattern).get_match(MatchType.FULLMATCH, self)

    @typechecked()
    def split(
        self, pattern: Union[bytes, str_scalars], maxsplit: int = 0, return_segments: bool = False
    ) -> Union[Strings, Tuple]:
        """
        Returns a new Strings split by the occurrences of pattern.
        If maxsplit is nonzero, at most maxsplit splits occur

        Parameters
        ----------
        pattern: str
            Regex used to split strings into substrings
        maxsplit: int
            The max number of pattern match occurences in each element to split.
            The default maxsplit=0 splits on all occurences
        return_segments: bool
            If True, return mapping of original strings to first substring
            in return array.

        Returns
        -------
        Strings
            Substrings with pattern matches removed
        pdarray, int64 (optional)
            For each original string, the index of first corresponding substring
            in the return array

        Examples
        --------
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.split('_+', maxsplit=2, return_segments=True)
        (array(['1', '2', '', '', '', '3', '', '4', '5____6___7', '']), array([0 3 5 6 9]))
        """
        return self._get_matcher(pattern).split(maxsplit, return_segments)

    @typechecked
    def findall(
        self, pattern: Union[bytes, str_scalars], return_match_origins: bool = False
    ) -> Union[Strings, Tuple]:
        """
        Return a new Strings containg all non-overlapping matches of pattern

        Parameters
        ----------
        pattern: str_scalars
            Regex used to find matches
        return_match_origins: bool
            If True, return a pdarray containing the index of the original string each
            pattern match is from

        Returns
        -------
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
        If count is nonzero, at most count substitutions occur

        Parameters
        ----------
        pattern: str_scalars
            The regex to substitue
        repl: str_scalars
            The substring to replace pattern matches with
        count: int
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
    ) -> Tuple:
        """
        Perform the same operation as sub(), but return a tuple (new_Strings, number_of_substitions)

        Parameters
        ----------
        pattern: str_scalars
            The regex to substitue
        repl: str_scalars
            The substring to replace pattern matches with
        count: int
            The max number of pattern match occurences in each element to replace.
            The default count=0 replaces all occurences of pattern with repl

        Returns
        -------
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
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.subn(pattern='_+', repl='-', count=2)
        (array(['1-2-', '-', '3', '-4-5____6___7', '']), array([2 1 0 2 0]))
        """
        if isinstance(repl, bytes):
            repl = repl.decode()
        return self._get_matcher(pattern).sub(repl, count, return_num_subs=True)

    @typechecked
    def contains(self, substr: Union[bytes, str_scalars], regex: bool = False) -> pdarray:
        """
        Check whether each element contains the given substring.

        Parameters
        ----------
        substr: str_scalars
            The substring in the form of string or byte array to search for
        regex: bool
            Indicates whether substr is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        pdarray, bool
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
        >>> strings = ak.array([f'{i} string {i}' for i in range(1, 6)])
        >>> strings
        array(['1 string 1', '2 string 2', '3 string 3', '4 string 4', '5 string 5'])
        >>> strings.contains('string')
        array([True, True, True, True, True])
        >>> strings.contains('string \\d', regex=True)
        array([True, True, True, True, True])
        """
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
                args={"objType": self.objtype, "obj": self.entry, "valType": "str", "val": substr},
            )
        )

    @typechecked
    def startswith(self, substr: Union[bytes, str_scalars], regex: bool = False) -> pdarray:
        """
        Check whether each element starts with the given substring.

        Parameters
        ----------
        substr: Union[bytes, str_scalars]
            The prefix to search for
        regex: bool
            Indicates whether substr is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        pdarray, bool
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
        >>> strings_end = ak.array([f'string {i}' for i in range(1, 6)])
        >>> strings_end
        array(['string 1', 'string 2', 'string 3', 'string 4', 'string 5'])
        >>> strings_end.startswith('string')
        array([True, True, True, True, True])
        >>> strings_start = ak.array([f'{i} string' for i in range(1,6)])
        >>> strings_start
        array(['1 string', '2 string', '3 string', '4 string', '5 string'])
        >>> strings_start.startswith('\\d str', regex = True)
        array([True, True, True, True, True])
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
        """
        Check whether each element ends with the given substring.

        Parameters
        ----------
        substr: Union[bytes, str_scalars]
            The suffix to search for
        regex: bool
            Indicates whether substr is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        pdarray, bool
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
        >>> strings_start = ak.array([f'{i} string' for i in range(1,6)])
        >>> strings_start
        array(['1 string', '2 string', '3 string', '4 string', '5 string'])
        >>> strings_start.endswith('ing')
        array([True, True, True, True, True])
        >>> strings_end = ak.array([f'string {i}' for i in range(1, 6)])
        >>> strings_end
        array(['string 1', 'string 2', 'string 3', 'string 4', 'string 5'])
        >>> strings_end.endswith('ing \\d', regex = True)
        array([True, True, True, True, True])
        """
        if isinstance(substr, bytes):
            substr = substr.decode()
        if not regex:
            substr = re.escape(substr)
        self._empty_pattern_verification(substr)
        return self.contains(substr + "$", regex=True)

    def flatten(
        self, delimiter: str, return_segments: bool = False, regex: bool = False
    ) -> Union[Strings, Tuple]:
        """Unpack delimiter-joined substrings into a flat array.

        Parameters
        ----------
        delimiter: str
            Characters used to split strings into substrings
        return_segments: bool
            If True, also return mapping of original strings to first substring
            in return array.
        regex: bool
            Indicates whether delimiter is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
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
        >>> orig = ak.array(['one|two', 'three|four|five', 'six'])
        >>> orig.flatten('|')
        array(['one', 'two', 'three', 'four', 'five', 'six'])
        >>> flat, map = orig.flatten('|', return_segments=True)
        >>> map
        array([0, 2, 5])
        >>> under = ak.array(['one_two', 'three_____four____five', 'six'])
        >>> under_flat, under_map = under.flatten('_+', return_segments=True, regex=True)
        >>> under_flat
        array(['one', 'two', 'three', 'four', 'five', 'six'])
        >>> under_map
        array([0, 2, 5])
        """
        if regex:
            try:
                re.compile(delimiter)
            except Exception as e:
                raise ValueError(e)
            return self.split(delimiter, return_segments=return_segments)
        else:
            cmd = "segmentedFlatten"
            repMsg = cast(
                str,
                generic_msg(
                    cmd=cmd,
                    args={
                        "values": self.entry,
                        "objtype": self.objtype,
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
    ) -> Tuple:
        """
        Peel off one or more delimited fields from each string (similar
        to string.partition), returning two new arrays of strings.
        *Warning*: This function is experimental and not guaranteed to work.

        Parameters
        ----------
        delimiter: Union[bytes, str_scalars]
            The separator where the split will occur
        times: Union[int, np.int64]
            The number of times the delimiter is sought, i.e. skip over
            the first (times-1) delimiters
        includeDelimiter: bool
            If true, append the delimiter to the end of the first return
            array. By default, it is prepended to the beginning of the
            second return array.
        keepPartial: bool
            If true, a string that does not contain <times> instances of
            the delimiter will be returned in the first array. By default,
            such strings are returned in the second array.
        fromRight: bool
            If true, peel from the right instead of the left (see also rpeel)
        regex: bool
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
                "objType": self.objtype,
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
    ):
        """
        Peel off one or more delimited fields from the end of each string
        (similar to string.rpartition), returning two new arrays of strings.
        *Warning*: This function is experimental and not guaranteed to work.

        Parameters
        ----------
        delimiter: Union[bytes, str_scalars]
            The separator where the split will occur
        times: Union[int, np.int64]
            The number of times the delimiter is sought, i.e. skip over
            the last (times-1) delimiters
        includeDelimiter: bool
            If true, prepend the delimiter to the start of the first return
            array. By default, it is appended to the end of the
            second return array.
        keepPartial: bool
            If true, a string that does not contain <times> instances of
            the delimiter will be returned in the second array. By default,
            such strings are returned in the first array.
        regex: bool
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
        >>> s = ak.array(['a.b', 'c.d', 'e.f.g'])
        >>> s.rpeel('.')
        (array(['a', 'c', 'e.f']), array(['b', 'd', 'g']))
        # Compared against peel
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
        delimiter : str
            String inserted between self and other
        toLeft : bool
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
        >>> s = ak.array(['a', 'c', 'e'])
        >>> t = ak.array(['b', 'd', 'f'])
        >>> s.stick(t, delimiter='.')
        array(['a.b', 'c.d', 'e.f'])
        """
        if isinstance(delimiter, bytes):
            delimiter = delimiter.decode()
        rep_msg = generic_msg(
            cmd="segmentedBinopvv",
            args={
                "op": "stick",
                "objType": self.objtype,
                "obj": self.entry,
                "otherType": other.objtype,
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
        delimiter : Union[bytes,str_scalars]
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
        Return the n-long prefix of each string, where possible

        Parameters
        ----------
        n : int
            Length of prefix
        return_origins : bool
            If True, return a logical index indicating which strings
            were long enough to return an n-prefix
        proper : bool
            If True, only return proper prefixes, i.e. from strings
            that are at least n+1 long. If False, allow the entire
            string to be returned as a prefix.

        Returns
        -------
        prefixes : Strings
            The array of n-character prefixes; the number of elements is the number of
            True values in the returned mask.
        origin_indices : pdarray, bool
            Boolean array that is True where the string was long enough to return
            an n-character prefix, False otherwise.
        """
        repMsg = cast(
            str,
            generic_msg(
                cmd="segmentedSubstring",
                args={
                    "objType": self.objtype,
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
        Return the n-long suffix of each string, where possible

        Parameters
        ----------
        n : int
            Length of suffix
        return_origins : bool
            If True, return a logical index indicating which strings
            were long enough to return an n-suffix
        proper : bool
            If True, only return proper suffixes, i.e. from strings
            that are at least n+1 long. If False, allow the entire
            string to be returned as a suffix.

        Returns
        -------
        suffixes : Strings
            The array of n-character suffixes; the number of elements is the number of
            True values in the returned mask.
        origin_indices : pdarray, bool
            Boolean array that is True where the string was long enough to return
            an n-character suffix, False otherwise.
        """
        repMsg = cast(
            str,
            generic_msg(
                cmd="segmentedSubstring",
                args={
                    "objType": self.objtype,
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
        # TODO fix this to return a single pdarray of hashes
        repMsg = generic_msg(cmd="segmentedHash", args={"objType": self.objtype, "obj": self.entry})
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
        return create_pdarray(
            generic_msg(cmd="segmentedGroup", args={"objType": self.objtype, "obj": self.entry})
        )

    def _get_grouping_keys(self) -> List[Strings]:
        """
        Private method for generating grouping keys used by GroupBy.

        API: this method must be defined by all groupable arrays, and it
        must return a list of arrays that can be (co)argsorted.
        """
        return [self]

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
        The number of bytes in the array cannot exceed ``arkouda.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        array()
        to_list()

        Examples
        --------
        >>> a = ak.array(["hello", "my", "world"])
        >>> a.to_ndarray()
        array(['hello', 'my', 'world'], dtype='<U5')
        >>> type(a.to_ndarray())
        numpy.ndarray
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
        for i, (o, l) in enumerate(zip(npoffsets, lengths)):
            res[i] = np.str_(codecs.decode(b"".join(npvalues[o : o + l])))
        return res

    def to_list(self) -> list:
        """
        Convert the SegString to a list, transferring data from the
        arkouda server to Python. If the SegString exceeds a built-in size limit,
        a RuntimeError is raised.

        Returns
        -------
        list
            A list with the same strings as this SegString

        Notes
        -----
        The number of bytes in the array cannot exceed ``arkouda.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        to_ndarray()

        Examples
        --------
        >>> a = ak.array(["hello", "my", "world"])
        >>> a.to_list()
        ['hello', 'my', 'world']
        >>> type(a.to_list())
        list
        """
        return self.to_ndarray().tolist()

    def _comp_to_ndarray(self, comp: str) -> np.ndarray:
        """
        This is an internal helper function to perform the to_ndarray for one
        of the string components.

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
        from arkouda.client import maxTransferBytes

        # Total number of bytes in the array data
        array_bytes = (
            self.size * arkouda.dtypes.int64.itemsize
            if comp == "offsets"
            else self.nbytes * arkouda.dtypes.uint8.itemsize
        )

        # Guard against overflowing client memory
        if array_bytes > maxTransferBytes:
            raise RuntimeError(
                "Array exceeds allowed size for transfer. Increase client.maxTransferBytes to allow"
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
        if arkouda.dtypes.get_server_byteorder() == "big":
            dt = dt.newbyteorder(">")
        else:
            dt = dt.newbyteorder("<")
        return (
            np.frombuffer(rep_msg.encode("utf_8"), dt).copy()
            if isinstance(rep_msg, str)
            else np.frombuffer(rep_msg, dt).copy()
        )

    def astype(self, dtype) -> pdarray:
        """
        Cast values of Strings object to provided dtype

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

    @typechecked
    def save(
        self,
        prefix_path: str,
        dataset: str = "strings_array",
        mode: str = "truncate",
        save_offsets: bool = True,
        compressed: bool = False,
        file_format: str = "HDF5",
        file_type: str = "distribute",
    ) -> str:
        """
        DEPRECATED
        Save the Strings object to HDF5 or Parquet. The result is a collection of
        files, one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the Strings array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            The name of the Strings dataset to be written, defaults to strings_array
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', create a new Strings dataset within existing files.
        save_offsets : bool
            Defaults to True which will instruct the server to save the offsets array to HDF5
            If False the offsets array will not be save and will be derived from the string values
            upon load/read. This is not supported for Parquet files.
        compressed : bool
            Defaults to False. When True, files will be written with Snappy compression
            and RLE bit packing. This is currently only supported on Parquet files and will
            not impact the generated files when writing HDF5 files.
        file_format : str
            By default, saved files will be written to the HDF5 file format. If
            'Parquet', the files will be written to the Parquet file format. This
            is case insensitive.
        file_type: str ("single" | "distribute")
            Default: Distribute
            Distribute the dataset over a file per locale.
            Single file will save the dataset to one file

        Returns
        -------
        String message indicating result of save operation

        Raises
        ------
        ValueError
            Raised if the lengths of columns and values differ, or the mode is
            neither 'truncate' nor 'append'
        TypeError
            Raised if prefix_path, dataset, or mode is not a str

        Notes
        -----
        Important implementation notes: (1) Strings state is saved as two datasets
        within an hdf5 group: one for the string characters and one for the
        segments corresponding to the start of each string, (2) the hdf5 group is named
        via the dataset parameter.
        """
        warn(
            "ak.Strings.save has been deprecated. Please use ak.Strings.to_hdf or ak.Strings.to_parquet",
            DeprecationWarning,
        )
        from arkouda.io import file_type_to_int, mode_str_to_int

        if file_format.lower() == "hdf5":
            args = {
                "values": self.entry,
                "dset": dataset,
                "write_mode": mode_str_to_int(mode),
                "filename": prefix_path,
                "dtype": self.dtype,
                "save_offsets": save_offsets,
                "compressed": compressed,
                "objType": "strings",
                "file_format": file_type_to_int(file_type),
            }
            return cast(str, generic_msg("tohdf", args))
        elif file_format.lower() == "parquet":
            args = {
                "values": self.entry,
                "dset": dataset,
                "mode": mode_str_to_int(mode),
                "prefix": prefix_path,
                "dtype": self.dtype,
                "save_offsets": save_offsets,
                "compressed": compressed,
            }
            return cast(str, generic_msg("writeParquet", args))
        else:
            raise ValueError("Supported file formats are 'HDF5' and 'Parquet'")

    def to_parquet(
        self,
        prefix_path: str,
        dataset: str = "strings_array",
        mode: str = "truncate",
        compressed: bool = False,
    ) -> str:
        """
        Save the Strings object to Parquet. The result is a collection of Parquet files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the Strings array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            The name of the Strings dataset to be written, defaults to strings_array
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', create a new Strings dataset within existing files.
        compressed : bool
            Defaults to False. When True, files will be written with Snappy compression
            and RLE bit packing.

        Returns
        -------
        String message indicating result of save operation

        Raises
        ------
        ValueError
            Raised if the lengths of columns and values differ, or the mode is
            neither 'truncate' nor 'append'
        TypeError
            Raised if prefix_path, dataset, or mode is not a str
        """
        from arkouda.io import mode_str_to_int

        return cast(
            str,
            generic_msg(
                "writeParquet",
                {
                    "values": self.entry,
                    "dset": dataset,
                    "mode": mode_str_to_int(mode),
                    "prefix": prefix_path,
                    "dtype": self.dtype,
                    "compressed": compressed,
                },
            ),
        )

    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "strings_array",
        mode: str = "truncate",
        save_offsets: bool = True,
        file_type: str = "distribute",
    ) -> str:
        """
        Save the Strings object to HDF5. The result is a collection of HDF5 files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the Strings array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            The name of the Strings dataset to be written, defaults to strings_array
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', create a new Strings dataset within existing files.
        save_offsets : bool
            Defaults to True which will instruct the server to save the offsets array to HDF5
            If False the offsets array will not be save and will be derived from the string values
            upon load/read.
        file_type: str ("single" | "distribute")
            Default: Distribute
            Distribute the dataset over a file per locale.
            Single file will save the dataset to one file

        Returns
        -------
        String message indicating result of save operation

        Raises
        ------
        ValueError
            Raised if the lengths of columns and values differ, or the mode is
            neither 'truncate' nor 'append'
        TypeError
            Raised if prefix_path, dataset, or mode is not a str
        """
        from arkouda.io import mode_str_to_int, file_type_to_int
        return cast(
            str,
            generic_msg(
                "tohdf",
                {
                    "values": self.entry,
                    "dset": dataset,
                    "write_mode": mode_str_to_int(mode),
                    "filename": prefix_path,
                    "dtype": self.dtype,
                    "save_offsets": save_offsets,
                    "objType": "strings",
                    "file_format": file_type_to_int(file_type),
                },
            ),
        )

    def save_parquet(
        self,
        prefix_path: str,
        dataset: str = "strings_array",
        mode: str = "truncate",
        compressed: bool = False,
    ) -> str:
        """
        DEPRECATED
        Save the Strings object to Parquet. The result is a collection of Parquet files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the Strings array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            The name of the Strings dataset to be written, defaults to strings_array
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', create a new Strings dataset within existing files.
        compressed : bool
            Defaults to False. When True, files will be written with Snappy compression
            and RLE bit packing.

        Returns
        -------
        String message indicating result of save operation

        Raises
        ------
        ValueError
            Raised if the lengths of columns and values differ, or the mode is
            neither 'truncate' nor 'append'
        TypeError
            Raised if prefix_path, dataset, or mode is not a str

        See Also
        --------
        strings.save()
        pdarray.save_parquet()
        """
        warn(
            "ak.pdarrayIO.save_parquet has been deprecated. Please use ak.Strings.to_parquet",
            DeprecationWarning,
        )
        return self.save(
            prefix_path=prefix_path,
            dataset=dataset,
            mode=mode,
            compressed=compressed,
            file_format="Parquet",
        )

    def save_hdf(
        self,
        prefix_path: str,
        dataset: str = "strings_array",
        mode: str = "truncate",
        save_offsets: bool = True,
        file_type: str = "",
    ) -> str:
        """
        DEPRECATED
        Save the Strings object to HDF5. The result is a collection of HDF5 files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the Strings array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            The name of the Strings dataset to be written, defaults to strings_array
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', create a new Strings dataset within existing files.
        save_offsets : bool
            Defaults to True which will instruct the server to save the offsets array to HDF5
            If False the offsets array will not be save and will be derived from the string values
            upon load/read.

        Returns
        -------
        String message indicating result of save operation

        Raises
        ------
        ValueError
            Raised if the lengths of columns and values differ, or the mode is
            neither 'truncate' nor 'append'
        TypeError
            Raised if prefix_path, dataset, or mode is not a str

        See Also
        --------
        strings.save()
        pdarray.save()
        """
        warn(
            "ak.Strings.save_hdf has been deprecated. Please use ak.Strings.to_hdf",
            DeprecationWarning,
        )
        return self.save(
            prefix_path=prefix_path,
            dataset=dataset,
            mode=mode,
            save_offsets=save_offsets,
            file_format="HDF5",
            file_type=file_type,
        )

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
        return np.bool_(self.entry.is_registered())

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
        return list(itertools.chain.from_iterable([self.entry._list_component_names()]))

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
        self.entry.pretty_print_info()

    @typechecked
    def register(self, user_defined_name: str) -> Strings:
        """
        Register this Strings object with a user defined name in the arkouda server
        so it can be attached to later using Strings.attach()
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

        See also
        --------
        attach, unregister

        Notes
        -----
        Registered names/Strings objects in the server are immune to deletion
        until they are unregistered.
        """
        self.entry.register(user_defined_name)
        self.name = user_defined_name
        return self

    def unregister(self) -> None:
        """
        Unregister a Strings object in the arkouda server which was previously
        registered using register() and/or attached to using attach()

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
        register, attach

        Notes
        -----
        Registered names/Strings objects in the server are immune to deletion until
        they are unregistered.
        """
        self.entry.unregister()
        self.name = None

    @staticmethod
    @typechecked
    def attach(user_defined_name: str) -> Strings:
        """
        class method to return a Strings object attached to the registered name in the arkouda
        server which was registered using register()

        Parameters
        ----------
        user_defined_name : str
            user defined name which the Strings object was registered under

        Returns
        -------
        Strings object
            the Strings object registered with user_defined_name in the arkouda server

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str

        See also
        --------
        register, unregister

        Notes
        -----
        Registered names/Strings objects in the server are immune to deletion
        until they are unregistered.
        """
        rep_msg: str = cast(str, generic_msg(cmd="attach", args={"name": user_defined_name}))
        s = Strings.from_return_msg(rep_msg)
        s.name = user_defined_name
        return s

    @staticmethod
    @typechecked
    def unregister_strings_by_name(user_defined_name: str) -> None:
        """
        Unregister a Strings object in the arkouda server previously registered via register()

        Parameters
        ----------
        user_defined_name : str
            The registered name of the Strings object

        See also
        --------
        register, unregister, attach, is_registered
        """
        unregister_pdarray_by_name(user_defined_name)
