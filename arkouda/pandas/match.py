r"""
Regular expression match results for segmented string arrays in Arkouda.

The `arkouda.match` module defines the `Match` class, which encapsulates results from
regex-based operations such as `search`, `match`, and `fullmatch` on Arkouda `Strings` arrays.

This class provides methods to retrieve:
- Match booleans (`matched`)
- Start and end positions of matches
- Capture groups
- Matched substrings
- Origin indices of matched elements (optional)

These operations enable powerful pattern recognition and substring extraction
on large-scale segmented string arrays, implemented efficiently in the Arkouda server.

Examples
--------
>>> import arkouda as ak
>>> strings = ak.array(["Isaac Newton", "Ada Lovelace", ""])
>>> m = strings.search("(\\w+) (\\w+)")
>>> m.matched()
array([True True False])
>>> m.group(1)
array(['Isaac', 'Ada'])
>>> m.group(2, return_group_origins=True)
(array(['Newton', 'Lovelace']), array([0 1]))

Notes
-----
- `group(0)` returns the full match by default.
- If `regexMaxCaptures` is exceeded, the server must be recompiled with a higher limit.
- `Match` objects are typically obtained via `Strings.search()`, /
`Strings.match()`, or `Strings.fullmatch()`.

See Also
--------
arkouda.strings.Strings
arkouda.client.regexMaxCaptures

"""

from enum import Enum
import json
from typing import TYPE_CHECKING, TypeVar, cast

from arkouda.numpy.pdarrayclass import create_pdarray, pdarray


if TYPE_CHECKING:
    from arkouda.client import generic_msg
else:
    generic_msg = TypeVar("generic_msg")

__all__ = ["Match"]

MatchType = Enum("MatchType", ["SEARCH", "MATCH", "FULLMATCH"])


class Match:
    """
    Encapsulates regular expression match results on Arkouda segmented string arrays.

    Created by calling `search()`, `match()`, or `fullmatch()` on a `Strings` object. Provides access
    to match booleans, span information, capture groups, and origin indices of matches.

    Attributes
    ----------
    re : str
        Regex pattern used.

    Examples
    --------
    >>> import arkouda as ak
    >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
    >>> m = strings.search('_+')
    >>> m
    <ak.Match object: matched=True, span=(1, 2); matched=True, span=(0, 4);
    matched=False; matched=True, span=(0, 2); matched=False>
    >>> type(m)
    <class 'arkouda.pandas.match.Match'>
    >>> m.matched()
    array([True True False True False])
    >>> m.start()
    array([1 0 0])
    >>> m.end()
    array([2 4 2])
    >>> m.match_type()
    'SEARCH'
    >>> m.re
    '_+'
    >>> m[1]
    'matched=True, span=(0, 4)'

    """

    re: str

    def __init__(
        self,
        matched: pdarray,
        starts: pdarray,
        lengths: pdarray,
        indices: pdarray,
        parent_entry_name: str,
        match_type: MatchType,
        pattern: str,
    ):
        self._objtype = type(self).__name__
        self._parent_entry_name = parent_entry_name
        self._match_type = match_type
        self._matched = matched
        self._starts = starts
        self._lengths = lengths
        self._ends = starts + lengths
        self._indices = indices
        self._parent_obj: object = None
        self.re = pattern

    def __str__(self):
        """
        Return a string representation of the match object, previewing match status and spans.

        Returns
        -------
        str
            Human-readable summary of match objects and spans.

        """
        from arkouda.client import pdarrayIterThresh

        if self._matched.size <= pdarrayIterThresh:
            vals = [self.__getitem__(i) for i in range(self._matched.size)]
        else:
            vals = [self.__getitem__(i) for i in range(3)]
            vals.append("... ")
            vals.extend([self.__getitem__(i) for i in range(self._matched.size - 3, self._matched.size)])
        return f"<ak.{self._objtype} object: {'; '.join(vals)}>"

    def __getitem__(self, item):
        """
        Return a summary string for the match at a given index.

        Parameters
        ----------
        item : int
            Index of the match to describe.

        Returns
        -------
        str
            Description of whether the item matched, and its span if it did.

        """
        return (
            f"matched={self._matched[item]}, span=({self._starts[self._indices[item]]}"
            f", {self._ends[self._indices[item]]})"
            if self._matched[item]
            else f"matched={self._matched[item]}"
        )

    def __repr__(self):
        """
        Return the formal string representation of the Match object.

        Returns
        -------
        str
            Same as __str__ for now.

        """
        return self.__str__()

    def matched(self) -> pdarray:
        """
        Return a boolean array indiciating whether each element matched.

        Returns
        -------
        pdarray
            True for elements that match, False otherwise

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.search('_+').matched()
        array([True True False True False])

        """
        return self._matched

    def start(self) -> pdarray:
        """
        Return the starts of matches.

        Returns
        -------
        pdarray
            The start positions of matches

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.search('_+').start()
        array([1 0 0])

        """
        return self._starts

    def end(self) -> pdarray:
        """
        Return the ends of matches.

        Returns
        -------
        pdarray
            The end positions of matches

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.search('_+').end()
        array([2 4 2])

        """
        return self._ends

    def match_type(self) -> str:
        """
        Return the type of the Match object.

        Returns
        -------
        str
            MatchType of the Match object

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.search('_+').match_type()
        'SEARCH'

        """
        return self._match_type.name

    def find_matches(self, return_match_origins: bool = False):
        """
        Return all matches as a new Strings object.

        Parameters
        ----------
        return_match_origins: bool
            If True, return a pdarray containing the index of the original string each pattern
            match is from

        Returns
        -------
        Strings
            Strings object containing only matches
        pdarray, int64 (optional)
            The index of the original string each pattern match is from

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(['1_2___', '____', '3', '__4___5____6___7', ''])
        >>> strings.search('_+').find_matches(return_match_origins=True)
        (array(['_', '____', '__']), array([0 1 3]))

        """
        from arkouda.client import generic_msg
        from arkouda.numpy.strings import Strings

        repMsg = cast(
            str,
            generic_msg(
                cmd="segmentedFindAll",
                args={
                    "objType": self._objtype,
                    "parent_name": self._parent_entry_name,
                    "num_matches": self._matched,
                    "starts": self._starts,
                    "lengths": self._lengths,
                    "indices": self._indices,
                    "rtn_origins": return_match_origins,
                },
            ),
        )
        if return_match_origins:
            arrays = repMsg.split("+", maxsplit=2)
            return Strings.from_return_msg("+".join(arrays[0:2])), create_pdarray(arrays[2])
        else:
            return Strings.from_return_msg(repMsg)

    def group(self, group_num: int = 0, return_group_origins: bool = False):
        r"""
        Return a new Strings containing the capture group corresponding to group_num.

        For the default, group_num=0, return the full match.

        Parameters
        ----------
        group_num: int
            The index of the capture group to be returned
        return_group_origins: bool
            If True, return a pdarray containing the index of the original string each
            capture group is from

        Returns
        -------
        Strings
            Strings object containing only the capture groups corresponding to group_num
        pdarray, int64 (optional)
            The index of the original string each group is from

        Examples
        --------
        >>> import arkouda as ak
        >>> strings = ak.array(["Isaac Newton, physics", '<-calculus->', 'Gottfried Leibniz, math'])
        >>> m = strings.search("(\\w+) (\\w+)")
        >>> m.group()
        array(['Isaac Newton', 'Gottfried Leibniz'])
        >>> m.group(1)
        array(['Isaac', 'Gottfried'])
        >>> m.group(2, return_group_origins=True)
        (array(['Newton', 'Leibniz']), array([0 2]))

        """
        from arkouda.client import generic_msg, regexMaxCaptures
        from arkouda.numpy.strings import Strings

        if group_num < 0:
            raise ValueError("group_num cannot be negative")
        if group_num > regexMaxCaptures:
            max_capture_flag = f"-e REGEX_MAX_CAPTURES={group_num}"
            e = (
                f"group_num={group_num} > regexMaxCaptures={regexMaxCaptures}."
                f" To run group({group_num}), recompile the server with flag '{max_capture_flag}'"
            )
            raise ValueError(e)

        # We don't cache the locations of groups, find the location info and call findAll
        repMsg = cast(
            str,
            generic_msg(
                cmd="segmentedFindLoc",
                args={
                    "objType": self._objtype,
                    "parent_name": self._parent_entry_name,
                    "groupNum": group_num,
                    "pattern": self.re,
                },
            ),
        )
        created_map = json.loads(repMsg)
        global_starts = create_pdarray(created_map["Starts"])
        global_lengths = create_pdarray(created_map["Lens"])
        global_indices = create_pdarray(created_map["Indices"])
        if self._match_type == MatchType.SEARCH:
            matched = create_pdarray(created_map["SearchBool"])
            indices = create_pdarray(created_map["SearchInd"])
        elif self._match_type == MatchType.MATCH:
            matched = create_pdarray(created_map["MatchBool"])
            indices = create_pdarray(created_map["MatchInd"])
        elif self._match_type == MatchType.FULLMATCH:
            matched = create_pdarray(created_map["FullMatchBool"])
            indices = create_pdarray(created_map["FullMatchInd"])
        else:
            raise ValueError(f"{self._match_type} is not a MatchType")
        starts = global_starts[global_indices[matched]]
        lengths = global_lengths[global_indices[matched]]
        repMsg = cast(
            str,
            generic_msg(
                cmd="segmentedFindAll",
                args={
                    "objType": self._objtype,
                    "parent_name": self._parent_entry_name,
                    "num_matches": matched,
                    "starts": starts,
                    "lengths": lengths,
                    "indices": indices,
                    "rtn_origins": return_group_origins,
                },
            ),
        )
        if return_group_origins:
            arrays = repMsg.split("+", maxsplit=2)
            return Strings.from_return_msg("+".join(arrays[0:2])), create_pdarray(arrays[2])
        else:
            return Strings.from_return_msg(repMsg)
