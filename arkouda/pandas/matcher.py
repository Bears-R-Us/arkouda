"""
Matching utilities for Arkouda arrays.

The `arkouda.matcher` module provides functions for efficiently aligning,
matching, and comparing Arkouda arrays. These tools are used internally to
support operations such as joins, merges, and reindexing across arrays,
particularly when working with categorical or structured data.

Functions in this module generally operate on one or more `pdarray` or
`Categorical` inputs and return index mappings or boolean masks indicating
matches or relationships between datasets.

Features
--------
- Index resolution for join operations
- Efficient value-matching across large arrays
- Support for type-aware and multi-column matching
- Underpins DataFrame merge, Series alignment, and MultiIndex comparisons

Typical Use Cases
-----------------
- Determining where values from one array appear in another
- Generating permutation indices for aligning arrays
- Supporting categorical equivalence testing

Notes
-----
- Functions in this module are primarily intended for internal use in
  `arkouda.pandas`-style functionality.
- Most functions assume inputs are distributed `pdarray`, `Strings`, or
  `Categorical` types with compatible shapes and data types.

See Also
--------
- arkouda.join
- arkouda.groupbyclass
- arkouda.pandas.merge
- arkouda.categorical.Categorical

"""

import json
import re
from typing import TYPE_CHECKING, TypeVar, cast

from arkouda.infoclass import list_symbol_table
from arkouda.logger import getArkoudaLogger
from arkouda.numpy.dtypes import str_scalars
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.pandas.match import Match, MatchType


if TYPE_CHECKING:
    from arkouda.client import generic_msg
else:
    generic_msg = TypeVar("generic_msg")


class Matcher:
    """
    Utility class for storing and standardizing information about pattern matches.

    The `Matcher` class defines a standard set of location-related fields that can be
    used to represent the results of search or match operations, typically involving
    string or pattern matching over Arkouda arrays.

    Attributes
    ----------
    frozenset
        LocationsInfo : A set of standardized string keys that describe match-related metadata.
        These include:
            - 'num_matches': total number of matches found.
            - 'starts': start positions of matches.
            - 'lengths': lengths of matches.
            - 'search_bool': boolean array indicating matches in search space.
            - 'search_ind': indices of matches in search space.
            - 'match_bool': boolean array indicating actual matches.
            - 'match_ind': indices of actual matches.
            - 'full_match_bool': boolean array for full string matches.
            - 'full_match_ind': indices of full matches.

    """

    LocationsInfo = frozenset(
        [
            "num_matches",
            "starts",
            "lengths",
            "search_bool",
            "search_ind",
            "match_bool",
            "match_ind",
            "full_match_bool",
            "full_match_ind",
        ]
    )

    def __init__(self, pattern: str_scalars, parent_entry_name: str) -> None:
        self.objType = type(self).__name__
        try:
            self.pattern = pattern
            re.compile(self.pattern)
        except Exception as e:
            raise ValueError(e)
        self.parent_entry_name = parent_entry_name
        self.num_matches: pdarray
        self.starts: pdarray
        self.lengths: pdarray
        self.indices: pdarray
        self.search_bool: pdarray
        self.search_ind: pdarray
        self.match_bool: pdarray
        self.match_ind: pdarray
        self.full_match_bool: pdarray
        self.full_match_ind: pdarray
        self.populated = False
        self.logger = getArkoudaLogger(name=__class__.__name__)  # type:ignore

    def find_locations(self) -> None:
        """Populate Matcher object by finding the positions of matches."""
        from arkouda.client import generic_msg

        sym_tab = list_symbol_table()
        if not self.populated or any(
            [getattr(self, pda).name not in sym_tab for pda in self.LocationsInfo]
        ):
            repMsg = cast(
                str,
                generic_msg(
                    cmd="segmentedFindLoc",
                    args={
                        "objType": self.objType,
                        "parent_name": self.parent_entry_name,
                        "groupNum": 0,  # groupNum is 0 for regular matches
                        "pattern": self.pattern,
                    },
                ),
            )
            created_map = json.loads(repMsg)
            self.num_matches = create_pdarray(created_map["NumMatches"])
            self.starts = create_pdarray(created_map["Starts"])
            self.lengths = create_pdarray(created_map["Lens"])
            self.indices = create_pdarray(created_map["Indices"])
            self.search_bool = create_pdarray(created_map["SearchBool"])
            self.search_ind = create_pdarray(created_map["SearchInd"])
            self.match_bool = create_pdarray(created_map["MatchBool"])
            self.match_ind = create_pdarray(created_map["MatchInd"])
            self.full_match_bool = create_pdarray(created_map["FullMatchBool"])
            self.full_match_ind = create_pdarray(created_map["FullMatchInd"])
            self.populated = True

    def get_match(self, match_type: MatchType, parent: object = None) -> Match:
        """Create a Match object of type match_type."""
        self.find_locations()
        if match_type == MatchType.SEARCH:
            matched = self.search_bool
            indices = self.search_ind
        elif match_type == MatchType.MATCH:
            matched = self.match_bool
            indices = self.match_ind
        elif match_type == MatchType.FULLMATCH:
            matched = self.full_match_bool
            indices = self.full_match_ind
        else:
            raise ValueError(f"{match_type} is not a MatchType")

        match = Match(
            matched=matched,
            starts=self.starts[self.indices[matched]],
            lengths=self.lengths[self.indices[matched]],
            indices=indices,
            parent_entry_name=self.parent_entry_name,
            match_type=match_type,
            pattern=self.pattern,
        )
        match._parent_obj = parent
        return match

    def split(self, maxsplit: int = 0, return_segments: bool = False):
        """
        Split string by the occurrences of pattern.

        If maxsplit is nonzero, at most maxsplit splits occur.
        """
        from arkouda.client import generic_msg
        from arkouda.numpy.strings import Strings

        if re.search(self.pattern, ""):
            raise ValueError("Cannot split or flatten with a pattern that matches the empty string")
        cmd = "segmentedSplit"
        repMsg = cast(
            str,
            generic_msg(
                cmd=cmd,
                args={
                    "parent_name": self.parent_entry_name,
                    "objtype": self.objType,
                    "max": maxsplit,
                    "return_segs": return_segments,
                    "pattern": self.pattern,
                },
            ),
        )
        if return_segments:
            arrays = repMsg.split("+", maxsplit=2)
            return Strings.from_return_msg("+".join(arrays[0:2])), create_pdarray(arrays[2])
        else:
            return Strings.from_return_msg(repMsg)

    def findall(self, return_match_origins: bool = False):
        """Return all non-overlapping matches of pattern in Strings as a new Strings object."""
        from arkouda.client import generic_msg
        from arkouda.numpy.strings import Strings

        self.find_locations()
        repMsg = cast(
            str,
            generic_msg(
                cmd="segmentedFindAll",
                args={
                    "objType": self.objType,
                    "parent_name": self.parent_entry_name,
                    "num_matches": self.num_matches,
                    "starts": self.starts,
                    "lengths": self.lengths,
                    "indices": self.indices,
                    "rtn_origins": return_match_origins,
                },
            ),
        )
        if return_match_origins:
            arrays = repMsg.split("+", maxsplit=2)
            return Strings.from_return_msg("+".join(arrays[0:2])), create_pdarray(arrays[2])
        else:
            return Strings.from_return_msg(repMsg)

    def sub(self, repl: str, count: int = 0, return_num_subs: bool = False):
        """
        Return the Strings obtained by replacing non-overlapping occurrences of pattern
        with the replacement repl.

        If count is nonzero, at most count substitutions occur
        If return_num_subs is True, return the number of substitutions that occurred.
        """
        from arkouda.client import generic_msg
        from arkouda.numpy.strings import Strings

        repMsg = cast(
            str,
            generic_msg(
                cmd="segmentedSub",
                args={
                    "objType": self.objType,
                    "obj": self.parent_entry_name,
                    "repl": repl,
                    "count": count,
                    "rtn_num_subs": return_num_subs,
                    "pattern": self.pattern,
                },
            ),
        )
        if return_num_subs:
            arrays = repMsg.split("+", maxsplit=2)
            return Strings.from_return_msg("+".join(arrays[0:2])), create_pdarray(arrays[2])
        else:
            return Strings.from_return_msg(repMsg)
