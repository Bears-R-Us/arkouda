import json
import re
from typing import cast

from arkouda.client import generic_msg
from arkouda.dtypes import str_scalars
from arkouda.infoclass import list_symbol_table
from arkouda.logger import getArkoudaLogger
from arkouda.match import Match, MatchType
from arkouda.pdarrayclass import create_pdarray, pdarray


class Matcher:
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
        self.objtype = type(self).__name__
        try:
            self.pattern = pattern
            re.compile(self.pattern)
        except Exception as e:
            raise ValueError(e)
        if re.search(self.pattern, ""):
            # TODO remove once changes from chapel issue #18639 are in arkouda
            raise ValueError(
                "regex operations with a pattern that matches the empty string are"
                "not currently supported"
            )
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
        self.logger = getArkoudaLogger(name=__class__.__name__)  # type: ignore

    def find_locations(self) -> None:
        """
        Populates Matcher object by finding the positions of matches
        """
        sym_tab = list_symbol_table()
        if not self.populated or any(
            [getattr(self, pda).name not in sym_tab for pda in self.LocationsInfo]
        ):
            cmd = "segmentedFindLoc"
            args = "{} {} {} {}".format(
                self.objtype,
                self.parent_entry_name,
                0,  # groupNum is 0 for regular matches
                json.dumps([self.pattern]),
            )
            repMsg = cast(str, generic_msg(cmd=cmd, args=args))
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
        """
        Create a Match object of type match_type
        """
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
        Split string by the occurrences of pattern. If maxsplit is nonzero, at most maxsplit splits occur
        """
        from arkouda.strings import Strings

        cmd = "segmentedSplit"
        args = "{} {} {} {} {}".format(
            self.objtype, self.parent_entry_name, maxsplit, return_segments, json.dumps([self.pattern])
        )
        repMsg = cast(str, generic_msg(cmd=cmd, args=args))
        if return_segments:
            arrays = repMsg.split("+", maxsplit=2)
            return Strings.from_return_msg("+".join(arrays[0:2])), create_pdarray(arrays[2])
        else:
            return Strings.from_return_msg(repMsg)

    def findall(self, return_match_origins: bool = False):
        """
        Return all non-overlapping matches of pattern in Strings as a new Strings object
        """
        from arkouda.strings import Strings

        self.find_locations()
        cmd = "segmentedFindAll"
        args = "{} {} {} {} {} {} {}".format(
            self.objtype,
            self.parent_entry_name,
            self.num_matches.name,
            self.starts.name,
            self.lengths.name,
            self.indices.name,
            return_match_origins,
        )
        repMsg = cast(str, generic_msg(cmd=cmd, args=args))
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
        If return_num_subs is True, return the number of substitutions that occurred
        """
        from arkouda.strings import Strings

        cmd = "segmentedSub"
        args = "{} {} {} {} {} {}".format(
            self.objtype,
            self.parent_entry_name,
            repl,
            count,
            return_num_subs,
            json.dumps([self.pattern]),
        )
        repMsg = cast(str, generic_msg(cmd=cmd, args=args))
        if return_num_subs:
            arrays = repMsg.split("+", maxsplit=2)
            return Strings.from_return_msg("+".join(arrays[0:2])), create_pdarray(arrays[2])
        else:
            return Strings.from_return_msg(repMsg)
