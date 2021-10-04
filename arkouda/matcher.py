from typing import cast, Optional
from arkouda.client import generic_msg
from arkouda.pdarrayclass import create_pdarray
from arkouda.logger import getArkoudaLogger
from arkouda.dtypes import str_scalars
from arkouda.match import Match, MatchType
import json
import re
from arkouda.infoclass import list_symbol_table


class Matcher:
    LocationsInfo = frozenset(["num_matches", "starts", "lengths", "search_bool", "search_ind", "match_bool",
                               "match_ind", "full_match_bool", "full_match_ind"])

    def __init__(self, pattern: str_scalars, parent_bytes_name: str, parent_offsets_name: str) -> None:
        self.objtype = type(self).__name__
        try:
            self.pattern = pattern
            re.compile(self.pattern)
        except Exception as e:
            raise ValueError(e)
        self.parent_bytes_name = parent_bytes_name
        self.parent_offsets_name = parent_offsets_name
        self.num_matches = None
        self.starts = None
        self.lengths = None
        self.indices = None
        self.search_bool = None
        self.search_ind = None
        self.match_bool = None
        self.match_ind = None
        self.full_match_bool = None
        self.full_match_ind = None
        self.logger = getArkoudaLogger(name=__class__.__name__)  # type: ignore

    def find_locations(self) -> None:
        """
        Populates Matcher object by finding the positions of matches
        """
        sym_tab = list_symbol_table()
        if any([getattr(self, pda) is None or getattr(self, pda).name not in sym_tab for pda in self.LocationsInfo]):
            cmd = "segmentedFindLoc"
            args = "{} {} {} {} {}".format(self.objtype,
                                           self.parent_offsets_name,
                                           self.parent_bytes_name,
                                           0,  # groupNum is 0 for regular matches
                                           json.dumps([self.pattern]))
            repMsg = cast(str, generic_msg(cmd=cmd, args=args))
            arrays = repMsg.split('+', maxsplit=9)
            self.num_matches = create_pdarray(arrays[0])
            self.starts = create_pdarray(arrays[1])
            self.lengths = create_pdarray(arrays[2])
            self.indices = create_pdarray(arrays[3])
            self.search_bool = create_pdarray(arrays[4])
            self.search_ind = create_pdarray(arrays[5])
            self.match_bool = create_pdarray(arrays[6])
            self.match_ind = create_pdarray(arrays[7])
            self.full_match_bool = create_pdarray(arrays[8])
            self.full_match_ind = create_pdarray(arrays[9])

    def get_match(self, match_type: MatchType) -> Optional[Match]:
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

        if matched is not None and indices is not None:
            return Match(matched=matched,
                         starts=self.starts[self.indices[matched]],
                         lengths=self.lengths[self.indices[matched]],
                         indices=indices,
                         parent_bytes_name=self.parent_bytes_name,
                         parent_offsets_name=self.parent_offsets_name,
                         match_type=match_type,
                         pattern=self.pattern)
        else:
            return None

    def split(self, maxsplit: int = 0, return_segments: bool = False):
        """
        Split string by the occurrences of pattern. If maxsplit is nonzero, at most maxsplit splits occur
        """
        from arkouda.strings import Strings
        cmd = "segmentedSplit"
        args = "{} {} {} {} {} {}".format(self.objtype,
                                          self.parent_offsets_name,
                                          self.parent_bytes_name,
                                          maxsplit,
                                          return_segments,
                                          json.dumps([self.pattern]))
        repMsg = cast(str, generic_msg(cmd=cmd, args=args))
        if return_segments:
            arrays = repMsg.split('+', maxsplit=2)
            return Strings(arrays[0], arrays[1]), create_pdarray(arrays[2])
        else:
            arrays = repMsg.split('+', maxsplit=1)
            return Strings(arrays[0], arrays[1])

    def findall(self, return_match_origins: bool = False):
        """
        Return all non-overlapping matches of pattern in Strings as a new Strings object
        """
        from arkouda.strings import Strings
        self.find_locations()
        # These should always be set after `find_locations` but mypy is not convinced
        if self.num_matches is not None and self.starts is not None and self.lengths is not None and self.indices is not None:
            cmd = "segmentedFindAll"
            args = "{} {} {} {} {} {} {} {}".format(self.objtype,
                                                    self.parent_offsets_name,
                                                    self.parent_bytes_name,
                                                    self.num_matches.name,
                                                    self.starts.name,
                                                    self.lengths.name,
                                                    self.indices.name,
                                                    return_match_origins)
            repMsg = cast(str, generic_msg(cmd=cmd, args=args))
            if return_match_origins:
                arrays = repMsg.split('+', maxsplit=2)
                return Strings(arrays[0], arrays[1]), create_pdarray(arrays[2])
            else:
                arrays = repMsg.split('+', maxsplit=1)
                return Strings(arrays[0], arrays[1])
        return None

    def sub(self, repl: str, count: int = 0, return_num_subs: bool = False):
        """
        Return the Strings obtained by replacing non-overlapping occurrences of pattern with the replacement repl.
        If count is nonzero, at most count substitutions occur
        If return_num_subs is True, return the number of substitutions that occurred
        """
        from arkouda.strings import Strings
        cmd = "segmentedSub"
        args = "{} {} {} {} {} {} {}".format(self.objtype,
                                             self.parent_offsets_name,
                                             self.parent_bytes_name,
                                             repl,
                                             count,
                                             return_num_subs,
                                             json.dumps([self.pattern]))
        repMsg = cast(str, generic_msg(cmd=cmd, args=args))
        if return_num_subs:
            arrays = repMsg.split('+', maxsplit=2)
            return Strings(arrays[0], arrays[1]), create_pdarray(arrays[2])
        else:
            arrays = repMsg.split('+', maxsplit=1)
            return Strings(arrays[0], arrays[1])
