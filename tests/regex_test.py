from itertools import chain
import re

import numpy as np
import pytest

import arkouda as ak
from arkouda.pandas import match, matcher


class TestRegex:
    def test_match_docstrings(self):
        import doctest

        result = doctest.testmod(match, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_matcher_docstrings(self):
        import doctest

        result = doctest.testmod(matcher, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @classmethod
    def match_objects_helper(
        cls,
        pattern="_+",
        s=[
            "",
            "____",
            "_1_2____",
            "3___4___",
            "5",
            "__6__",
            "___7",
            "__8___9____10____11",
        ],
    ):
        strings = ak.array(s)

        for match_type in "search", "match", "fullmatch":
            ak_match_obj = getattr(strings, match_type)(pattern)
            re_match_obj = [getattr(re, match_type)(pattern, strings[i]) for i in range(strings.size)]
            re_non_none = [m for m in re_match_obj if m is not None]
            is_non_none = [m is not None for m in re_match_obj]

            assert ak_match_obj.matched().tolist() == is_non_none
            assert ak_match_obj.start().tolist() == [m.start() for m in re_non_none]
            assert ak_match_obj.end().tolist() == [m.end() for m in re_non_none]
            assert ak_match_obj.match_type() == match_type.upper()

            matches, origins = ak_match_obj.find_matches(return_match_origins=True)

            assert matches.tolist() == [m.string[m.start() : m.end()] for m in re_non_none]
            assert all(origins.to_ndarray() == np.arange(len(re_match_obj))[is_non_none])

    @classmethod
    def sub_helper(
        cls,
        pattern="_+",
        s=[
            "",
            "____",
            "_1_2____",
            "3___4___",
            "5",
            "__6__",
            "___7",
            "__8___9____10____11",
        ],
        repl="-",
        count=3,
    ):
        strings = ak.array(s)
        ak_sub, ak_sub_counts = strings.subn(pattern, repl, count)
        re_sub, re_sub_counts = zip(
            *(re.subn(pattern, repl, strings[i], count=count) for i in range(strings.size))
        )

        assert ak_sub.tolist() == list(re_sub)
        assert ak_sub_counts.tolist() == list(re_sub_counts)

    def test_empty_string_patterns(self):
        lit_str = ["0 String 0", "^", " "]
        ak_str = ak.array(lit_str)
        has_regex_arg = ["contains", "startswith", "endswith", "peel", "split"]

        for pattern in "", "|", "^":
            TestRegex.match_objects_helper(pattern, lit_str)
            TestRegex.sub_helper(pattern, lit_str, " +||+ ", 100)

            assert ak_str.contains(pattern, regex=True).all()
            assert ak_str.startswith(pattern, regex=True).tolist() == [
                re.search("^" + pattern, si) is not None for si in lit_str
            ]
            if pattern != "":
                assert ak_str.endswith(pattern, regex=True).tolist() == [
                    re.search(pattern + "$", si) is not None for si in lit_str
                ]

            assert ak_str.findall(pattern).tolist() == list(
                chain(*(re.findall(pattern, si) for si in lit_str))
            )

            # peel is broken on one char strings with patterns that match empty string
            # str split and non-regex flatten don't work with empty separator, so
            # it makes sense for the regex versions to return a value error
            for fn in "peel", "regex_split", "split":
                func = getattr(ak_str, fn)
                with pytest.raises(ValueError):
                    func(pattern, regex=True) if fn in has_regex_arg else func(pattern)

        # verify we value error with both
        # empty string matching patterns and empty string; See Chapel issue #20441
        # when pattern='$'; see Chapel issue #20431
        for fn in (
            "search",
            "match",
            "fullmatch",
            "sub",
            "contains",
            "startswith",
            "endswith",
            "findall",
            "peel",
            "regex_split",
            "split",
        ):
            for s, pat in zip([ak.array([""]), ak.array(["0 String 0"])], ["", "$"]):
                func = getattr(s, fn)
                with pytest.raises(ValueError):
                    if fn == "sub":
                        func(pat, " +||+ ")
                    else:
                        func(pat, regex=True) if fn in has_regex_arg else func(pat)

    def test_match_objects(self):
        TestRegex.match_objects_helper()

    def test_sub(self):
        # test with shorter and longer repl
        TestRegex.sub_helper(repl="-")
        TestRegex.sub_helper(repl="---------")

    def test_caputure_groups(self):
        tug_of_war = ak.array(
            [
                "Isaac Newton, physicist",
                "<--calculus-->",
                "Gottfried Leibniz, mathematician",
            ]
        )
        pattern = "(\\w+) (\\w+)"
        ak_captures = tug_of_war.search(pattern)
        re_captures = [re.search(pattern, tug_of_war[i]) for i in range(tug_of_war.size)]
        for i in range(3):
            assert ak_captures.group(i).tolist() == [m.group(i) for m in re_captures if m is not None]

        group, group_origins = ak_captures.group(1, return_group_origins=True)
        assert group_origins.tolist() == [
            i for i in range(len(re_captures)) if re_captures[i] is not None
        ]

        for i in -1, 40:
            with pytest.raises(ValueError):
                ak_captures.group(i)

        # verify fluid programming with Match object doesn't raise a RuntimeError
        ak.array(["1_2___", "____", "3", "__4___5____6___7", ""]).search("_+").find_matches()

    def test_regex_split(self):
        strings = ak.array(
            [
                "",
                "____",
                "_1_2____",
                "3___4___",
                "5",
                "__6__",
                "___7",
                "__8___9____10____11",
            ]
        )
        pattern = "_+"
        maxsplit = 3
        split, split_map = strings.regex_split(pattern, maxsplit, return_segments=True)
        for i in range(strings.size):
            re_split = re.split(pattern, strings[i], maxsplit=maxsplit)
            ak_split = (
                split[split_map[i] :]
                if i == strings.size - 1
                else split[split_map[i] : split_map[i + 1]]
            )
            assert re_split == ak_split.tolist()

    def test_regex_substr_search(self):
        digit_strings = ak.array([f"{i} string {i}" for i in range(6)])
        aaa_strings = ak.array([f"{'a' * i} string {'a' * i}" for i in range(1, 6)])

        assert ak.all(digit_strings.contains("\\d str", regex=True))
        assert ak.all(digit_strings.contains("ing \\d", regex=True))
        assert ak.all(aaa_strings.contains("a+ str", regex=True))
        assert ak.all(aaa_strings.contains("ing a+", regex=True))

        assert ak.all(digit_strings.startswith("\\d str", regex=True))
        assert not ak.any(digit_strings.startswith("ing \\d", regex=True))
        assert ak.all(aaa_strings.startswith("a+ str", regex=True))
        assert not ak.any(aaa_strings.startswith("ing a+", regex=True))

        assert not ak.any(digit_strings.endswith("\\d str", regex=True))
        assert ak.all(digit_strings.endswith("ing \\d", regex=True))
        assert not ak.any(aaa_strings.endswith("a+ str", regex=True))
        assert ak.all(aaa_strings.endswith("ing a+", regex=True))

    def test_regex_find_locations(self):
        strings = ak.array([f"{i} string {i}" for i in range(5)])

        actual_num_matches, actual_starts, actual_lens = strings.find_locations("\\d")
        assert [2, 2, 2, 2, 2] == actual_num_matches.tolist()
        assert [0, 9, 0, 9, 0, 9, 0, 9, 0, 9] == actual_starts.tolist()
        assert [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] == actual_lens.tolist()

        actual_num_matches, actual_starts, actual_lens = strings.find_locations("string \\d")
        assert [1, 1, 1, 1, 1] == actual_num_matches.tolist()
        assert [2, 2, 2, 2, 2] == actual_starts.tolist()
        assert [8, 8, 8, 8, 8] == actual_lens.tolist()

    def test_regex_findall(self):
        strings = ak.array([f"{i} string {i}" for i in range(1, 6)])
        expected_matches = ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"]
        expected_match_origins = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        actual_matches, actual_match_origins = strings.findall("\\d", return_match_origins=True)
        assert expected_matches == actual_matches.tolist()
        assert expected_match_origins == actual_match_origins.tolist()

        actual_matches = strings.findall("\\d")
        assert expected_matches == actual_matches.tolist()

        expected_matches = ["string 1", "string 2", "string 3", "string 4", "string 5"]
        expected_match_origins = [0, 1, 2, 3, 4]
        actual_matches, actual_match_origins = strings.findall("string \\d", return_match_origins=True)
        assert expected_matches == actual_matches.tolist()
        assert expected_match_origins == actual_match_origins.tolist()

        under = ak.array(["", "____", "_1_2", "3___4___", "5"])
        expected_matches = ["____", "_", "_", "___", "___"]
        expected_match_origins = [1, 2, 2, 3, 3]
        actual_matches, actual_match_origins = under.findall("_+", return_match_origins=True)
        assert expected_matches == actual_matches.tolist()
        assert expected_match_origins == actual_match_origins.tolist()

    def test_regex_peel(self):
        orig = ak.array(["a.b", "c.d", "e.f.g"])
        digit = ak.array(["a1b", "c1d", "e1f2g"])
        under = ak.array(["a_b", "c___d", "e__f____g"])

        o_left, o_right = orig.peel(".")
        d_left, d_right = digit.peel("\\d", regex=True)
        u_left, u_right = under.peel("_+", regex=True)
        assert ["a", "c", "e"] == o_left.tolist()
        assert ["a", "c", "e"] == d_left.tolist()
        assert ["a", "c", "e"] == u_left.tolist()
        assert ["b", "d", "f.g"] == o_right.tolist()
        assert ["b", "d", "f2g"] == d_right.tolist()
        assert ["b", "d", "f____g"] == u_right.tolist()

        o_left, o_right = orig.peel(".", includeDelimiter=True)
        d_left, d_right = digit.peel("\\d", includeDelimiter=True, regex=True)
        u_left, u_right = under.peel("_+", includeDelimiter=True, regex=True)
        assert ["a.", "c.", "e."] == o_left.tolist()
        assert ["a1", "c1", "e1"] == d_left.tolist()
        assert ["a_", "c___", "e__"] == u_left.tolist()
        assert ["b", "d", "f.g"] == o_right.tolist()
        assert ["b", "d", "f2g"] == d_right.tolist()
        assert ["b", "d", "f____g"] == u_right.tolist()

        o_left, o_right = orig.peel(".", times=2, keepPartial=True)
        d_left, d_right = digit.peel("\\d", times=2, keepPartial=True, regex=True)
        u_left, u_right = under.peel("_+", times=2, keepPartial=True, regex=True)
        assert ["a.b", "c.d", "e.f"] == o_left.tolist()
        assert ["a1b", "c1d", "e1f"] == d_left.tolist()
        assert ["a_b", "c___d", "e__f"] == u_left.tolist()
        assert ["", "", "g"] == o_right.tolist()
        assert ["", "", "g"] == d_right.tolist()
        assert ["", "", "g"] == u_right.tolist()

        # rpeel / fromRight: digit is testing fromRight and under is testing rpeel
        o_left, o_right = orig.peel(".", times=2, includeDelimiter=True, fromRight=True)
        d_left, d_right = digit.peel("\\d", times=2, includeDelimiter=True, fromRight=True, regex=True)
        u_left, u_right = under.rpeel("_+", times=2, includeDelimiter=True, regex=True)
        assert ["a.b", "c.d", "e"] == o_left.tolist()
        assert ["a1b", "c1d", "e"] == d_left.tolist()
        assert ["a_b", "c___d", "e"] == u_left.tolist()
        assert ["", "", ".f.g"] == o_right.tolist()
        assert ["", "", "1f2g"] == d_right.tolist()
        assert ["", "", "__f____g"] == u_right.tolist()

    def test_regex_on_split(self):
        orig = ak.array(["one|two", "three|four|five", "six", "seven|eight|nine|ten|", "eleven"])
        digit = ak.array(["one1two", "three2four3five", "six", "seven4eight5nine6ten7", "eleven"])
        under = ak.array(
            [
                "one_two",
                "three_four__five",
                "six",
                "seven_____eight__nine____ten_",
                "eleven",
            ]
        )

        answer_flat = [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "",
            "eleven",
        ]
        answer_map = [0, 2, 5, 6, 11]
        for pattern, strings in zip(["|", "\\d", "_+"], [orig, digit, under]):
            ak_flat, ak_map = strings.split(pattern, return_segments=True, regex=pattern != "|")
            assert answer_flat == ak_flat.tolist()
            assert answer_map == ak_map.tolist()

        # empty string, start with delim, end with delim, and only delim cases
        orig = ak.array(["", "|", "|1|2", "3|4|", "5"])
        regex = ak.array(["", "____", "_1_2", "3___4___", "5"])

        answer_flat = ["", "", "", "", "1", "2", "3", "4", "", "5"]
        answer_map = [0, 1, 3, 6, 9]

        orig_flat, orig_map = orig.split("|", return_segments=True)
        regex_flat, regex_map = regex.split("_+", return_segments=True, regex=True)

        assert answer_flat == orig_flat.tolist()
        assert answer_flat == regex_flat.tolist()
        assert answer_map == orig_map.tolist()
        assert answer_map == regex_map.tolist()
