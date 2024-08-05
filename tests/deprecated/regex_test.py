import re

import numpy as np

from base_test import ArkoudaTest
from context import arkouda as ak


class RegexTest(ArkoudaTest):
    def test_empty_string_patterns(self):
        # verify methods which use Regex.matches work with pattern matches the empty string
        s = ["0 String 0", "^", " "]
        s2 = ak.array(s)
        for pattern in ["", "|", "^"]:
            self.test_match_objects(pattern, s)
            self.sub_help(pattern, s, "repl", 100)
            self.assertTrue(s2.contains(pattern, regex=True).all())
            self.assertListEqual(
                s2.startswith(pattern, regex=True).to_list(),
                [re.search("^" + pattern, si) is not None for si in s],
            )
            if pattern != "":
                self.assertListEqual(
                    s2.endswith(pattern, regex=True).to_list(),
                    [re.search(pattern + "$", si) is not None for si in s],
                )
            self.assertListEqual(
                s2.findall(pattern).to_list(),
                np.concatenate([re.findall(pattern, si) for si in s]).tolist(),
            )
            # peel is broken on one char strings with patterns that match empty string
            # str split and non-regex flatten don't work with empty separator, so
            # it makes sense for the regex versions to return a value error
            with self.assertRaises(ValueError):
                s2.peel(pattern, regex=True)
            with self.assertRaises(ValueError):
                s2.split(pattern)
            with self.assertRaises(ValueError):
                s2.flatten(pattern, regex=True)

        # verify we value error with both
        # empty string matching patterns and empty string; See Chapel issue #20441
        # when pattern='$'; see Chapel issue #20431
        for s, p in zip([ak.array([""]), ak.array(["0 String 0"])], ["", "$"]):
            with self.assertRaises(ValueError):
                s.search(p)
            with self.assertRaises(ValueError):
                s.match(p)
            with self.assertRaises(ValueError):
                s.fullmatch(p)
            with self.assertRaises(ValueError):
                s.sub(p, "repl")
            with self.assertRaises(ValueError):
                s.contains(p, regex=True)
            with self.assertRaises(ValueError):
                s.startswith(p, regex=True)
            with self.assertRaises(ValueError):
                s.endswith(p, regex=True)
            with self.assertRaises(ValueError):
                s.findall(p)
            with self.assertRaises(ValueError):
                s.peel(p, regex=True)
            with self.assertRaises(ValueError):
                s.split(p)
            with self.assertRaises(ValueError):
                s.flatten(p, regex=True)

    def test_match_objects(
        self,
        pattern="_+",
        s=["", "____", "_1_2____", "3___4___", "5", "__6__", "___7", "__8___9____10____11"],
    ):
        strings = ak.array(s)
        ak_search = strings.search(pattern)
        ak_match = strings.match(pattern)
        ak_fullmatch = strings.fullmatch(pattern)
        re_search = [re.search(pattern, strings[i]) for i in range(strings.size)]
        re_match = [re.match(pattern, strings[i]) for i in range(strings.size)]
        re_fullmatch = [re.fullmatch(pattern, strings[i]) for i in range(strings.size)]

        self.assertListEqual(ak_search.matched().to_list(), [m is not None for m in re_search])
        self.assertListEqual(ak_match.matched().to_list(), [m is not None for m in re_match])
        self.assertListEqual(ak_fullmatch.matched().to_list(), [m is not None for m in re_fullmatch])

        self.assertListEqual(
            ak_search.start().to_list(), [m.start() for m in re_search if m is not None]
        )
        self.assertListEqual(ak_match.start().to_list(), [m.start() for m in re_match if m is not None])
        self.assertListEqual(
            ak_fullmatch.start().to_list(),
            [m.start() for m in re_fullmatch if m is not None],
        )

        self.assertListEqual(ak_search.end().to_list(), [m.end() for m in re_search if m is not None])
        self.assertListEqual(ak_match.end().to_list(), [m.end() for m in re_match if m is not None])
        self.assertListEqual(
            ak_fullmatch.end().to_list(), [m.end() for m in re_fullmatch if m is not None]
        )

        self.assertEqual(ak_search.match_type(), "SEARCH")
        self.assertEqual(ak_match.match_type(), "MATCH")
        self.assertEqual(ak_fullmatch.match_type(), "FULLMATCH")

        search_matches, search_origins = ak_search.find_matches(return_match_origins=True)
        match_matches, match_origins = ak_match.find_matches(return_match_origins=True)
        fullmatch_matches, fullmatch_origins = ak_fullmatch.find_matches(return_match_origins=True)
        self.assertListEqual(
            search_matches.to_list(),
            [m.string[m.start() : m.end()] for m in re_search if m is not None],
        )
        self.assertListEqual(
            search_origins.to_list(),
            [i for i in range(len(re_search)) if re_search[i] is not None],
        )
        self.assertListEqual(
            match_matches.to_list(),
            [m.string[m.start() : m.end()] for m in re_match if m is not None],
        )
        self.assertListEqual(
            match_origins.to_list(),
            [i for i in range(len(re_match)) if re_match[i] is not None],
        )
        self.assertListEqual(
            fullmatch_matches.to_list(),
            [m.string[m.start() : m.end()] for m in re_fullmatch if m is not None],
        )
        self.assertListEqual(
            fullmatch_origins.to_list(),
            [i for i in range(len(re_fullmatch)) if re_fullmatch[i] is not None],
        )

    def test_caputure_groups(self):
        tug_of_war = ak.array(
            ["Isaac Newton, physicist", "<--calculus-->", "Gottfried Leibniz, mathematician"]
        )
        ak_captures = tug_of_war.search("(\\w+) (\\w+)")
        re_captures = [re.search("(\\w+) (\\w+)", tug_of_war[i]) for i in range(tug_of_war.size)]
        self.assertListEqual(
            ak_captures.group().to_list(), [m.group() for m in re_captures if m is not None]
        )
        self.assertListEqual(
            ak_captures.group(1).to_list(),
            [m.group(1) for m in re_captures if m is not None],
        )
        self.assertListEqual(
            ak_captures.group(2).to_list(),
            [m.group(2) for m in re_captures if m is not None],
        )

        group, group_origins = ak_captures.group(1, return_group_origins=True)
        self.assertListEqual(
            group_origins.to_list(),
            [i for i in range(len(re_captures)) if re_captures[i] is not None],
        )

        with self.assertRaises(ValueError):
            ak_captures.group(-1)
        with self.assertRaises(ValueError):
            ak_captures.group(40)

        # verify fluid programming with Match object doesn't raise a RuntimeError
        ak.array(["1_2___", "____", "3", "__4___5____6___7", ""]).search("_+").find_matches()

    def test_sub(self):
        # test short repl
        self.sub_help(repl="-")
        # test long repl
        self.sub_help(repl="---------")

    def sub_help(
        self,
        pattern="_+",
        s=["", "____", "_1_2____", "3___4___", "5", "__6__", "___7", "__8___9____10____11"],
        repl="-",
        count=3,
    ):
        strings = ak.array(s)
        re_sub = [re.sub(pattern, repl, strings[i], count) for i in range(strings.size)]
        re_sub_counts = [re.subn(pattern, repl, strings[i], count)[1] for i in range(strings.size)]
        ak_sub, ak_sub_counts = strings.subn(pattern, repl, count)
        self.assertListEqual(re_sub, ak_sub.to_list())
        self.assertListEqual(re_sub_counts, ak_sub_counts.to_list())

    def test_split(self):
        strings = ak.array(
            ["", "____", "_1_2____", "3___4___", "5", "__6__", "___7", "__8___9____10____11"]
        )
        pattern = "_+"
        maxsplit = 3
        split, split_map = strings.split(pattern, maxsplit, return_segments=True)
        for i in range(strings.size):
            re_split = re.split(pattern, strings[i], maxsplit)
            ak_split = (
                split[split_map[i] :]
                if i == strings.size - 1
                else split[split_map[i] : split_map[i + 1]]
            )
            self.assertListEqual(re_split, ak_split.to_list())

    def test_regex_contains(self):
        digit_strings = ak.array(["{} string {}".format(i, i) for i in range(1, 6)])
        self.assertTrue(digit_strings.contains("\\d str", regex=True).all())
        self.assertTrue(digit_strings.contains("ing \\d", regex=True).all())

        aaa_strings = ak.array(["{} string {}".format("a" * i, "a" * i) for i in range(1, 6)])
        self.assertTrue(aaa_strings.contains("a+ str", regex=True).all())
        self.assertTrue(aaa_strings.contains("ing a+", regex=True).all())

    def test_regex_startswith(self):
        digit_strings = ak.array(["{} string {}".format(i, i) for i in range(1, 6)])
        self.assertTrue(digit_strings.startswith("\\d str", regex=True).all())
        self.assertFalse(digit_strings.startswith("ing \\d", regex=True).any())

        aaa_strings = ak.array(["{} string {}".format("a" * i, "a" * i) for i in range(1, 6)])
        self.assertTrue(aaa_strings.startswith("a+ str", regex=True).all())
        self.assertFalse(aaa_strings.startswith("ing a+", regex=True).any())

    def test_regex_endswith(self):
        digit_strings = ak.array(["{} string {}".format(i, i) for i in range(1, 6)])
        self.assertTrue(digit_strings.endswith("ing \\d", regex=True).all())
        self.assertFalse(digit_strings.endswith("\\d str", regex=True).any())

        aaa_strings = ak.array(["{} string {}".format("a" * i, "a" * i) for i in range(1, 6)])
        self.assertTrue(aaa_strings.endswith("ing a+", regex=True).all())
        self.assertFalse(aaa_strings.endswith("a+ str", regex=True).any())

    def test_regex_find_locations(self):
        strings = ak.array(["{} string {}".format(i, i) for i in range(1, 6)])

        actual_num_matches, actual_starts, actual_lens = strings.find_locations("\\d")
        self.assertListEqual([2, 2, 2, 2, 2], actual_num_matches.to_list())
        self.assertListEqual([0, 9, 0, 9, 0, 9, 0, 9, 0, 9], actual_starts.to_list())
        self.assertListEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], actual_lens.to_list())

        actual_num_matches, actual_starts, actual_lens = strings.find_locations("string \\d")
        self.assertListEqual([1, 1, 1, 1, 1], actual_num_matches.to_list())
        self.assertListEqual([2, 2, 2, 2, 2], actual_starts.to_list())
        self.assertListEqual([8, 8, 8, 8, 8], actual_lens.to_list())

    def test_regex_findall(self):
        strings = ak.array(["{} string {}".format(i, i) for i in range(1, 6)])
        expected_matches = ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"]
        expected_match_origins = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        actual_matches, actual_match_origins = strings.findall("\\d", return_match_origins=True)
        self.assertListEqual(expected_matches, actual_matches.to_list())
        self.assertListEqual(expected_match_origins, actual_match_origins.to_list())
        actual_matches = strings.findall("\\d")
        self.assertListEqual(expected_matches, actual_matches.to_list())

        expected_matches = ["string 1", "string 2", "string 3", "string 4", "string 5"]
        expected_match_origins = [0, 1, 2, 3, 4]
        actual_matches, actual_match_origins = strings.findall("string \\d", return_match_origins=True)
        self.assertListEqual(expected_matches, actual_matches.to_list())
        self.assertListEqual(expected_match_origins, actual_match_origins.to_list())

        under = ak.array(["", "____", "_1_2", "3___4___", "5"])
        expected_matches = ["____", "_", "_", "___", "___"]
        expected_match_origins = [1, 2, 2, 3, 3]
        actual_matches, actual_match_origins = under.findall("_+", return_match_origins=True)
        self.assertListEqual(expected_matches, actual_matches.to_list())
        self.assertListEqual(expected_match_origins, actual_match_origins.to_list())

    def test_regex_peel(self):
        orig = ak.array(["a.b", "c.d", "e.f.g"])
        digit = ak.array(["a1b", "c1d", "e1f2g"])
        under = ak.array(["a_b", "c___d", "e__f____g"])

        o_left, o_right = orig.peel(".")
        d_left, d_right = digit.peel("\\d", regex=True)
        u_left, u_right = under.peel("_+", regex=True)
        self.assertListEqual(["a", "c", "e"], o_left.to_list())
        self.assertListEqual(["a", "c", "e"], d_left.to_list())
        self.assertListEqual(["a", "c", "e"], u_left.to_list())
        self.assertListEqual(["b", "d", "f.g"], o_right.to_list())
        self.assertListEqual(["b", "d", "f2g"], d_right.to_list())
        self.assertListEqual(["b", "d", "f____g"], u_right.to_list())

        o_left, o_right = orig.peel(".", includeDelimiter=True)
        d_left, d_right = digit.peel("\\d", includeDelimiter=True, regex=True)
        u_left, u_right = under.peel("_+", includeDelimiter=True, regex=True)
        self.assertListEqual(["a.", "c.", "e."], o_left.to_list())
        self.assertListEqual(["a1", "c1", "e1"], d_left.to_list())
        self.assertListEqual(["a_", "c___", "e__"], u_left.to_list())
        self.assertListEqual(["b", "d", "f.g"], o_right.to_list())
        self.assertListEqual(["b", "d", "f2g"], d_right.to_list())
        self.assertListEqual(["b", "d", "f____g"], u_right.to_list())

        o_left, o_right = orig.peel(".", times=2, keepPartial=True)
        d_left, d_right = digit.peel("\\d", times=2, keepPartial=True, regex=True)
        u_left, u_right = under.peel("_+", times=2, keepPartial=True, regex=True)
        self.assertListEqual(["a.b", "c.d", "e.f"], o_left.to_list())
        self.assertListEqual(["a1b", "c1d", "e1f"], d_left.to_list())
        self.assertListEqual(["a_b", "c___d", "e__f"], u_left.to_list())
        self.assertListEqual(["", "", "g"], o_right.to_list())
        self.assertListEqual(["", "", "g"], d_right.to_list())
        self.assertListEqual(["", "", "g"], u_right.to_list())

        # rpeel / fromRight: digit is testing fromRight and under is testing rpeel
        o_left, o_right = orig.peel(".", times=2, includeDelimiter=True, fromRight=True)
        d_left, d_right = digit.peel("\\d", times=2, includeDelimiter=True, fromRight=True, regex=True)
        u_left, u_right = under.rpeel("_+", times=2, includeDelimiter=True, regex=True)
        self.assertListEqual(["a.b", "c.d", "e"], o_left.to_list())
        self.assertListEqual(["a1b", "c1d", "e"], d_left.to_list())
        self.assertListEqual(["a_b", "c___d", "e"], u_left.to_list())
        self.assertListEqual(["", "", ".f.g"], o_right.to_list())
        self.assertListEqual(["", "", "1f2g"], d_right.to_list())
        self.assertListEqual(["", "", "__f____g"], u_right.to_list())

    def test_regex_flatten(self):
        orig = ak.array(["one|two", "three|four|five", "six", "seven|eight|nine|ten|", "eleven"])
        digit = ak.array(["one1two", "three2four3five", "six", "seven4eight5nine6ten7", "eleven"])
        under = ak.array(
            ["one_two", "three_four__five", "six", "seven_____eight__nine____ten_", "eleven"]
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

        orig_flat, orig_map = orig.flatten("|", return_segments=True)
        digit_flat, digit_map = digit.flatten("\\d", return_segments=True, regex=True)
        under_flat, under_map = under.flatten("_+", return_segments=True, regex=True)

        self.assertListEqual(answer_flat, orig_flat.to_list())
        self.assertListEqual(answer_flat, digit_flat.to_list())
        self.assertListEqual(answer_flat, under_flat.to_list())

        self.assertListEqual(answer_map, orig_map.to_list())
        self.assertListEqual(answer_map, digit_map.to_list())
        self.assertListEqual(answer_map, under_map.to_list())

        # empty string, start with delim, end with delim, and only delim cases
        orig = ak.array(["", "|", "|1|2", "3|4|", "5"])
        regex = ak.array(["", "____", "_1_2", "3___4___", "5"])

        answer_flat = ["", "", "", "", "1", "2", "3", "4", "", "5"]
        answer_map = [0, 1, 3, 6, 9]

        orig_flat, orig_map = orig.flatten("|", return_segments=True)
        regex_flat, regex_map = regex.flatten("_+", return_segments=True, regex=True)

        self.assertListEqual(answer_flat, orig_flat.to_list())
        self.assertListEqual(answer_flat, regex_flat.to_list())

        self.assertListEqual(answer_map, orig_map.to_list())
        self.assertListEqual(answer_map, regex_map.to_list())
