from collections import Counter, namedtuple
from typing import List

import numpy as np
import pandas as pd
import pytest

from typeguard import TypeCheckError

import arkouda as ak

from arkouda import Strings
from arkouda.numpy import strings
from arkouda.testing import assert_equal as ak_assert_equal
from arkouda.testing import assert_equivalent


ak.verbose = False
N = 100
UNIQUE = N // 4


@pytest.mark.requires_chapel_module("EncodingMsg")
class TestString:
    def test_strings_docstrings(self):
        import doctest

        result = doctest.testmod(strings, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    Gremlins = namedtuple(
        "Gremlins",
        "gremlins_base_words gremlins_strings gremlins_test_strings gremlins_cat",
    )

    def setup_class(self):
        # Need this here so that we are connected to server
        self.gremlins = ak.array(np.array(['"', " ", ""]))

    @staticmethod
    def base_words(size):
        base_words1 = ak.random_strings_uniform(1, 10, size // 4, characters="printable")
        base_words2 = ak.random_strings_lognormal(2, 0.25, size // 4, characters="printable")
        base_words = ak.concatenate((base_words1, base_words2))
        np_base_words = np.hstack((base_words1.to_ndarray(), base_words2.to_ndarray()))

        return base_words, np_base_words

    @staticmethod
    def get_strings(size, base_words):
        choices = ak.randint(0, (size // 4) * 2, size)
        strings = base_words[choices]
        return strings

    def _get_ak_gremlins(self, size):
        choices = ak.randint(0, (size // 4) * 2, size)
        base_words, _ = self.base_words(size)
        gremlins_base_words = ak.concatenate((base_words, self.gremlins))
        gremlins_strings = ak.concatenate((base_words[choices], self.gremlins))
        gremlins_test_strings = gremlins_strings.to_ndarray()
        gremlins_cat = ak.Categorical(gremlins_strings)
        return self.Gremlins(gremlins_base_words, gremlins_strings, gremlins_test_strings, gremlins_cat)

    def delim(self, base_words):
        x, w = tuple(zip(*Counter("".join(base_words)).items()))
        g = self.gremlins.to_ndarray()
        delim = np.random.choice(x, p=(np.array(w) / sum(w)))
        if delim in g:
            self.delim(base_words)
        return delim

    @staticmethod
    def compare_strings(s1, s2):
        return all(x == y for x, y in zip(s1, s2))

    def test_equals(self):
        s = ak.array(["a", "b", "c"])
        s_cpy = ak.array(["a", "b", "c"])
        assert ak.sum((s == s_cpy) != ak.array([True, True, True])) == 0
        assert ak.sum((s != s_cpy) != ak.array([False, False, False])) == 0
        assert s.equals(s_cpy)

        s2 = ak.array(["a", "x", "c"])
        assert ak.sum((s == s2) != ak.array([True, False, True])) == 0
        assert ak.sum((s != s2) != ak.array([False, True, False])) == 0
        assert not s.equals(s2)

        s3 = ak.array(["a", "b", "c", "d"])
        assert not s.equals(s3)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_compare_strings(self, size):
        base_words, np_base_words = self.base_words(size)
        assert self.compare_strings(base_words.to_ndarray(), np_base_words)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_unique(self, size):
        base_words, _ = self.base_words(size)
        strings = self.get_strings(size, base_words)
        # unique
        ak_uniq = ak.unique(strings)
        cat = ak.Categorical(strings)
        cat_uniq = ak.unique(cat)
        akset = set(ak_uniq.to_ndarray())
        catset = set(cat_uniq.to_ndarray())
        assert akset == catset
        # There should be no duplicates
        assert ak_uniq.size == len(akset)
        npset = set(np.unique(strings.to_ndarray()))
        # When converted to a set, should agree with numpy
        assert akset == npset

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_groupby(self, size):
        base_words, _ = self.base_words(size)
        strings = self.get_strings(size, base_words)
        akset = set(ak.unique(strings).to_ndarray())
        g = ak.GroupBy(strings)
        cat = ak.Categorical(strings)
        gc = ak.GroupBy(cat)
        # Unique keys should be same result as ak.unique
        assert akset == set(g.unique_keys.to_ndarray())
        assert akset == set(gc.unique_keys.to_ndarray())
        assert gc.permutation.tolist() == g.permutation.tolist()
        permStrings = strings[g.permutation].to_ndarray()
        # Check each group individually
        lengths = np.diff(np.hstack((g.segments.to_ndarray(), np.array([g.length]))))
        for uk, s, l in zip(g.unique_keys.to_ndarray(), g.segments.to_ndarray(), lengths):
            # All values in group should equal key
            assert (permStrings[s : s + l] == uk).all()
            # Key should not appear anywhere outside of group
            assert not (permStrings[:s] == uk).any()
            assert not (permStrings[s + l :] == uk).any()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_index(self, size):
        base_words, _ = self.base_words(size)
        strings = self.get_strings(size, base_words)
        cat = ak.Categorical(strings)
        test_strings = strings.to_ndarray()

        assert strings[size // 3] == test_strings[size // 3]
        assert cat[size // 3] == test_strings[size // 3]
        for i in range(-len(self.gremlins), 0):
            assert strings[i] == test_strings[i]
            assert cat[i] == test_strings[i]

        g = self._get_ak_gremlins(size)
        assert g.gremlins_strings[size // 3] == g.gremlins_test_strings[size // 3]
        assert g.gremlins_cat[size // 3] == g.gremlins_test_strings[size // 3]
        for i in range(-len(self.gremlins), 0):
            assert g.gremlins_strings[i] == g.gremlins_test_strings[i]
            assert g.gremlins_cat[i] == g.gremlins_test_strings[i]

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_slice(self, size):
        base_words, _ = self.base_words(size)
        strings = self.get_strings(size, base_words)
        test_strings = strings.to_ndarray()
        cat = ak.Categorical(strings)
        assert self.compare_strings(
            strings[size // 4 : size // 3].to_ndarray(),
            test_strings[size // 4 : size // 3],
        )
        assert self.compare_strings(
            cat[size // 4 : size // 3].to_ndarray(), test_strings[size // 4 : size // 3]
        )

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_pdarray_index(self, size):
        base_words, _ = self.base_words(size)
        strings = self.get_strings(size, base_words)
        test_strings = strings.to_ndarray()
        cat = ak.Categorical(strings)

        inds = ak.arange(0, strings.size, 10)
        assert self.compare_strings(strings[inds].to_ndarray(), test_strings[inds.to_ndarray()])
        assert self.compare_strings(cat[inds].to_ndarray(), test_strings[inds.to_ndarray()])
        logical = ak.zeros(strings.size, dtype=ak.bool_)
        logical[inds] = True
        assert self.compare_strings(strings[logical].to_ndarray(), test_strings[logical.to_ndarray()])
        # Indexing with a one-element pdarray (int) should return Strings array, not string scalar
        i = size // 2
        singleton = ak.array([i])
        result = strings[singleton]
        assert isinstance(result, ak.Strings) and (result.size == 1)
        assert result[0] == strings[i]
        # Logical indexing with all-False array should return empty Strings array
        logicalSingleton = ak.zeros(strings.size, dtype=ak.bool_)
        result = strings[logicalSingleton]
        assert isinstance(result, ak.Strings) and (result.size == 0)
        # Logical indexing with a single True should return one-element Strings array, not string scalar
        logicalSingleton[i] = True
        result = strings[logicalSingleton]
        assert isinstance(result, ak.Strings) and (result.size == 1)
        assert result[0] == strings[i]

    @staticmethod
    def _contains_help(strings, test_strings, delim):
        if isinstance(delim, bytes):
            delim = delim.decode()
        found = strings.contains(delim).to_ndarray()
        npfound = np.array([s.count(delim) > 0 for s in test_strings])
        assert (found == npfound).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_contains(self, size):
        base_words, np_base_words = self.base_words(size)
        strings = self.get_strings(size, base_words)
        test_strings = strings.to_ndarray()
        delim = self.delim(np_base_words)
        self._contains_help(strings, test_strings, delim)
        self._contains_help(strings, test_strings, np.str_(delim))
        self._contains_help(strings, test_strings, str.encode(str(delim)))

    @staticmethod
    def _starts_with_help(strings, test_strings, delim):
        if isinstance(delim, bytes):
            delim = delim.decode()
        found = strings.startswith(delim).to_ndarray()
        npfound = np.array([s.startswith(delim) for s in test_strings])
        assert (found == npfound).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_starts_with(self, size):
        base_words, np_base_words = self.base_words(size)
        strings = self.get_strings(size, base_words)
        test_strings = strings.to_ndarray()
        delim = self.delim(np_base_words)

        self._starts_with_help(strings, test_strings, delim)
        self._starts_with_help(strings, test_strings, np.str_(delim))
        self._starts_with_help(strings, test_strings, str.encode(str(delim)))

    @staticmethod
    def _ends_with_help(strings, test_strings, delim):
        if isinstance(delim, bytes):
            delim = delim.decode()
        found = strings.endswith(delim).to_ndarray()
        npfound = np.array([s.endswith(delim) for s in test_strings])
        if len(found) != len(npfound):
            raise AttributeError("found and npfound are of different lengths")
        assert (found == npfound).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_ends_with(self, size):
        base_words, np_base_words = self.base_words(size)
        strings = self.get_strings(size, base_words)
        test_strings = strings.to_ndarray()
        delim = self.delim(np_base_words)
        self._ends_with_help(strings, test_strings, delim)
        self._ends_with_help(strings, test_strings, np.str_(delim))
        self._ends_with_help(strings, test_strings, str.encode(str(delim)))

        # Test gremlins delimiters
        g = self._get_ak_gremlins(size)
        self._ends_with_help(g.gremlins_strings, g.gremlins_test_strings, " ")
        self._ends_with_help(g.gremlins_strings, g.gremlins_test_strings, '"')
        with pytest.raises(ValueError):
            # updated to raise ValueError since regex doesn't currently support patterns
            # matching empty string
            assert not self._ends_with_help(g.gremlins_strings, g.gremlins_test_strings, "")

    def test_ends_with_delimiter_match(self):
        strings = ak.array(["string{} ".format(i) for i in range(0, 5)])
        assert (strings.endswith(" ").to_ndarray()).all()

        strings = ak.array(['string{}"'.format(i) for i in range(0, 5)])
        assert (strings.endswith('"').to_ndarray()).all()

        strings = ak.array(["string{}$".format(i) for i in range(0, 5)])
        assert (strings.endswith("$").to_ndarray()).all()

        strings = ak.array(["string{}yyz".format(i) for i in range(0, 5)])
        assert (strings.endswith("z").to_ndarray()).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_error_handling(self, size):
        stringsOne = ak.random_strings_uniform(1, 10, size // 4, characters="printable")
        stringsTwo = ak.random_strings_uniform(1, 10, size // 4, characters="printable")

        with pytest.raises(TypeCheckError):
            stringsOne.lstick(stringsTwo, delimiter=1)

        with pytest.raises(TypeCheckError):
            stringsOne.lstick([1], 1)

        with pytest.raises(TypeCheckError):
            stringsOne.startswith(1)

        with pytest.raises(TypeCheckError):
            stringsOne.endswith(1)

        with pytest.raises(TypeCheckError):
            stringsOne.contains(1)

        with pytest.raises(TypeCheckError):
            stringsOne.peel(1)

        with pytest.raises(ValueError):
            stringsOne.peel("", -5)

    @staticmethod
    def _peel_help(strings, test_strings, delim):
        if isinstance(delim, bytes):
            delim = delim.decode()
        import itertools as it

        tf = (True, False)

        def munge(triple, inc, part):
            ret = []
            for h, s, t in triple:
                if not part and s == "":
                    ret.append(("", h))
                else:
                    if inc:
                        ret.append((h + s, t))
                    else:
                        ret.append((h, t))
            l, r = tuple(zip(*ret))
            return np.array(l), np.array(r)

        def rmunge(triple, inc, part):
            ret = []
            for h, s, t in triple:
                if not part and s == "":
                    ret.append((t, ""))
                else:
                    if inc:
                        ret.append((h, s + t))
                    else:
                        ret.append((h, t))
            l, r = tuple(zip(*ret))
            return np.array(l), np.array(r)

        def slide(triple, delim):
            h, s, t = triple
            h2, s2, t2 = t.partition(delim)
            newh = h + s + h2
            return newh, s2, t2

        def rslide(triple, delim):
            h, s, t = triple
            h2, s2, t2 = h.rpartition(delim)
            newt = t2 + s + t
            return h2, s2, newt

        for times, inc, part in it.product(range(1, 4), tf, tf):
            ls, rs = strings.peel(delim, times=times, includeDelimiter=inc, keepPartial=part)
            triples = [s.partition(delim) for s in test_strings]
            for _ in range(times - 1):
                triples = [slide(t, delim) for t in triples]
            ltest, rtest = munge(triples, inc, part)
            assert (ltest == ls.to_ndarray()).all() and (rtest == rs.to_ndarray()).all()

        for times, inc, part in it.product(range(1, 4), tf, tf):
            ls, rs = strings.rpeel(delim, times=times, includeDelimiter=inc, keepPartial=part)
            triples = [s.rpartition(delim) for s in test_strings]
            for _ in range(times - 1):
                triples = [rslide(t, delim) for t in triples]
            ltest, rtest = rmunge(triples, inc, part)
            assert (ltest == ls.to_ndarray()).all() and (rtest == rs.to_ndarray()).all()

    @staticmethod
    def convert_to_ord(s: List[str]) -> List[int]:
        return [ord(i) for i in "\x00".join(s)] + [ord("\x00")]

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_peel(self, size):
        base_words, np_base_words = self.base_words(size)
        strings = self.get_strings(size, base_words)
        test_strings = strings.to_ndarray()
        delim = self.delim(np_base_words)
        self._peel_help(strings, test_strings, delim)
        self._peel_help(strings, test_strings, np.str_(delim))
        self._peel_help(strings, test_strings, str.encode(str(delim)))

        # Test gremlins delimiters
        g = self._get_ak_gremlins(size)
        with pytest.raises(ValueError):
            self._peel_help(g.gremlins_strings, g.gremlins_test_strings, "")
        self._peel_help(g.gremlins_strings, g.gremlins_test_strings, '"')
        self._peel_help(g.gremlins_strings, g.gremlins_test_strings, " ")

        # Run a test with a specific set of strings to verify strings.bytes matches expected output
        series = pd.Series(["k1:v1", "k2:v2", "k3:v3", "no_colon"])
        pda = ak.from_series(series, "string")

        # Convert Pandas series of strings into a byte array where each string is terminated
        # by a null byte.
        # This mimics what should be stored server-side in the strings.bytes pdarray
        expected_series_dec = self.convert_to_ord(series.tolist())
        actual_dec = pda._comp_to_ndarray("values").tolist()  # pda.bytes.tolist()
        assert expected_series_dec == actual_dec

        # Now perform the peel and verify
        a, b = pda.peel(":")
        expected_a = self.convert_to_ord(["k1", "k2", "k3", ""])
        expected_b = self.convert_to_ord(["v1", "v2", "v3", "no_colon"])
        assert expected_a == a._comp_to_ndarray("values").tolist()
        assert expected_b == b._comp_to_ndarray("values").tolist()

    def test_peel_delimiter_length_issue(self):
        # See Issue 838
        d = "-" * 25  # 25 dashes as delimiter
        series = pd.Series([f"abc{d}xyz", f"small{d}dog", f"blue{d}hat", "last"])
        pda = ak.from_series(series)
        a, b = pda.peel(d)
        assert ["abc", "small", "blue", ""] == a.tolist()
        assert ["xyz", "dog", "hat", "last"] == b.tolist()

        # Try a slight permutation since we were able to get both versions to fail at one point
        series = pd.Series([f"abc{d}xyz", f"small{d}dog", "last"])
        pda = ak.from_series(series)
        a, b = pda.peel(d)
        assert ["abc", "small", ""] == a.tolist()
        assert ["xyz", "dog", "last"] == b.tolist()

    @staticmethod
    def _stick_help(strings, test_strings, base_words, delim, size):
        if isinstance(delim, bytes):
            delim = delim.decode()
        test_strings2 = np.random.choice(base_words.to_ndarray(), size, replace=True)
        strings2 = ak.array(test_strings2)
        stuck = strings.stick(strings2, delimiter=delim).to_ndarray()
        tstuck = np.array([delim.join((a, b)) for a, b in zip(test_strings, test_strings2)])
        assert (stuck == tstuck).all()
        assert ((strings + strings2) == strings.stick(strings2, delimiter="")).all()

        lstuck = strings.lstick(strings2, delimiter=delim).to_ndarray()
        tlstuck = np.array([delim.join((b, a)) for a, b in zip(test_strings, test_strings2)])
        assert (lstuck == tlstuck).all()
        assert ((strings2 + strings) == strings.lstick(strings2, delimiter="")).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_stick(self, size):
        base_words, np_base_words = self.base_words(size)
        strings = self.get_strings(size, base_words)
        test_strings = strings.to_ndarray()
        delim = self.delim(np_base_words)
        self._stick_help(strings, test_strings, base_words, delim, size)
        self._stick_help(strings, test_strings, base_words, np.str_(delim), size)
        self._stick_help(strings, test_strings, base_words, str.encode(str(delim)), size)

        # Test gremlins delimiters
        g = self._get_ak_gremlins(size)
        for delim in " ", "", '"':
            self._stick_help(
                g.gremlins_strings,
                g.gremlins_test_strings,
                g.gremlins_base_words,
                delim,
                size + 3,
            )

    def test_str_output(self):
        strings = ak.array(["string {}".format(i) for i in range(0, 101)])
        str_ans = "['string 0', 'string 1', 'string 2', ... , 'string 98', 'string 99', 'string 100']"
        assert str_ans == str(strings)

    def test_split(self):
        orig = ak.array(["one|two", "three|four|five", "six"])
        flat, mapping = orig.split("|", return_segments=True)
        assert flat.tolist() == ["one", "two", "three", "four", "five", "six"]
        assert mapping.tolist() == [0, 2, 5]
        thirds = [ak.cast(ak.arange(i, 99, 3), "str") for i in range(3)]
        thickrange = thirds[0].stick(thirds[1], delimiter=", ").stick(thirds[2], delimiter=", ")
        flatrange = thickrange.split(", ")
        assert ak.cast(flatrange, "int64").tolist(), np.arange(99).tolist()

    def test_get_lengths(self):
        base = ["one", "two", "three", "four", "five"]
        s1 = ak.array(base)
        lengths = s1.get_lengths()
        assert [len(x) for x in base] == lengths.tolist()

    def test_strip(self):
        s = ak.array([" Jim1", "John1   ", "Steve1 2"])
        assert s.strip(" 12").tolist() == ["Jim", "John", "Steve"]
        assert s.strip("12 ").tolist() == ["Jim", "John", "Steve"]
        assert s.strip("1 2").tolist() == ["Jim", "John", "Steve"]

        s = ak.array([" Jim", "John 1", "Steve1 2  "])
        assert s.strip().tolist() == ["Jim", "John 1", "Steve1 2"]

        s = ak.array(["\nStrings ", " \n StringS \r", "bbabStringS \r\t "])
        assert s.strip().tolist() == ["Strings", "StringS", "bbabStringS"]

        s = ak.array(["abcStringsbac", "cabStringScc", "bbabStringS abc"])
        assert s.strip("abc").tolist() == ["Strings", "StringS", "StringS "]

        s = ak.array(["\nStrings ", " \n StringS \r", " \t   StringS \r\t "])
        assert s.strip().tolist() == ["Strings", "StringS", "StringS"]

    def test_case_change(self):
        mixed = ak.array([f"StrINgS hErE {i}" for i in range(10)])

        lower = mixed.lower()
        assert lower.tolist() == [f"strings here {i}" for i in range(10)]

        upper = mixed.upper()
        assert upper.tolist() == [f"STRINGS HERE {i}" for i in range(10)]

        title = mixed.title()
        assert title.tolist() == [f"Strings Here {i}" for i in range(10)]

        capital = mixed.capitalize()
        assert capital.tolist() == [f"Strings here {i}" for i in range(10)]

        # first 10 all lower, second 10 mixed case (not lower, upper, or title), third 10 all upper,
        # last 10 all title
        lmut = ak.concatenate([lower, mixed, upper, title])

        islower = lmut.islower()
        expected = 10 > ak.arange(40)
        assert islower.tolist() == expected.tolist()

        isupper = lmut.isupper()
        expected = (30 > ak.arange(40)) & (ak.arange(40) >= 20)
        assert isupper.tolist() == expected.tolist()

        istitle = lmut.istitle()
        expected = ak.arange(40) >= 30
        assert istitle.tolist() == expected.tolist()

    def test_string_isalnum(self):
        not_alnum = ak.array([f"%Strings {i}" for i in range(3)])
        alnum = ak.array([f"Strings{i}" for i in range(3)])
        example = ak.concatenate([not_alnum, alnum])
        assert example.isalnum().tolist() == [False, False, False, True, True, True]

    def test_string_isalpha(self):
        not_alpha = ak.array([f"%Strings {i}" for i in range(3)])
        alpha = ak.array(["StringA", "StringB", "StringC"])
        example = ak.concatenate([not_alpha, alpha])
        assert example.isalpha().tolist() == [False, False, False, True, True, True]

        example2 = ak.array(
            [
                "",
                "string1",
                "stringA",
                "String",
                "12345",
                "Hello\tWorld",
                " ",
                "\n",
                "3.14",
                "\u0030",
                "\u00b2",
            ]
        )

        expected = [
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

        assert example2.isalpha().tolist() == expected

    def test_string_isdecimal(self):
        not_decimal = ak.array([f"Strings {i}" for i in range(3)])
        decimal = ak.array([f"12{i}" for i in range(3)])
        example = ak.concatenate([not_decimal, decimal])
        assert example.isdecimal().tolist() == [False, False, False, True, True, True]

        example2 = ak.array(
            [
                "",
                "string1",
                "stringA",
                "String",
                "12345",
                "Hello\tWorld",
                " ",
                "\n",
                "3.14",
                "\u0030",  # Unicode for zero
                "\u00b2",
                "2³₇",  # additional tests for super/subscripts
                "2³x₇",
            ]
        )

        expected = [
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        ]

        assert example2.isdecimal().tolist() == expected

    def test_string_isdigit(self):
        not_digit = ak.array([f"Strings {i}" for i in range(3)])
        digit = ak.array([f"12{i}" for i in range(3)])
        example = ak.concatenate([not_digit, digit])
        assert example.isdigit().tolist() == [False, False, False, True, True, True]

        example2 = ak.array(
            [
                "",
                "string1",
                "stringA",
                "String",
                "12345",
                "Hello\tWorld",
                " ",
                "\n",
                "3.14",
                "\u0030",  # Unicode for zero
                "\u00b2",
                "2³₇",  # additional tests for super/subscripts
                "2³x₇",
            ]
        )

        expected = [
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
        ]

        assert example2.isdigit().tolist() == expected

    def test_string_empty(self):
        not_empty = ak.array([f"Strings {i}" for i in range(3)])
        empty = ak.array(["" for i in range(3)])
        example = ak.concatenate([not_empty, empty])

        assert example.isempty().tolist() == [False, False, False, True, True, True]

        example2 = ak.array(
            [
                "",
                "string1",
                "stringA",
                "String",
                "12345",
                "Hello\tWorld",
                " ",
                "\n",
                "3.14",
                "\u0030",
                "\u00b2",
            ]
        )

        expected = [
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]

        assert example2.isempty().tolist() == expected

    def test_string_isspace(self):
        not_space = ak.array([f"Strings {i}" for i in range(3)])
        space = ak.array([" ", "\t", "\n", "\v", "\f", "\r", " \t\n\v\f\r"])
        example = ak.concatenate([not_space, space])
        assert example.isspace().tolist() == [
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]

        example2 = ak.array(
            [
                "",
                "string1",
                "stringA",
                "String",
                "12345",
                "Hello\tWorld",
                " ",
                "\n",
                "3.14",
                "\u0030",
                "\u00b2",
            ]
        )

        expected = [
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
        ]

        assert example2.isspace().tolist() == expected

    def test_where(self):
        revs = ak.arange(10) % 2 == 0
        s1 = ak.array([f"str {i}" for i in range(10)])

        # SegString and str literal
        str_lit = "str 1222222"
        ans = ak.where(revs, s1, str_lit)
        assert s1[revs].tolist() == ans[revs].tolist()
        for s in ans[~revs].tolist():
            assert s == str_lit

        # str literal first
        ans = ak.where(revs, str_lit, s1)
        assert s1[~revs].tolist() == ans[~revs].tolist()
        for s in ans[revs].tolist():
            assert s == str_lit

        # 2 SegStr
        s2 = ak.array([f"str {i * 2}" for i in range(10)])
        ans = ak.where(revs, s1, s2)
        assert s1[revs].tolist() == ans[revs].tolist()
        assert s2[~revs].tolist() == ans[~revs].tolist()

    @staticmethod
    def _get_strings(prefix: str = "string", size: int = 11) -> ak.Strings:
        return ak.array(["{} {}".format(prefix, i) for i in range(1, size)])

    def test_concatenate(self):
        s1 = self._get_strings("string", 51)
        s2 = self._get_strings("string-two", 51)

        resultStrings = ak.concatenate([s1, s2])
        assert isinstance(resultStrings, ak.Strings)
        assert 100 == resultStrings.size

        resultStrings = ak.concatenate([s1, s1], ordered=False)
        assert isinstance(resultStrings, ak.Strings)
        assert 100 == resultStrings.size

        s1 = self._get_strings("string", 6)
        s2 = self._get_strings("string-two", 6)
        expected_result = [
            "string 1",
            "string 2",
            "string 3",
            "string 4",
            "string 5",
            "string-two 1",
            "string-two 2",
            "string-two 3",
            "string-two 4",
            "string-two 5",
        ]

        # Ordered concatenation
        s12ord = ak.concatenate([s1, s2], ordered=True)
        assert expected_result == s12ord.tolist()
        # Unordered (but still deterministic) concatenation
        # TODO: the unordered concatenation test is disabled per #710 #721
        # s12unord = ak.concatenate([s1, s2], ordered=False)

    def test_get_fixes(self):
        a = ["abc", "d", "efghi", "xyz", "12", "m"]
        strings = ak.array(a)
        for n in 1, 3:  # test multiple sizes of suffix and prefix
            prefix, origin = strings.get_prefixes(n, return_origins=True, proper=True)
            assert [x[0:n] for x in a if len(x) > n] == prefix.tolist()
            assert [True if len(x) > n else False for x in a]

            prefix, origin = strings.get_prefixes(n, return_origins=True, proper=False)
            assert [x[0:n] for x in a if len(x) >= n] == prefix.tolist()
            assert [True if len(x) >= n else False for x in a]

            prefix = strings.get_prefixes(n, return_origins=False, proper=False)
            assert [x[0:n] for x in a if len(x) >= n] == prefix.tolist()

            suffix, origin = strings.get_suffixes(n, return_origins=True, proper=True)
            assert [x[len(x) - n :] for x in a if len(x) > n] == suffix.tolist()
            assert [True if len(x) >= n else False for x in a]

            suffix, origin = strings.get_suffixes(n, return_origins=True, proper=False)
            assert [x[len(x) - n :] for x in a if len(x) >= n] == suffix.tolist()
            assert [True if len(x) >= n else False for x in a]

            suffix = strings.get_suffixes(n, return_origins=False, proper=False)
            assert [x[len(x) - n :] for x in a if len(x) >= n] == suffix.tolist()

    def test_encoding(self):
        idna_strings = ak.array(
            ["Bücher.example", "ドメイン.テスト", "домен.испытание", "Königsgäßchen"]
        )
        expected = ak.array(
            [
                "xn--bcher-kva.example",
                "xn--eckwd4c7c.xn--zckzah",
                "xn--d1acufc.xn--80akhbyknj4f",
                "xn--knigsgchen-b4a3dun",
            ]
        )
        assert (idna_strings.encode("idna") == expected).all()

        # IDNA test
        a1 = ["münchen", "zürich"]
        s1 = ak.array(a1)
        result = s1.encode("idna")
        assert [i.encode("idna").decode("ascii") for i in a1] == result.tolist()

        # validate encoding with empty string
        e = ["", "abc", "", "123"]
        ak_e = ak.array(e)
        result = ak_e.encode("idna")
        assert [i.encode("idna").decode("ascii") for i in e] == result.tolist()

        a2 = [
            "xn--mnchen-3ya",
            "xn--zrich-kva",
            "xn--zrich-boguscode",
            "xn--!!",
            "example.com",
        ]
        s2 = ak.array(a2)
        result = s2.decode("idna")
        # using the below assertion due to a bug in `Strings.to_ndarray`. See issue #1828
        assert ["münchen", "zürich", "", "", "example.com"] == result.tolist()

        a3 = ak.random_strings_uniform(1, 10, 100 // 4, characters="printable")
        assert (a3 == a3.encode("ascii").decode("ascii")).all()

    def test_idna_utf16(self):
        ex = ["xn--mnchen-3ya", "xn--zrich-kva", "example.com"]
        s = ak.array(ex)

        # go from idna -> utf-16
        result = s.encode(fromEncoding="idna", toEncoding="utf-16")

        # roundtrip to return back to decoded values in UTF-8
        decoded = result.decode(fromEncoding="utf-16", toEncoding="utf-8")
        assert ["münchen", "zürich", "example.com"] == decoded.tolist()

    def test_tondarray(self):
        v1 = ["münchen", "zürich", "abc", "123", ""]
        s1 = ak.array(v1)
        nd1 = s1.to_ndarray()
        assert nd1.tolist() == v1

    def test_inferred_type(self):
        a = ak.array(["a", "b", "c"])
        assert a.inferred_type == "string"

    def test_string_broadcast(self):
        keys = ak.randint(0, 10, 100, int)
        while ak.unique(keys).size != 10:
            # keep generating until every element appears at least once
            keys = ak.randint(0, 10, 100, int)

        g = ak.GroupBy(keys)
        str_vals = ak.random_strings_uniform(0, 3, 10, characters="printable")
        str_broadcast_ans = str_vals[keys]

        gb_broadcasted = g.broadcast(str_vals)
        manual_broadcasted = ak.broadcast(g.segments, str_vals, permutation=g.permutation)
        assert (gb_broadcasted == str_broadcast_ans).all()
        assert (manual_broadcasted == str_broadcast_ans).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_in1d(self, size):
        base_words, _ = self.base_words(size)
        strings = self.get_strings(size, base_words)
        cat = ak.Categorical(strings)

        akwords = strings
        more_words = akwords.to_ndarray()
        matches = ak.in1d(strings, akwords)
        catmatches = ak.in1d(cat, akwords)
        assert (matches == catmatches).all()
        # Every word in matches should be in the target set
        for word in strings[matches].to_ndarray():
            assert word in more_words
        # Exhaustively find all matches to make sure we didn't miss any
        inds = ak.zeros(size, dtype=ak.bool_)
        for word in more_words:
            inds |= strings == word
        assert (inds == matches).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_flatten(self, size):
        base_words, _ = self.base_words(size)
        strings = self.get_strings(size, base_words)

        ak_assert_equal(strings.flatten(), strings)

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_concatenate_uniquely(self, size):
        import random

        from arkouda import Strings

        base_array = [str(i) for i in range(size)]
        list_1 = []
        for i in range(size // 5):
            val = random.choice(base_array)
            list_1.append(val)
            base_array.remove(val)

        base_array = [str(i) for i in range(size)]
        list_2 = []
        for i in range(size // 5):
            val = random.choice(base_array)
            list_2.append(val)
            base_array.remove(val)

        ak_arr_1 = ak.array(list_1)
        ak_arr_2 = ak.array(list_2)

        output = Strings.concatenate_uniquely([ak_arr_1, ak_arr_2])
        correct = set(list_1).union(set(list_2))

        assert set(output.to_ndarray()) == correct

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_argsort(self, size):
        base_words, _ = self.base_words(size)
        strings = self.get_strings(size, base_words)
        akperm = ak.argsort(strings)
        aksorted = strings[akperm].to_ndarray()
        npsorted = np.sort(strings.to_ndarray())
        assert (aksorted == npsorted).all()
        cat = ak.Categorical(strings)
        catperm = ak.argsort(cat)
        catsorted = cat[catperm].to_ndarray()
        assert (catsorted == npsorted).all()

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("ascending", [True, False])
    def test_argsort_basic(self, size, ascending):
        strings = ak.random_strings_uniform(minlen=1, maxlen=8, size=size, seed=pytest.seed)
        result = strings.argsort(ascending=ascending)
        np_strings = strings.to_ndarray()
        np_result = np_strings.argsort(stable=True)
        if not ascending:
            np_result = np.flip(np_result)
        assert_equivalent(result, np_result)

    def test_argsort_empty(self):
        strings = ak.array([], dtype="str_")
        result = strings.argsort()
        assert result.size == 0

    def test_argsort_duplicates(self):
        strings = ak.array(["apple", "banana", "apple", "banana"])
        result = strings.argsort()
        sorted_values = strings[result].tolist()
        assert sorted_values == sorted(strings.tolist())

    def test_argsort_case_sensitive(self):
        strings = ak.array(["Apple", "apple", "Banana", "banana"])
        result = strings.argsort()
        sorted_values = strings[result].tolist()
        assert sorted_values == sorted(strings.tolist())

    def test_argsort_unicode(self):
        strings = ak.array(["😀", "a", "🎉", "b", "👍"])
        result = strings.argsort()
        sorted_values = strings[result].tolist()
        assert sorted_values == sorted(strings.tolist())

    def test_copy(self):
        fixed_size = 100
        a = ak.arange(fixed_size, dtype=ak.str_)
        a_cpy = a.copy()

        assert a_cpy is not a
        ak_assert_equal(a, a_cpy)

    def test_copy_empty(self):
        a = ak.array([], dtype="str_")
        a_cpy = a.copy()

        assert a_cpy is not a

        assert isinstance(a_cpy, Strings)
        assert a_cpy.size == 0
