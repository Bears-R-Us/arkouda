from collections import Counter, namedtuple
from typing import List, Tuple

import numpy as np
import pandas as pd
from base_test import ArkoudaTest
from context import arkouda as ak

ak.verbose = False
N = 100
UNIQUE = N // 4


def compare_strings(a, b):
    return all(x == y for x, y in zip(a, b))


def convert_to_ord(s: List[str]) -> List[int]:
    return [ord(i) for i in "\x00".join(s)] + [ord("\x00")]


errors = False


def run_test_argsort(strings, test_strings, cat):
    akperm = ak.argsort(strings)
    aksorted = strings[akperm].to_ndarray()
    npsorted = np.sort(test_strings)
    assert (aksorted == npsorted).all()
    catperm = ak.argsort(cat)
    catsorted = cat[catperm].to_ndarray()
    assert (catsorted == npsorted).all()


def run_test_unique(strings, test_strings, cat):
    # unique
    akuniq = ak.unique(strings)
    catuniq = ak.unique(cat)
    akset = set(akuniq.to_ndarray())
    catset = set(catuniq.to_ndarray())
    assert akset == catset
    # There should be no duplicates
    assert akuniq.size == len(akset)
    npset = set(np.unique(test_strings))
    # When converted to a set, should agree with numpy
    assert akset == npset
    return akset


def run_test_index(strings, test_strings, cat, specificInds):
    # int index
    assert strings[N // 3] == test_strings[N // 3]
    assert cat[N // 3] == test_strings[N // 3]
    for i in specificInds:
        assert strings[i] == test_strings[i]
        assert cat[i] == test_strings[i]


def run_test_slice(strings, test_strings, cat):
    assert compare_strings(strings[N // 4 : N // 3].to_ndarray(), test_strings[N // 4 : N // 3])
    assert compare_strings(cat[N // 4 : N // 3].to_ndarray(), test_strings[N // 4 : N // 3])


def run_test_pdarray_index(strings, test_strings, cat):
    inds = ak.arange(0, strings.size, 10)
    assert compare_strings(strings[inds].to_ndarray(), test_strings[inds.to_ndarray()])
    assert compare_strings(cat[inds].to_ndarray(), test_strings[inds.to_ndarray()])
    logical = ak.zeros(strings.size, dtype=ak.bool_)
    logical[inds] = True
    assert compare_strings(strings[logical].to_ndarray(), test_strings[logical.to_ndarray()])
    # Indexing with a one-element pdarray (int) should return Strings array, not string scalar
    i = N // 2
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


def run_comparison_test(strings, test_strings, cat):
    akinds = strings == test_strings[N // 4]
    npinds = test_strings == test_strings[N // 4]
    assert np.allclose(akinds.to_ndarray(), npinds)


def run_test_in1d(strings, cat, base_words):
    more_choices = ak.randint(0, UNIQUE, 100)
    akwords = base_words[more_choices]
    more_words = akwords.to_ndarray()
    matches = ak.in1d(strings, akwords)
    catmatches = ak.in1d(cat, akwords)
    assert (matches == catmatches).all()
    # Every word in matches should be in the target set
    for word in strings[matches].to_ndarray():
        assert word in more_words
    # Exhaustively find all matches to make sure we didn't miss any
    inds = ak.zeros(strings.size, dtype=ak.bool_)
    for word in more_words:
        inds |= strings == word
    assert (inds == matches).all()


def run_test_groupby(strings, cat, akset):
    g = ak.GroupBy(strings)
    gc = ak.GroupBy(cat)
    # Unique keys should be same result as ak.unique
    assert akset == set(g.unique_keys.to_ndarray())
    assert akset == set(gc.unique_keys.to_ndarray())
    assert (gc.permutation == g.permutation).all()
    permStrings = strings[g.permutation].to_ndarray()
    # Check each group individually
    lengths = np.diff(np.hstack((g.segments.to_ndarray(), np.array([g.length]))))
    for uk, s, l in zip(g.unique_keys.to_ndarray(), g.segments.to_ndarray(), lengths):
        # All values in group should equal key
        assert (permStrings[s : s + l] == uk).all()
        # Key should not appear anywhere outside of group
        assert not (permStrings[:s] == uk).any()
        assert not (permStrings[s + l :] == uk).any()


def run_test_contains(strings, test_strings, delim):
    if isinstance(delim, bytes):
        delim = delim.decode()
    found = strings.contains(delim).to_ndarray()
    npfound = np.array([s.count(delim) > 0 for s in test_strings])
    assert (found == npfound).all()


def run_test_starts_with(strings, test_strings, delim):
    if isinstance(delim, bytes):
        delim = delim.decode()
    found = strings.startswith(delim).to_ndarray()
    npfound = np.array([s.startswith(delim) for s in test_strings])
    assert (found == npfound).all()


def run_test_ends_with(strings, test_strings, delim):
    if isinstance(delim, bytes):
        delim = delim.decode()
    found = strings.endswith(delim).to_ndarray()
    npfound = np.array([s.endswith(delim) for s in test_strings])
    if len(found) != len(npfound):
        raise AttributeError("found and npfound are of different lengths")
    assert (found == npfound).all()


def run_test_peel(strings, test_strings, delim):
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


def run_test_stick(strings, test_strings, base_words, delim, N):
    if isinstance(delim, bytes):
        delim = delim.decode()
    test_strings2 = np.random.choice(base_words.to_ndarray(), N, replace=True)
    strings2 = ak.array(test_strings2)
    stuck = strings.stick(strings2, delimiter=delim).to_ndarray()
    tstuck = np.array([delim.join((a, b)) for a, b in zip(test_strings, test_strings2)])
    assert (stuck == tstuck).all()
    assert ((strings + strings2) == strings.stick(strings2, delimiter="")).all()

    lstuck = strings.lstick(strings2, delimiter=delim).to_ndarray()
    tlstuck = np.array([delim.join((b, a)) for a, b in zip(test_strings, test_strings2)])
    assert (lstuck == tlstuck).all()
    assert ((strings2 + strings) == strings.lstick(strings2, delimiter="")).all()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ak.connect(server=sys.argv[1], port=sys.argv[2])
    else:
        ak.connect()

    print("Running test from string_test.__main__")
    # with open(__file__, 'r') as f:
    #     base_words = np.array(f.read().split())
    # test_strings = np.random.choice(base_words, N, replace=True)
    # strings = ak.array(test_strings)

    base_words1 = ak.random_strings_uniform(1, 10, UNIQUE, characters="printable")
    base_words2 = ak.random_strings_lognormal(2, 0.25, UNIQUE, characters="printable")
    gremlins = ak.array(['"', " ", ""])
    base_words = ak.concatenate((base_words1, base_words2))
    np_base_words = np.hstack((base_words1.to_ndarray(), base_words2.to_ndarray()))
    assert compare_strings(base_words.to_ndarray(), np_base_words)
    choices = ak.randint(0, base_words.size, N)
    strings = base_words[choices]
    test_strings = strings.to_ndarray()
    cat = ak.Categorical(strings)
    print("strings =", strings)
    print("categorical =", cat)
    print("Generation and concatenate passed")

    # int index
    run_test_index(strings, test_strings, cat, range(-len(gremlins), 0))
    print("int index passed")

    # slice
    run_test_slice(strings, test_strings, cat)
    print("slice passed")

    # pdarray int index
    run_test_pdarray_index(strings, test_strings, cat)
    print("pdarray int index passed")

    # comparison
    run_comparison_test(strings, test_strings, cat)
    print("comparison passed")

    # pdarray bool index
    run_test_pdarray_index(strings, test_strings, cat)
    print("pdarray bool index passed")

    # in1d and iter
    # more_words = np.random.choice(base_words, 100)
    # akwords = ak.array(more_words)
    run_test_in1d(strings, cat, base_words)
    print("in1d and iter passed")

    # argsort
    run_test_argsort(strings, test_strings, cat)

    # unique
    akset = run_test_unique(strings, test_strings, cat)

    # groupby
    run_test_groupby(strings, cat, akset)
    print("groupby passed")

    # substring functions
    x, w = tuple(zip(*Counter("".join(base_words.to_ndarray())).items()))
    delim = np.random.choice(x, p=(np.array(w) / sum(w)))

    # contains
    run_test_contains(strings, test_strings, delim)
    print("contains passed")

    # startswith
    run_test_starts_with(strings, test_strings, delim)
    print("startswith passed")

    # endswith
    run_test_ends_with(strings, test_strings, delim)
    print("endswith passed")

    # peel
    run_test_peel(strings, test_strings, delim)
    print("peel passed")

    # stick
    run_test_stick(strings, test_strings, base_words, delim, 100)
    print("stick passed")


class StringTest(ArkoudaTest):
    Gremlins = namedtuple(
        "Gremlins", "gremlins_base_words gremlins_strings gremlins_test_strings gremlins_cat"
    )

    def setUp(self):
        self.maxDiff = None
        ArkoudaTest.setUp(self)
        base_words1 = ak.random_strings_uniform(1, 10, UNIQUE, characters="printable")
        base_words2 = ak.random_strings_lognormal(2, 0.25, UNIQUE, characters="printable")
        gremlins = np.array(['"', " ", ""])
        self.gremlins = ak.array(gremlins)
        self.base_words = ak.concatenate((base_words1, base_words2))
        self.np_base_words = np.hstack((base_words1.to_ndarray(), base_words2.to_ndarray()))
        self.choices = ak.randint(0, self.base_words.size, N)
        self.strings = self.base_words[self.choices]
        self.test_strings = self.strings.to_ndarray()

        x, w = tuple(zip(*Counter("".join(self.base_words.to_ndarray())).items()))
        self.delim = self._get_delimiter(x, w, gremlins)

    def _get_strings(self, prefix: str = "string", size: int = 11) -> ak.Strings:
        return ak.array(["{} {}".format(prefix, i) for i in range(1, size)])

    def _get_delimiter(self, x: Tuple, w: Tuple, gremlins: np.ndarray) -> str:
        delim = np.random.choice(x, p=(np.array(w) / sum(w)))
        if delim in gremlins:
            self._get_delimiter(x, w, gremlins)
        return delim

    def _get_ak_gremlins(self):
        gremlins_base_words = ak.concatenate((self.base_words, self.gremlins))
        gremlins_strings = ak.concatenate((self.base_words[self.choices], self.gremlins))
        gremlins_test_strings = gremlins_strings.to_ndarray()
        gremlins_cat = ak.Categorical(gremlins_strings)
        return self.Gremlins(gremlins_base_words, gremlins_strings, gremlins_test_strings, gremlins_cat)

    def _get_categorical(self):
        return ak.Categorical(self.strings)

    def test_compare_strings(self):
        assert compare_strings(self.base_words.to_ndarray(), self.np_base_words)

    def test_equals(self):
        s = ak.array(["a", "b", "c"])
        s_cpy = ak.array(["a", "b", "c"])
        self.assertTrue(ak.sum((s == s_cpy) != ak.array([True, True, True])) == 0)
        self.assertTrue(ak.sum((s != s_cpy) != ak.array([False, False, False])) == 0)
        assert s.equals(s_cpy)

        s2 = ak.array(["a", "x", "c"])
        self.assertTrue(ak.sum((s == s2) != ak.array([True, False, True])) == 0)
        self.assertTrue(ak.sum((s != s2) != ak.array([False, True, False])) == 0)
        assert not s.equals(s2)

        s3 = ak.array(["a", "b", "c", "d"])
        assert not s.equals(s3)

    def test_argsort(self):
        run_test_argsort(self.strings, self.test_strings, self._get_categorical())

    def test_in1d(self):
        run_test_in1d(self.strings, self._get_categorical(), self.base_words)

    def test_unique(self):
        run_test_unique(self.strings, self.test_strings, self._get_categorical())

    def test_groupby(self):
        akset = set(ak.unique(self.strings).to_ndarray())
        run_test_groupby(self.strings, self._get_categorical(), akset)

    def test_index(self):
        run_test_index(
            self.strings, self.test_strings, self._get_categorical(), range(-len(self.gremlins), 0)
        )
        g = self._get_ak_gremlins()
        run_test_index(
            g.gremlins_strings, g.gremlins_test_strings, g.gremlins_cat, range(-len(self.gremlins), 0)
        )

    def test_slice(self):
        run_test_slice(self.strings, self.test_strings, self._get_categorical())

    def test_pdarray_index(self):
        run_test_pdarray_index(self.strings, self.test_strings, self._get_categorical())

    def test_contains(self):
        run_test_contains(self.strings, self.test_strings, self.delim)
        run_test_contains(self.strings, self.test_strings, np.str_(self.delim))
        run_test_contains(self.strings, self.test_strings, str.encode(str(self.delim)))

    def test_starts_with(self):
        run_test_starts_with(self.strings, self.test_strings, self.delim)
        run_test_starts_with(self.strings, self.test_strings, np.str_(self.delim))
        run_test_starts_with(self.strings, self.test_strings, str.encode(str(self.delim)))

    def test_ends_with(self):
        run_test_ends_with(self.strings, self.test_strings, self.delim)
        run_test_ends_with(self.strings, self.test_strings, np.str_(self.delim))
        run_test_ends_with(self.strings, self.test_strings, str.encode(str(self.delim)))

        # Test gremlins delimiters
        g = self._get_ak_gremlins()
        run_test_ends_with(g.gremlins_strings, g.gremlins_test_strings, " ")
        run_test_ends_with(g.gremlins_strings, g.gremlins_test_strings, '"')
        with self.assertRaises(ValueError):
            # updated to raise ValueError since regex doesn't currently support patterns
            # matching empty string
            self.assertFalse(run_test_ends_with(g.gremlins_strings, g.gremlins_test_strings, ""))

    def test_ends_with_delimiter_match(self):
        strings = ak.array(["string{} ".format(i) for i in range(0, 5)])
        self.assertTrue((strings.endswith(" ").to_ndarray()).all())

        strings = ak.array(['string{}"'.format(i) for i in range(0, 5)])
        self.assertTrue((strings.endswith('"').to_ndarray()).all())

        strings = ak.array(["string{}$".format(i) for i in range(0, 5)])
        self.assertTrue((strings.endswith("$").to_ndarray()).all())

        strings = ak.array(["string{}yyz".format(i) for i in range(0, 5)])
        self.assertTrue((strings.endswith("z").to_ndarray()).all())

    def test_error_handling(self):
        stringsOne = ak.random_strings_uniform(1, 10, UNIQUE, characters="printable")
        stringsTwo = ak.random_strings_uniform(1, 10, UNIQUE, characters="printable")

        with self.assertRaises(TypeError):
            stringsOne.lstick(stringsTwo, delimiter=1)

        with self.assertRaises(TypeError):
            stringsOne.lstick([1], 1)

        with self.assertRaises(TypeError):
            stringsOne.startswith(1)

        with self.assertRaises(TypeError):
            stringsOne.endswith(1)

        with self.assertRaises(TypeError):
            stringsOne.contains(1)

        with self.assertRaises(TypeError):
            stringsOne.peel(1)

        with self.assertRaises(ValueError):
            stringsOne.peel("", -5)

    def test_peel(self):
        run_test_peel(self.strings, self.test_strings, self.delim)
        run_test_peel(self.strings, self.test_strings, np.str_(self.delim))
        run_test_peel(self.strings, self.test_strings, str.encode(str(self.delim)))

        # Test gremlins delimiters
        g = self._get_ak_gremlins()
        with self.assertRaises(ValueError):
            run_test_peel(g.gremlins_strings, g.gremlins_test_strings, "")
        run_test_peel(g.gremlins_strings, g.gremlins_test_strings, '"')
        run_test_peel(g.gremlins_strings, g.gremlins_test_strings, " ")

        # Run a test with a specific set of strings to verify strings.bytes matches expected output
        series = pd.Series(["k1:v1", "k2:v2", "k3:v3", "no_colon"])
        pda = ak.from_series(series, "string")

        # Convert Pandas series of strings into a byte array where each string is terminated
        # by a null byte.
        # This mimics what should be stored server-side in the strings.bytes pdarray
        expected_series_dec = convert_to_ord(series.to_list())
        actual_dec = pda._comp_to_ndarray("values").tolist()  # pda.bytes.to_list()
        self.assertListEqual(expected_series_dec, actual_dec)

        # Now perform the peel and verify
        a, b = pda.peel(":")
        expected_a = convert_to_ord(["k1", "k2", "k3", ""])
        expected_b = convert_to_ord(["v1", "v2", "v3", "no_colon"])
        self.assertListEqual(expected_a, a._comp_to_ndarray("values").tolist())
        self.assertListEqual(expected_b, b._comp_to_ndarray("values").tolist())

    def test_peel_delimiter_length_issue(self):
        # See Issue 838
        d = "-" * 25  # 25 dashes as delimiter
        series = pd.Series([f"abc{d}xyz", f"small{d}dog", f"blue{d}hat", "last"])
        pda = ak.from_series(series)
        a, b = pda.peel(d)
        self.assertListEqual(["abc", "small", "blue", ""], a.to_list())
        self.assertListEqual(["xyz", "dog", "hat", "last"], b.to_list())

        # Try a slight permutation since we were able to get both versions to fail at one point
        series = pd.Series([f"abc{d}xyz", f"small{d}dog", "last"])
        pda = ak.from_series(series)
        a, b = pda.peel(d)
        self.assertListEqual(["abc", "small", ""], a.to_list())
        self.assertListEqual(["xyz", "dog", "last"], b.to_list())

    def test_stick(self):
        run_test_stick(self.strings, self.test_strings, self.base_words, self.delim, 100)
        run_test_stick(self.strings, self.test_strings, self.base_words, np.str_(self.delim), 100)
        run_test_stick(
            self.strings, self.test_strings, self.base_words, str.encode(str(self.delim)), 100
        )

        # Test gremlins delimiters
        g = self._get_ak_gremlins()
        run_test_stick(g.gremlins_strings, g.gremlins_test_strings, g.gremlins_base_words, " ", 103)
        run_test_stick(g.gremlins_strings, g.gremlins_test_strings, g.gremlins_base_words, "", 103)
        run_test_stick(g.gremlins_strings, g.gremlins_test_strings, g.gremlins_base_words, '"', 103)

    def test_str_output(self):
        strings = ak.array(["string {}".format(i) for i in range(0, 101)])
        self.assertEqual(
            "['string 0', 'string 1', 'string 2', ... , 'string 98', 'string 99', 'string 100']",
            str(strings),
        )

    def test_flatten(self):
        orig = ak.array(["one|two", "three|four|five", "six"])
        flat, mapping = orig.flatten("|", return_segments=True)
        self.assertListEqual(flat.to_list(), ["one", "two", "three", "four", "five", "six"])
        self.assertListEqual(mapping.to_list(), [0, 2, 5])
        thirds = [ak.cast(ak.arange(i, 99, 3), "str") for i in range(3)]
        thickrange = thirds[0].stick(thirds[1], delimiter=", ").stick(thirds[2], delimiter=", ")
        flatrange = thickrange.flatten(", ")
        self.assertListEqual(ak.cast(flatrange, "int64").to_list(), np.arange(99).tolist())

    def test_get_lengths(self):
        s1 = ak.array(["one", "two", "three", "four", "five"])
        lengths = s1.get_lengths()
        self.assertListEqual([3, 3, 5, 4, 4], lengths.to_list())

    def test_strip(self):
        s = ak.array([" Jim1", "John1   ", "Steve1 2"])
        self.assertListEqual(s.strip(" 12").to_list(), ["Jim", "John", "Steve"])
        self.assertListEqual(s.strip("12 ").to_list(), ["Jim", "John", "Steve"])
        self.assertListEqual(s.strip("1 2").to_list(), ["Jim", "John", "Steve"])

        s = ak.array([" Jim", "John 1", "Steve1 2  "])
        self.assertListEqual(s.strip().to_list(), ["Jim", "John 1", "Steve1 2"])

        s = ak.array(["\nStrings ", " \n StringS \r", "bbabStringS \r\t "])
        self.assertListEqual(s.strip().to_list(), ["Strings", "StringS", "bbabStringS"])

        s = ak.array(["abcStringsbac", "cabStringScc", "bbabStringS abc"])
        self.assertListEqual(s.strip("abc").to_list(), ["Strings", "StringS", "StringS "])

        s = ak.array(["\nStrings ", " \n StringS \r", " \t   StringS \r\t "])
        self.assertListEqual(s.strip().to_list(), ["Strings", "StringS", "StringS"])

    def test_case_change(self):
        mixed = ak.array([f"StrINgS hErE {i}" for i in range(10)])

        lower = mixed.lower()
        self.assertListEqual(lower.to_list(), [f"strings here {i}" for i in range(10)])

        upper = mixed.upper()
        self.assertListEqual(upper.to_list(), [f"STRINGS HERE {i}" for i in range(10)])

        title = mixed.title()
        self.assertListEqual(title.to_list(), [f"Strings Here {i}" for i in range(10)])

        capital = mixed.capitalize()
        self.assertListEqual(capital.to_list(), [f"Strings here {i}" for i in range(10)])

        # first 10 all lower, second 10 mixed case (not lower, upper, or title), third 10 all upper,
        # last 10 all title
        lmut = ak.concatenate([lower, mixed, upper, title])

        islower = lmut.islower()
        expected = 10 > ak.arange(40)
        self.assertListEqual(islower.to_list(), expected.to_list())

        isupper = lmut.isupper()
        expected = (30 > ak.arange(40)) & (ak.arange(40) >= 20)
        self.assertListEqual(isupper.to_list(), expected.to_list())

        istitle = lmut.istitle()
        expected = ak.arange(40) >= 30
        self.assertListEqual(istitle.to_list(), expected.to_list())

    def test_string_isalnum(self):
        not_alnum = ak.array([f"%Strings {i}" for i in range(3)])
        alnum = ak.array([f"Strings{i}" for i in range(3)])
        example = ak.concatenate([not_alnum, alnum])
        self.assertListEqual(example.isalnum().to_list(), [False, False, False, True, True, True])

    def test_string_isalpha(self):
        not_alpha = ak.array([f"%Strings {i}" for i in range(3)])
        alpha = ak.array(["StringA", "StringB", "StringC"])
        example = ak.concatenate([not_alpha, alpha])
        self.assertListEqual(example.isalpha().to_list(), [False, False, False, True, True, True])

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
                "\u00B2",
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

        self.assertListEqual(example2.isalpha().to_list(), expected)

    def test_string_isdecimal(self):
        not_decimal = ak.array([f"Strings {i}" for i in range(3)])
        decimal = ak.array([f"12{i}" for i in range(3)])
        example = ak.concatenate([not_decimal, decimal])
        self.assertListEqual(example.isdecimal().to_list(), [False, False, False, True, True, True])

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
                "\u00B2",
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

        self.assertListEqual(example2.isdecimal().to_list(), expected)

    def test_string_isdigit(self):
        not_digit = ak.array([f"Strings {i}" for i in range(3)])
        digit = ak.array([f"12{i}" for i in range(3)])
        example = ak.concatenate([not_digit, digit])
        self.assertListEqual(example.isdigit().to_list(), [False, False, False, True, True, True])

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
                "\u00B2",
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

        self.assertListEqual(example2.isdigit().to_list(), expected)

    def test_string_empty(self):
        not_empty = ak.array([f"Strings {i}" for i in range(3)])
        empty = ak.array(["" for i in range(3)])
        example = ak.concatenate([not_empty, empty])
        self.assertListEqual(example.isempty().to_list(), [False, False, False, True, True, True])

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
                "\u00B2",
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

        self.assertListEqual(example2.isempty().to_list(), expected)

    def test_string_isspace(self):
        not_space = ak.array([f"Strings {i}" for i in range(3)])
        space = ak.array([" ", "\t", "\n", "\v", "\f", "\r", " \t\n\v\f\r"])
        example = ak.concatenate([not_space, space])
        self.assertListEqual(
            example.isspace().to_list(), [False, False, False, True, True, True, True, True, True, True]
        )

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
                "\u00B2",
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

        self.assertListEqual(example2.isspace().to_list(), expected)

    def test_where(self):
        revs = ak.arange(10) % 2 == 0
        s1 = ak.array([f"str {i}" for i in range(10)])

        # SegString and str literal
        str_lit = "str 1222222"
        ans = ak.where(revs, s1, str_lit)
        self.assertListEqual(s1[revs].to_list(), ans[revs].to_list())
        for s in ans[~revs].to_list():
            self.assertEqual(s, str_lit)

        # str literal first
        ans = ak.where(revs, str_lit, s1)
        self.assertListEqual(s1[~revs].to_list(), ans[~revs].to_list())
        for s in ans[revs].to_list():
            self.assertEqual(s, str_lit)

        # 2 SegStr
        s2 = ak.array([f"str {i*2}" for i in range(10)])
        ans = ak.where(revs, s1, s2)
        self.assertListEqual(s1[revs].to_list(), ans[revs].to_list())
        self.assertListEqual(s2[~revs].to_list(), ans[~revs].to_list())

    def test_concatenate(self):
        s1 = self._get_strings("string", 51)
        s2 = self._get_strings("string-two", 51)

        resultStrings = ak.concatenate([s1, s2])
        self.assertIsInstance(resultStrings, ak.Strings)
        self.assertEqual(100, resultStrings.size)

        resultStrings = ak.concatenate([s1, s1], ordered=False)
        self.assertIsInstance(resultStrings, ak.Strings)
        self.assertEqual(100, resultStrings.size)

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
        self.assertListEqual(expected_result, s12ord.to_list())
        # Unordered (but still deterministic) concatenation
        # TODO: the unordered concatenation test is disabled per #710 #721
        # s12unord = ak.concatenate([s1, s2], ordered=False)

    def test_get_fixes(self):
        strings = ak.array(["abc", "d", "efghi"])
        p, r = strings.get_prefixes(1, return_origins=True, proper=True)
        self.assertListEqual(["a", "e"], p.to_list())
        self.assertListEqual([True, False, True], r.to_list())

        p, r = strings.get_prefixes(1, return_origins=True, proper=False)
        self.assertListEqual(["a", "d", "e"], p.to_list())
        self.assertListEqual([True, True, True], r.to_list())

        p = strings.get_prefixes(1, return_origins=False, proper=False)
        self.assertListEqual(["a", "d", "e"], p.to_list())

        p, r = strings.get_suffixes(1, return_origins=True, proper=True)
        self.assertListEqual(["c", "i"], p.to_list())
        self.assertListEqual([True, False, True], r.to_list())

        p, r = strings.get_suffixes(1, return_origins=True, proper=False)
        self.assertListEqual(["c", "d", "i"], p.to_list())
        self.assertListEqual([True, True, True], r.to_list())

        p = strings.get_suffixes(1, return_origins=False, proper=False)
        self.assertListEqual(["c", "d", "i"], p.to_list())

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
        self.assertTrue((idna_strings.encode("idna") == expected).all())

        # IDNA test
        a1 = ["münchen", "zürich"]
        s1 = ak.array(a1)
        result = s1.encode("idna")
        self.assertListEqual([i.encode("idna").decode("ascii") for i in a1], result.to_list())

        # validate encoding with empty string
        e = ["", "abc", "", "123"]
        ak_e = ak.array(e)
        result = ak_e.encode("idna")
        self.assertListEqual([i.encode("idna").decode("ascii") for i in e], result.to_list())

        a2 = ["xn--mnchen-3ya", "xn--zrich-kva", "xn--zrich-boguscode", "xn--!!", "example.com"]
        s2 = ak.array(a2)
        result = s2.decode("idna")
        # using the below assertion due to a bug in `Strings.to_ndarray`. See issue #1828
        self.assertListEqual(["münchen", "zürich", "", "", "example.com"], result.to_list())

        a3 = ak.random_strings_uniform(1, 10, UNIQUE, characters="printable")
        self.assertTrue((a3 == a3.encode("ascii").decode("ascii")).all())

    def test_idna_utf16(self):
        s = ak.array(["xn--mnchen-3ya", "xn--zrich-kva", "example.com"])

        # go from idna -> utf-16
        result = s.encode(fromEncoding="idna", toEncoding="utf-16")

        # roundtrip to return back to decoded values in UTF-8
        decoded = result.decode(fromEncoding="utf-16", toEncoding="utf-8")
        self.assertListEqual(["münchen", "zürich", "example.com"], decoded.to_list())

    def test_tondarray(self):
        v1 = ["münchen", "zürich", "abc", "123", ""]
        s1 = ak.array(v1)
        nd1 = s1.to_ndarray()
        self.assertListEqual(nd1.tolist(), v1)

    def test_inferred_type(self):
        a = ak.array(["a", "b", "c"])
        self.assertTrue(a.inferred_type, "string")

    def test_string_broadcast(self):
        keys = ak.randint(0, 10, 100, int)
        g = ak.GroupBy(keys)
        str_vals = ak.random_strings_uniform(0, 3, 10, characters="printable")
        str_broadcast_ans = str_vals[keys]

        gb_broadcasted = g.broadcast(str_vals)
        manual_broadcasted = ak.broadcast(g.segments, str_vals, permutation=g.permutation)
        self.assertTrue((gb_broadcasted == str_broadcast_ans).all())
        self.assertTrue((manual_broadcasted == str_broadcast_ans).all())
