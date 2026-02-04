import os
import tempfile

import numpy as np
import pytest

from pandas import Categorical as pd_Categorical

import arkouda as ak

from arkouda.numpy.pdarraycreation import array
from arkouda.pandas import io, io_util
from arkouda.pandas.categorical import Categorical
from arkouda.testing import assert_categorical_equal, assert_equal, assert_equivalent


SEED = 12345


@pytest.fixture
def df_test_base_tmp(request):
    df_test_base_tmp = "{}/.categorical_test".format(os.getcwd())
    io_util.get_directory(df_test_base_tmp)

    # Define a finalizer function for teardown
    def finalizer():
        # Clean up any resources if needed
        io_util.delete_directory(df_test_base_tmp)

    # Register the finalizer to ensure cleanup
    request.addfinalizer(finalizer)
    return df_test_base_tmp


class TestCategorical:
    def test_categorical_docstrings(self):
        import doctest

        from arkouda.pandas import categorical

        result = doctest.testmod(
            categorical, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @classmethod
    def setup_class(cls):
        cls.non_unique_strs = [
            "string",
            "string1",
            "non-string",
            "non-string2",
            "string",
            "non-string",
            "string3",
            "non-string2",
            "string",
            "non-string",
        ]
        cls.non_unique_cat = ak.Categorical(ak.array(cls.non_unique_strs))

    def create_basic_categorical(self, prefix: str = "string", size: int = 10) -> ak.Categorical:
        return ak.Categorical(ak.array([f"{prefix} {i}" for i in range(size)]))

    def create_randomized_categorical(self) -> ak.Categorical:
        return ak.Categorical(
            ak.array(
                [
                    "string",
                    "string1",
                    "non-string",
                    "non-string2",
                    "string",
                    "non-string",
                    "string3",
                    "non-string2",
                    "string",
                    "non-string",
                ]
            )
        )

    def test_basic_categorical(self):
        prefix, size = "string", 10
        cat = self.create_basic_categorical(prefix, size)

        assert list(range(size)) == cat.codes.tolist() == cat.segments.tolist()
        assert ([f"{prefix} {i}" for i in range(size)] + ["N/A"]) == cat.categories.tolist()
        assert size == cat.size
        assert "Categorical" == cat.objType

        with pytest.raises(ValueError):
            ak.Categorical(ak.arange(0, 5, 10))

    def test_inferred_type(self):
        prefix, size = "string", 10
        cat = self.create_basic_categorical(prefix, size)
        assert cat.inferred_type == "categorical"

    def test_equals(self):
        c = Categorical(ak.array(["a", "b", "c"]))
        c_cpy = Categorical(ak.array(["a", "b", "c"]))
        assert ak.sum((c == c_cpy) != ak.array([True, True, True])) == 0
        assert ak.sum((c != c_cpy) != ak.array([False, False, False])) == 0
        assert c.equals(c_cpy)

        c2 = Categorical(ak.array(["a", "x", "c"]))
        assert ak.sum((c == c2) != ak.array([True, False, True])) == 0
        assert ak.sum((c != c2) != ak.array([False, True, False])) == 0
        assert not c.equals(c2)

        c3 = Categorical(ak.array(["a", "b", "c", "d"]))
        assert not c.equals(c3)

    def test_from_codes(self):
        codes = ak.array([7, 5, 9, 8, 2, 1, 4, 0, 3, 6])
        categories = ak.array([f"string {i}" for i in range(10)] + ["N/A"])

        cat = ak.Categorical.from_codes(codes, categories)
        assert codes.tolist() == cat.codes.tolist()
        assert categories.tolist() == cat.categories.tolist()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_from_pd_categorical(self, size):
        strings1 = ak.random_strings_uniform(1, 2, size)
        pd_cat = pd_Categorical(strings1.to_ndarray())
        ak_cat = ak.Categorical(pd_cat)

        assert np.array_equal(pd_cat.to_numpy(), ak_cat.to_pandas().to_numpy())
        assert np.array_equal(pd_cat.codes.astype("int64"), ak_cat.codes.to_ndarray())

        filter = ak_cat.categories != "N/A"
        assert np.array_equal(pd_cat.categories.values, ak_cat.categories[filter].to_ndarray())

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_creation_from_categorical(self, size):
        strings1 = ak.random_strings_uniform(1, 2, size)
        pd_cat = pd_Categorical(strings1.to_ndarray())
        ak_cat = ak.Categorical(pd_cat)
        expected_cat = ak.Categorical(strings1)
        assert_categorical_equal(ak_cat, expected_cat)

        ak_cat2 = ak.Categorical(ak.Categorical(strings1))
        expected_cat = ak.Categorical(strings1)
        assert_categorical_equal(ak_cat2, expected_cat)

    def test_substring_search(self):
        cat = self.create_basic_categorical()
        # update to use regex flag once #2714 is resolved
        assert cat.contains("ing").all()
        assert cat.endswith("1").any()
        assert cat.startswith("string").all()

        assert cat.contains("\\w", regex=True).all()
        assert cat.endswith("ing \\d", regex=True).all()

    def test_group(self):
        non_unique_cat = self.non_unique_cat
        grouped = non_unique_cat[non_unique_cat.group()]

        # to verify we correctly grouped, make sure we never re-encounter
        seen_before = set()
        curr = ""
        for val in grouped.tolist():
            if val != curr:
                assert val not in seen_before
                curr = val
                seen_before.add(val)

    def test_unique(self):
        non_unique_cat = self.non_unique_cat
        unique_cat = ak.Categorical(
            ak.array(["string", "string1", "string3", "non-string", "non-string2"])
        )
        assert non_unique_cat.unique().tolist() == unique_cat.tolist()

    def test_to_ndarray(self):
        cat = self.create_randomized_categorical()
        ndcat = np.array(
            [
                "string",
                "string1",
                "non-string",
                "non-string2",
                "string",
                "non-string",
                "string3",
                "non-string2",
                "string",
                "non-string",
            ]
        )
        assert (cat.to_ndarray() == ndcat).all()

    def test_tolist(self):
        assert self.non_unique_cat.tolist() == np.array(self.non_unique_strs).tolist()

    def test_to_strings(self):
        cat = self.non_unique_cat
        cat_list = [
            "string",
            "string1",
            "non-string",
            "non-string2",
            "string",
            "non-string",
            "string3",
            "non-string2",
            "string",
            "non-string",
        ]

        assert cat.to_strings().tolist() == cat_list
        assert isinstance(cat.to_strings(), ak.Strings)

    def test_equality(self):
        cat = self.create_basic_categorical()
        cat_dupe = self.create_basic_categorical()
        cat_non_dupe = self.non_unique_cat

        assert cat.tolist() == cat_dupe.tolist()
        assert (cat != cat_non_dupe).all()

        c1 = ak.Categorical(ak.array(["a", "b", "c", "a", "b"]))
        c2 = ak.Categorical(ak.array(["a", "x", "c", "y", "b"]))
        assert (c1 == c2).tolist() == [True, False, True, False, True]

    def test_binop(self):
        size = 10
        cat = self.create_basic_categorical(size=size)
        cat_dupe = self.create_basic_categorical(size=size)
        cat_non_dupe = self.non_unique_cat

        assert cat._binop(cat_dupe, "==").all()
        assert cat._binop(cat_non_dupe, "!=").all()

        for i in range(size):
            for op in "==", "!=":
                ans = ak.arange(10)._binop(i, op).tolist()
                assert ans == cat._binop(f"string {i}", op).tolist()
                assert ans == cat._binop(np.str_(f"string {i}"), op).tolist()

        with pytest.raises(NotImplementedError):
            cat._binop("string 1", "===")

        with pytest.raises(TypeError):
            cat._binop(1, "==")

    def test_in1d(self):
        vals = [i % 3 for i in range(10)]
        cat_one = ak.Categorical(ak.array([f"String {i}" for i in vals]))
        strings_two = ak.array([f"String {i % 2}" for i in range(10)])
        cat_two = ak.Categorical(strings_two)

        answer = [x < 2 for x in vals]

        assert answer == ak.in1d(cat_one, cat_two).tolist()
        assert answer == ak.in1d(cat_one, strings_two).tolist()

        with pytest.raises(TypeError):
            ak.in1d(cat_one, ak.randint(0, 5, 5))

    def test_where(self):
        revs = ak.arange(10) % 2 == 0
        cat1 = self.create_basic_categorical(prefix="str", size=10)

        # test string literal in and not in categories
        for str_lit in "str 1", "str 122222":
            ans = ak.where(revs, cat1, str_lit)
            assert cat1[revs].tolist() == ans[revs].tolist()
            for s in ans[~revs].tolist():
                assert s == str_lit

            ans = ak.where(revs, str_lit, cat1)
            assert cat1[~revs].tolist() == ans[~revs].tolist()
            for s in ans[revs].tolist():
                assert s == str_lit

        # 2 categorical, same and different categories
        for cat2 in (
            ak.Categorical(ak.array([f"str {i}" for i in range(9, -1, -1)])),
            ak.Categorical(ak.array([f"str {i * 2}" for i in range(10)])),
        ):
            ans = ak.where(revs, cat1, cat2)
            assert cat1[revs].tolist() == ans[revs].tolist()
            assert cat2[~revs].tolist() == ans[~revs].tolist()

    def test_where_cat_with_scalar_existing_and_new_labels(self):
        revs = ak.arange(6) % 2 == 0
        cat = ak.Categorical(ak.array(["a", "a", "a", "a", "a", "a"]))

        # existing label (after we seed one)
        cat2 = ak.Categorical(ak.array(["b", "a", "a", "a", "a", "a"]))
        out_existing = ak.where(revs, "b", cat2)
        assert out_existing.to_ndarray().tolist() == ["b", "a", "b", "a", "b", "a"]

        # new label not in categories
        out_new = ak.where(revs, "None", cat)
        assert out_new.to_ndarray().tolist() == ["None", "a", "None", "a", "None", "a"]

    def test_where_scalar_then_cat_symmetry(self):
        mask = ak.array([True, False, True, False])
        cat = ak.Categorical(ak.array(["x", "y", "y", "y"]))

        out1 = ak.where(mask, cat, "z")  # new label
        out2 = ak.where(mask, "z", cat)  # symmetric path
        assert out1.to_ndarray().tolist() == ["x", "z", "y", "z"]
        assert out2.to_ndarray().tolist() == ["z", "y", "z", "y"]

    def test_concatenate(self):
        cat_one = self.create_basic_categorical("string", 50)
        cat_two = self.create_basic_categorical("string-two", 50)

        for result_cat in (
            cat_one.concatenate([cat_two]),
            ak.concatenate([cat_one, cat_one], ordered=False),
        ):
            assert "Categorical" == result_cat.objType
            assert isinstance(result_cat, ak.Categorical)
            assert 100 == result_cat.size

            # Since Categorical.concatenate uses Categorical.from_codes method, confirm
            # that both permutation and segments are None
            assert not result_cat.permutation
            assert not result_cat.segments

        # Concatenate two Categoricals with different categories,
        # and test result against original strings
        s1 = ak.array(["abc", "de", "abc", "fghi", "de"])
        s2 = ak.array(["jkl", "mno", "fghi", "abc", "fghi", "mno"])
        c1 = ak.Categorical(s1)
        c2 = ak.Categorical(s2)
        for order in True, False:
            str_concat = ak.concatenate([s1, s2], ordered=order)
            cat_concat = ak.concatenate([c1, c2], ordered=order)
            assert str_concat.tolist() == cat_concat.tolist()

        # Tiny concatenation
        # Used to fail when length of array was less than numLocales
        # CI uses 2 locales, so try with length-1 arrays
        a = ak.Categorical(ak.array(["a"]))
        b = ak.Categorical(ak.array(["b"]))
        assert ak.concatenate((a, b), ordered=False).tolist() == ak.array(["a", "b"]).tolist()

    @pytest.mark.requires_chapel_module("HDF5Msg")
    def test_save_and_load_categorical(self, df_test_base_tmp):
        """Test to save categorical to hdf5 and read it back successfully."""
        num_elems = 51  # create_basic_categorical starts counting at 1, so the size is really off by one
        cat = self.create_basic_categorical(size=num_elems)
        with pytest.raises(TypeError):
            # Expect error for mode not being append or truncate
            cat.to_hdf("foo", dataset="bar", mode="not_allowed")

        with tempfile.TemporaryDirectory(dir=df_test_base_tmp) as tmp_dirname:
            dset_name = "categorical_array"  # name of categorical array

            # Test the save functionality & confirm via h5py
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name)

            import h5py

            f = h5py.File(tmp_dirname + "/cat-save-test_LOCALE0000", mode="r")
            keys = list(f.keys())
            if io.ARKOUDA_HDF5_FILE_METADATA_GROUP in keys:  # Ignore the metadata group if it exists
                keys.remove(io.ARKOUDA_HDF5_FILE_METADATA_GROUP)
            assert len(keys) == 1, f"Expected 1 key, {dset_name}"
            assert [dset_name] == keys
            d = f[dset_name]  # noqa: F841
            f.close()

            # Now try to read them back with load_all
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            assert dset_name in x
            cat_from_hdf = x[dset_name]

            expected_categories = [f"string {i}" for i in range(0, num_elems)] + ["N/A"]

            # Note assertCountEqual asserts a and b have the same elements
            # in the same amount regardless of order
            assert cat_from_hdf.categories.tolist() == expected_categories

            # Asserting the optional components and sizes are correct
            # for both constructors should be sufficient
            assert cat_from_hdf.segments is not None
            assert cat_from_hdf.permutation is not None
            assert cat_from_hdf.size == num_elems

    @pytest.mark.requires_chapel_module("HDF5Msg")
    def test_save_and_load_categorical_multi(self, df_test_base_tmp):
        """
        Test to build a pseudo dataframe with multiple
        categoricals, pdarrays, strings objects and successfully
        write/read it from HDF5
        """
        c1 = self.create_basic_categorical(prefix="c1", size=51)
        c2 = self.create_basic_categorical(prefix="c2", size=52)
        pda1 = ak.zeros(51)
        strings1 = ak.random_strings_uniform(9, 10, 52)

        with tempfile.TemporaryDirectory(dir=df_test_base_tmp) as tmp_dirname:
            df = {"cat1": c1, "cat2": c2, "pda1": pda1, "strings1": strings1}
            ak.to_hdf(df, f"{tmp_dirname}/cat-save-test")
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            assert len(x.items()) == 4
            # Note assertCountEqual asserts a and b have the same
            # elements in the same amount regardless of order
            assert x["cat1"].categories.tolist() == c1.categories.tolist()
            assert x["cat2"].categories.tolist() == c2.categories.tolist()
            assert x["pda1"].tolist() == pda1.tolist()
            assert x["strings1"].tolist() == strings1.tolist()

    @pytest.mark.requires_chapel_module("HDF5Msg")
    def test_hdf_update(self, df_test_base_tmp):
        num_elems = 51  # create_basic_categorical starts counting at 1, so the size is really off by one
        cat = self.create_basic_categorical(size=num_elems)
        with tempfile.TemporaryDirectory(dir=df_test_base_tmp) as tmp_dirname:
            dset_name = "categorical_array"  # name of categorical array
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name)

            dset_name2 = "to_replace"
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name2, mode="append")

            dset_name3 = "cat_array2"
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name3, mode="append")

            replace_cat = self.create_basic_categorical(size=23)
            replace_cat.update_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name2)

            data = ak.read_hdf(f"{tmp_dirname}/cat-save-test_*")
            assert dset_name in data
            assert dset_name2 in data
            assert dset_name3 in data

            d = data[dset_name2]
            assert d.codes.tolist() == replace_cat.codes.tolist()
            assert d.permutation.tolist() == replace_cat.permutation.tolist()
            assert d.segments.tolist() == replace_cat.segments.tolist()
            assert d._akNAcode.tolist() == replace_cat._akNAcode.tolist()
            assert d.categories.tolist() == replace_cat.categories.tolist()

    def test_unused_categories_logic(self):
        # Reproducer for issue #990
        s = ak.array([str(i) for i in range(10)])
        s12 = s[1:3]
        cat = ak.Categorical(s)
        cat12 = cat[1:3]
        assert ak.in1d(s, s12).tolist() == ak.in1d(cat, cat12).tolist()
        assert set(ak.unique(s12).tolist()) == set(ak.unique(cat12).tolist())

        cat_from_codes = ak.Categorical.from_codes(ak.array([1, 2]), s)
        assert ak.in1d(s, s12).tolist() == ak.in1d(cat, cat_from_codes).tolist()
        assert set(ak.unique(s12).tolist()) == set(ak.unique(cat_from_codes).tolist())

    def test_na(self):
        s = ak.array(["A", "B", "C", "B", "C"])
        # NAval present in categories
        c = ak.Categorical(s, na_value="C")
        assert (c.isna() == (s == "C")).all()
        assert c.na_value == "C"
        # Test that NAval survives registration
        c.register("my_categorical")
        c2 = ak.attach("my_categorical")
        assert c2.na_value == "C"

        c.unregister()

        # default NAval not present in categories
        c = ak.Categorical(s)
        assert not c.isna().any()
        assert c.na_value == "N/A"

    def test_standardize_categories(self):
        c1 = ak.Categorical(ak.array(["A", "B", "C"]))
        c2 = ak.Categorical(ak.array(["B", "C", "D"]))
        c3, c4 = ak.Categorical.standardize_categories([c1, c2])
        assert c3.categories.tolist() == c4.categories.tolist()
        assert not c3.isna().any() and not c4.isna().any()
        assert c1.categories.size + 1 == c3.categories.size == c4.categories.size

    def test_lookup(self):
        keys = ak.array([1, 2, 3])
        values = ak.Categorical(ak.array(["A", "B", "C"]))
        args = ak.array([3, 2, 1, 0])
        ret = ak.numpy.alignment.lookup(keys, values, args)
        assert ret.tolist() == ["C", "B", "A", "N/A"]

    def test_deletion(self):
        cat = ak.Categorical(ak.array(["a", "b", "c"]))
        # validate registration with server
        assert len(ak.list_symbol_table()) > 0

        # set to none and validate no entries in symbol table
        cat = None  # noqa: F841
        assert len(ak.list_symbol_table()), 0

    def test_sort(self):
        rand_cats = ak.random_strings_uniform(1, 16, 10)
        rand_codes = ak.randint(0, rand_cats.size, 100)
        cat = ak.Categorical.from_codes(codes=rand_codes, categories=rand_cats)

        assert sorted(cat.tolist()) == cat.sort_values().tolist()

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_to_pandas(self, size):
        strings1 = ak.random_strings_uniform(1, 2, size)
        ak_cat = ak.Categorical(strings1)
        pd_cat = ak_cat.to_pandas()

        expected_cat = pd_Categorical.from_codes(
            codes=ak_cat.codes.to_ndarray(), categories=ak_cat.categories.to_ndarray()
        )

        assert np.array_equal(pd_cat.to_numpy(), expected_cat.to_numpy())
        assert np.array_equal(pd_cat.codes.astype("int64"), expected_cat.codes.astype("int64"))
        assert np.array_equal(pd_cat.categories.values, expected_cat.categories.values)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("ascending", [True, False])
    def test_argsort_basic(self, size, ascending):
        strings = ak.random_strings_uniform(
            minlen=1, maxlen=8, size=size, seed=SEED if pytest.seed is None else pytest.seed
        )
        cat = Categorical(strings)
        result = cat.argsort(ascending=ascending)
        np_strings = cat.to_ndarray()
        np_result = np_strings.argsort(stable=True)
        if not ascending:
            np_result = np.flip(np_result)
        assert_equivalent(result, np_result)

    def test_argsort_empty(self):
        strings = ak.array([], dtype="str_")
        cat = ak.Categorical(strings)
        result = cat.argsort()
        assert result.size == 0

    def test_argsort_duplicates(self):
        strings = array(["apple", "banana", "apple", "banana"])
        cat = Categorical(strings)
        result = cat.argsort()
        sorted_values = cat[result].tolist()
        assert sorted_values == sorted(strings.tolist())

    def test_argsort_case_sensitive(self):
        strings = array(["Apple", "apple", "Banana", "banana"])
        cat = Categorical(strings)
        result = cat.argsort()
        sorted_values = cat[result].tolist()
        assert sorted_values == sorted(strings.tolist())

    def test_argsort_unicode(self):
        strings = array(["😀", "a", "🎉", "b", "👍"])
        cat = Categorical(strings)
        result = cat.argsort()
        sorted_values = strings[result].tolist()
        assert sorted_values == sorted(strings.tolist())

    def basic_cat(self):
        # categories: ["a", "b", "c"], codes map to labels
        categories = ak.array(["a", "b", "c"])  # Strings
        codes = ak.array([0, 1, 2, 1, 0])  # pdarray[int64]
        return Categorical.from_codes(codes, categories)

    def cat_with_perm_segments(self):
        # Build a small example with explicit permutation/segments metadata.
        categories = ak.array(["red", "green", "blue"])
        codes = ak.array([2, 0, 2, 1, 1, 0])  # labels: blue, red, blue, green, green, red
        # Example permutation/segments (shape/meaning may vary by impl;
        # these are just nontrivial, valid-length pdarrays to pass through)
        permutation = ak.array([1, 5, 3, 4, 0, 2])  # some reordering indices
        segments = ak.array([0, 2, 4])  # example segment starts
        return Categorical.from_codes(codes, categories, permutation=permutation, segments=segments)

    def empty_cat(self):
        return Categorical(ak.array([], dtype="str"))

    # ------------------------------ Basic copy behavior ------------------------------

    def test_copy_returns_new_instance(self):
        c = self.basic_cat()
        c2 = c.copy()
        assert isinstance(c2, Categorical)
        assert c2 is not c
        assert len(c2) == len(c)
        assert_equal(c, c2)

    def test_copy_is_deep_for_internal_arrays(self):
        c = self.basic_cat()
        c2 = c.copy()

        # Internals should be distinct objects with equal contents
        assert c is not c2
        assert_categorical_equal(c, c2, check_category_order=True)

        # permutation/segments might be None; if present, they should also be deep-copied
        for name in ("permutation", "segments"):
            a = getattr(c, name, None)
            b = getattr(c2, name, None)
            assert (a is None) == (b is None)
            if a is not None:
                assert_equal(b, a)

    # ------------------------------ Metadata preserved ------------------------------

    def test_copy_preserves_metadata_when_present(self):
        c = self.cat_with_perm_segments()
        c2 = c.copy()

        # Distinct instances
        assert c2 is not c
        assert c2.codes is not c.codes
        assert c2.categories is not c.categories
        assert c2.segments is not c.segments
        assert c2.permutation is not c.permutation

        # Distinct internals; equal contents
        assert_equal(c2.permutation, c.permutation)
        assert_equal(c2.segments, c.segments)
        assert_categorical_equal(c, c2, check_category_order=True)

    # ------------------------------ Empty case ------------------------------

    def test_copy_empty_categorical(self):
        empty_cat = self.empty_cat()
        c2 = empty_cat.copy()
        assert len(c2) == 0
        assert c2.categories is not empty_cat.categories
        assert c2.codes is not empty_cat.codes

        assert_categorical_equal(empty_cat, c2, check_category_order=True)

        # permutation/segments may be None or empty; only check distinctness if present
        for name in ("permutation", "segments"):
            a = getattr(empty_cat, name, None)
            b = getattr(c2, name, None)
            assert (a is None) == (b is None)
            if a is not None:
                assert a is not b
                assert_equal(a, b)

    def test_categorical_NAvalue_deprecated_returns_find_na(self):
        cat = Categorical(ak.array(["a", "b"]))
        with pytest.deprecated_call():
            assert cat.NAvalue == cat.na_value
