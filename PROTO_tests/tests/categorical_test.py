import os
import tempfile
import unittest

import numpy as np
import pytest

import arkouda as ak
from arkouda import io, io_util


def get_categorical(prefix: str = "string", size: int = 11) -> ak.Categorical:
    return ak.Categorical(ak.array([f"{prefix} {i}" for i in range(1, size)]))


def get_randomized_categorical() -> ak.Categorical:
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


class TestCategorical:
    @classmethod
    def setup_class(cls):
        cls.cat_test_base_tmp = f"{os.getcwd()}/categorical_test"
        io_util.get_directory(cls.cat_test_base_tmp)

    def test_base_categorical(self):
        cat = get_categorical()

        assert [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] == cat.codes.to_list() == cat.segments.to_list()

        assert (
            [
                "string 1",
                "string 2",
                "string 3",
                "string 4",
                "string 5",
                "string 6",
                "string 7",
                "string 8",
                "string 9",
                "string 10",
                "N/A",
            ]
            == cat.categories.to_list(),
        )
        assert 10 == cat.size
        assert "Categorical" == cat.objType

        with pytest.raises(ValueError):
            ak.Categorical(ak.arange(0, 5, 10))

    def test_categorical_from_codes_and_categories(self):
        codes = ak.array([7, 5, 9, 8, 2, 1, 4, 0, 3, 6])
        categories = ak.unique(
            ak.array(
                [
                    "string 8",
                    "string 6",
                    "string 5",
                    "string 9",
                    "string 7",
                    "string 2",
                    "string 10",
                    "string 1",
                    "string 4",
                    "string 3",
                    "N/A",
                ]
            )
        )

        cat = ak.Categorical.from_codes(codes, categories)
        assert codes.to_list() == cat.codes.to_list()
        assert categories.to_list() == cat.categories.to_list()

    def test_contains(self):
        cat = get_categorical()
        assert cat.contains("string").all()

    def test_ends_with(self):
        cat = get_categorical()
        assert cat.endswith("1").any()

    def test_starts_with(self):
        cat = get_categorical()
        assert cat.startswith("string").all()

    def test_group(self):
        group = get_randomized_categorical().group()
        assert [0, 4, 8, 1, 6, 2, 5, 9, 3, 7] == group.to_list()

    def test_unique(self):
        cat = get_randomized_categorical()

        assert (
            ak.Categorical(
                ak.array(["string", "string1", "string3", "non-string", "non-string2"])
            ).to_list()
            == cat.unique().to_list(),
        )

    def test_to_ndarray(self):
        cat = get_randomized_categorical()
        nd_cat = np.array(
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
        assert cat.to_list() == nd_cat.tolist()

    def test_equality(self):
        cat = get_categorical()
        cat_dupe = get_categorical()
        cat_non_dupe = get_randomized_categorical()

        assert cat.to_list() == cat_dupe.to_list()
        assert not (cat == cat_non_dupe).any()

        c1 = ak.Categorical(ak.array(["a", "b", "c", "a", "b"]))
        c2 = ak.Categorical(ak.array(["a", "x", "c", "y", "b"]))
        res = c1 == c2
        assert res.to_list() == [True, False, True, False, True]

    def test_binop(self):
        cat = get_categorical()
        cat_dupe = get_categorical()
        cat_non_dupe = get_randomized_categorical()

        assert cat._binop(cat_dupe, "==").all()
        assert cat._binop(cat_non_dupe, "!=").all()

        assert ak.ones(10, dtype=bool).to_list() == cat._binop(cat_dupe, "==").to_list()

        assert ak.zeros(10, dtype=bool).to_list() == cat._binop(cat_dupe, "!=").to_list()

        assert (ak.arange(10) == 0).to_list() == cat._binop("string 1", "==").to_list()

        assert (ak.arange(10) == 0).to_list() == cat._binop(np.str_("string 1"), "==").to_list()

        assert (ak.arange(10) != 0).to_list() == cat._binop("string 1", "!=").to_list()

        assert (ak.arange(10) != 0).to_list() == cat._binop(np.str_("string 1"), "!=").to_list()

        with pytest.raises(NotImplementedError):
            cat._binop("string 1", "===")

        with pytest.raises(TypeError):
            cat._binop(1, "==")

    def test_in1d(self):
        vals = [i % 3 for i in range(10)]
        vals_two = [i % 2 for i in range(10)]

        strings_one = ak.array(["String {}".format(i) for i in vals])
        strings_two = ak.array(["String {}".format(i) for i in vals_two])
        cat_one = ak.Categorical(strings_one)
        cat_two = ak.Categorical(strings_two)

        answer = [x < 2 for x in vals]

        assert answer == ak.in1d(cat_one, cat_two).to_list()
        assert answer == ak.in1d(cat_one, strings_two).to_list()

        with pytest.raises(TypeError):
            ak.in1d(cat_one, ak.randint(0, 5, 5))

    def test_where(self):
        revs = ak.arange(10) % 2 == 0
        cat1 = ak.Categorical(ak.array([f"str {i}" for i in range(10)]))

        # str in categories, cat first
        str_in_cat = "str 1"
        ans = ak.where(revs, cat1, str_in_cat)
        assert cat1[revs].to_list() == ans[revs].to_list()
        for s in ans[~revs].to_list():
            assert s == str_in_cat

        # str in categories, str first
        ans = ak.where(revs, str_in_cat, cat1)
        assert cat1[~revs].to_list() == ans[~revs].to_list()
        for s in ans[revs].to_list():
            assert s == str_in_cat

        # str not in categories, cat first
        str_not_in_cat = "str 122222"
        ans = ak.where(revs, cat1, str_not_in_cat)
        assert cat1[revs].to_list() == ans[revs].to_list()
        for s in ans[~revs].to_list():
            assert s == str_not_in_cat

        # str not in categories, str first
        ans = ak.where(revs, str_not_in_cat, cat1)
        assert cat1[~revs].to_list() == ans[~revs].to_list()
        for s in ans[revs].to_list():
            assert s == str_not_in_cat

        # 2 categorical, same categories
        cat2 = ak.Categorical(ak.array([f"str {i}" for i in range(9, -1, -1)]))
        ans = ak.where(revs, cat1, cat2)
        assert cat1[revs].to_list() == ans[revs].to_list()
        assert cat2[~revs].to_list() == ans[~revs].to_list()

        # 2 categorical, different categories
        cat2 = ak.Categorical(ak.array([f"str {i*2}" for i in range(10)]))
        ans = ak.where(revs, cat1, cat2)
        assert cat1[revs].to_list() == ans[revs].to_list()
        assert cat2[~revs].to_list() == ans[~revs].to_list()

    def test_concatenate(self):
        cat_one = get_categorical("string", 51)
        cat_two = get_categorical("string-two", 51)

        result_cat = cat_one.concatenate([cat_two])
        assert "Categorical" == result_cat.objType
        assert isinstance(result_cat, ak.Categorical)
        assert 100 == result_cat.size

        # Since Categorical.concatenate uses Categorical.from_codes method, confirm
        # that both permutation and segments are None
        assert not result_cat.permutation
        assert not result_cat.segments

        result_cat = ak.concatenate([cat_one, cat_one], ordered=False)
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
        # Ordered concatenation
        s12_ord = ak.concatenate([s1, s2], ordered=True)
        c12_ord = ak.concatenate([c1, c2], ordered=True)
        assert ak.Categorical(s12_ord).to_list() == c12_ord.to_list()
        # Unordered (but still deterministic) concatenation
        s12_unord = ak.concatenate([s1, s2], ordered=False)
        c12_unord = ak.concatenate([c1, c2], ordered=False)
        assert ak.Categorical(s12_unord).to_list() == c12_unord.to_list()

        # Tiny concatenation
        # Used to fail when length of array was less than numLocales
        # CI uses 2 locales, so try with length-1 arrays
        a = ak.Categorical(ak.array(["a"]))
        b = ak.Categorical(ak.array(["b"]))
        c = ak.concatenate((a, b), ordered=False)
        ans = ak.Categorical(ak.array(["a", "b"]))
        assert c.to_list() == ans.to_list()

    def test_save_and_load_categorical(self):
        """
        Test to save categorical to hdf5 and read it back successfully
        """
        num_elems = 51  # _getCategorical starts counting at 1, so the size is really off by one
        cat = get_categorical(size=num_elems)
        with pytest.raises(ValueError):
            # Expect error for mode not being append or truncate
            cat.to_hdf("foo", dataset="bar", mode="not_allowed")

        with tempfile.TemporaryDirectory(dir=self.cat_test_base_tmp) as tmp_dirname:
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
            d = f[dset_name]
            f.close()

            # Now try to read them back with load_all
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            assert dset_name in x
            cat_from_hdf = x[dset_name]

            expected_categories = [f"string {i}" for i in range(1, num_elems)] + ["N/A"]

            # Note assertCountEqual asserts a and b have the same elements
            # in the same amount regardless of order
            unittest.TestCase().assertCountEqual(cat_from_hdf.categories.to_list(), expected_categories)

            # Asserting the optional components and sizes are correct
            # for both constructors should be sufficient
            assert cat_from_hdf.segments is not None
            assert cat_from_hdf.permutation is not None
            print(f"==> cat_from_hdf.size:{cat_from_hdf.size}")
            assert cat_from_hdf.size == num_elems - 1

    def test_hdf_update(self):
        num_elems = 51  # _getCategorical starts counting at 1, so the size is really off by one
        cat = get_categorical(size=num_elems)
        with tempfile.TemporaryDirectory(dir=self.cat_test_base_tmp) as tmp_dirname:
            dset_name = "categorical_array"  # name of categorical array
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name)

            dset_name2 = "to_replace"
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name2, mode="append")

            dset_name3 = "cat_array2"
            cat.to_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name3, mode="append")

            replace_cat = get_categorical(size=23)
            replace_cat.update_hdf(f"{tmp_dirname}/cat-save-test", dataset=dset_name2)

            data = ak.read_hdf(f"{tmp_dirname}/cat-save-test_*")
            assert dset_name in data
            assert dset_name2 in data
            assert dset_name3 in data

            d = data[dset_name2]
            assert d.codes.to_list() == replace_cat.codes.to_list()
            assert d.permutation.to_list() == replace_cat.permutation.to_list()
            assert d.segments.to_list() == replace_cat.segments.to_list()
            assert d._akNAcode.to_list() == replace_cat._akNAcode.to_list()
            assert d.categories.to_list() == replace_cat.categories.to_list()

    def test_unused_categories_logic(self):
        """
        Test that Categoricals built from_codes and from slices
        that have unused categories behave correctly
        """
        s = ak.array([str(i) for i in range(10)])
        s12 = s[1:3]
        cat = ak.Categorical(s)
        cat12 = cat[1:3]
        assert ak.in1d(s, s12).to_list() == ak.in1d(cat, cat12).to_list()
        assert set(ak.unique(s12).to_list()) == set(ak.unique(cat12).to_list())

        cat_from_codes = ak.Categorical.from_codes(ak.array([1, 2]), s)
        assert ak.in1d(s, s12).to_list() == ak.in1d(cat, cat_from_codes).to_list()
        assert set(ak.unique(s12).to_list()) == set(ak.unique(cat_from_codes).to_list())

    def test_save_and_load_categorical_multi(self):
        """
        Test to build a pseudo dataframe with multiple
        categoricals, pdarrays, strings objects and successfully
        write/read it from HDF5
        """
        c1 = get_categorical(prefix="c1", size=51)
        c2 = get_categorical(prefix="c2", size=52)
        pda1 = ak.zeros(51)
        strings1 = ak.random_strings_uniform(9, 10, 52)

        with tempfile.TemporaryDirectory(dir=self.cat_test_base_tmp) as tmp_dirname:
            df = {"cat1": c1, "cat2": c2, "pda1": pda1, "strings1": strings1}
            ak.to_hdf(df, f"{tmp_dirname}/cat-save-test")
            x = ak.load_all(path_prefix=f"{tmp_dirname}/cat-save-test")
            assert len(x.items()) == 4
            # Note assertCountEqual asserts a and b have the same
            # elements in the same amount regardless of order
            unittest.TestCase().assertCountEqual(x["cat1"].categories.to_list(), c1.categories.to_list())
            unittest.TestCase().assertCountEqual(x["cat2"].categories.to_list(), c2.categories.to_list())
            unittest.TestCase().assertCountEqual(x["pda1"].to_list(), pda1.to_list())
            unittest.TestCase().assertCountEqual(x["strings1"].to_list(), strings1.to_list())

    def test_isna(self):
        s = ak.array(["A", "B", "C", "B", "C"])
        # NAval present in categories
        c = ak.Categorical(s, NAvalue="C")
        assert c.isna().to_list() == [False, False, True, False, True]
        assert c.NAvalue == "C"
        # Test that NAval survives registration
        c.register("my_categorical")
        c2 = ak.attach("my_categorical")
        assert c2.NAvalue == "C"

        c.unregister()

        # default NAval not present in categories
        c = ak.Categorical(s)
        assert not c.isna().any()
        assert c.NAvalue == "N/A"

    def test_standardize_categories(self):
        c1 = ak.Categorical(ak.array(["A", "B", "C"]))
        c2 = ak.Categorical(ak.array(["B", "C", "D"]))
        c3, c4 = ak.Categorical.standardize_categories([c1, c2])
        assert c3.categories.to_list() == c4.categories.to_list()
        assert not c3.isna().any()
        assert not c4.isna().any()
        assert c3.categories.size == c1.categories.size + 1
        assert c4.categories.size == c2.categories.size + 1

    def test_lookup(self):
        keys = ak.array([1, 2, 3])
        values = ak.Categorical(ak.array(["A", "B", "C"]))
        args = ak.array([3, 2, 1, 0])
        ret = ak.lookup(keys, values, args)
        assert ret.to_list() == ["C", "B", "A", "N/A"]

    def test_deletion(self):
        cat = ak.Categorical(ak.array(["a", "b", "c"]))
        # validate registration with server
        assert len(ak.list_symbol_table()) > 0

        # set to none and validate no entries in symbol table
        cat = None
        assert len(ak.list_symbol_table()) == 0
