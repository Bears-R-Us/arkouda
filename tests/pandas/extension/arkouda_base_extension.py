import numpy as np
from numpy.testing import assert_array_equal
import pytest

import arkouda as ak
from arkouda.pandas.extension import ArkoudaArray, ArkoudaCategoricalArray, ArkoudaStringArray
from arkouda.pandas.extension._arkouda_base_array import ArkoudaBaseArray
from arkouda.testing import assert_equal


class TestArkoudaBaseExtension:
    def test_base_extension_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _arkouda_base_array

        result = doctest.testmod(
            _arkouda_base_array, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def base_objs(self):
        """Provide canonical Arkouda objects for reuse in tests."""
        nums = ak.array([1, 2, 3, 4])  # pdarray[int64]
        strs = ak.array(["a", "bb", "ccc"])  # Strings
        cat = ak.Categorical(strs)
        return {"pdarray": nums, "Strings": strs, "Categorical": cat}

    def data_triplets(self):
        """
        Build (label, arkouda_object, expected_len) triplets for:
        - numeric pdarray
        - Strings
        - Categorical (derived from Strings)
        """
        # numeric pdarray
        pdarr = ak.array([1, 2, 3, 4])

        # Strings
        s = ak.array(["a", "bb", "ccc"])

        # Categorical derived from strings (stable and portable way to make one)
        cat = ak.Categorical(s)

        return [
            ("pdarray", pdarr, 4),
            ("Strings", s, 3),
            ("Categorical", cat, 3),
        ]

    @pytest.mark.parametrize("label_idx", [0, 1, 2])
    def test_len_matches_expected(self, label_idx):
        label, ak_obj, expected = self.data_triplets()[label_idx]
        arr = ArkoudaBaseArray(ak_obj)
        assert len(arr) == expected, f"len failed for {label}"

    @pytest.mark.parametrize("label_idx", [0, 1, 2])
    def test_len_matches_backend_len(self, label_idx):
        label, ak_obj, _ = self.data_triplets()[label_idx]
        arr = ArkoudaBaseArray(ak_obj)
        assert len(arr) == len(ak_obj), f"wrapper len mismatch for {label}"

    @pytest.mark.parametrize("label_idx", [0, 1, 2])
    def test_len_consistency_after_concat(self, label_idx):
        """
        Sanity check that length tracks when we build a bigger backend object
        and wrap it again (helps catch off-by-one issues in __len__).
        """
        label, ak_obj, expected = self.data_triplets()[label_idx]
        # duplicate and concatenate

        bigger = ak.concatenate([ak_obj, ak_obj])
        arr_big = ArkoudaBaseArray(bigger)
        assert len(arr_big) == 2 * len(ak_obj), f"concat len wrong for {label}"

    def test_len_zero_length_cases(self):
        """
        Edge cases: zero-length arrays across pdarray and Strings
        (Categorical empty creation via empty Strings).
        """
        empty_pd = ak.array([], dtype="int64")
        empty_s = ak.array([], dtype="str_")
        empty_cat = ak.Categorical(empty_s)

        for label, obj in [("pdarray", empty_pd), ("Strings", empty_s), ("Categorical", empty_cat)]:
            arr = ArkoudaBaseArray(obj)
            assert len(arr) == 0, f"expected zero length for empty {label}"

    # ------------------------ pdarray: dtype + copy semantics ------------------------

    def test_to_numpy_pdarray_default_no_args(self):
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])
        out = arr.to_numpy()
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.dtype("int64")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype="int64"))

    def test_to_numpy_pdarray_dtype_cast_to_float64(self):
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])  # int64 backend
        out = arr.to_numpy(dtype="float64")
        assert out.dtype == np.dtype("float64")
        np.testing.assert_allclose(out, np.array([1, 2, 3, 4], dtype="float64"))

    def test_to_numpy_pdarray_dtype_same_with_copy_true(self):
        """If dtype matches and copy=True, ensure result is equal and of same dtype."""
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])
        out = arr.to_numpy(dtype="int64", copy=True)
        assert out.dtype == np.dtype("int64")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype="int64"))
        # Sanity: local mutation doesn't change a freshly fetched array
        out[0] = 999
        fresh = arr.to_numpy()
        assert fresh[0] == 1

    def test_to_numpy_pdarray_copy_true_no_dtype(self):
        """copy=True with no dtype specified should still return equal values."""
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])
        out = arr.to_numpy(copy=True)
        assert out.dtype == np.dtype("int64")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype="int64"))
        out[1] = -123
        fresh = arr.to_numpy()
        assert fresh[1] == 2

    def test_to_numpy_pdarray_empty_edge_case(self):
        empty = ak.array(np.array([], dtype="int64"))
        arr = ArkoudaBaseArray(empty)
        out = arr.to_numpy()
        assert out.dtype == np.dtype("int64")
        assert out.size == 0

    # ----------------------------- Strings behavior -----------------------------

    def test_to_numpy_strings_default_to_numpy(self):
        arr = ArkoudaBaseArray(self.base_objs()["Strings"])
        out = arr.to_numpy()
        np.testing.assert_array_equal(out, np.array(["a", "bb", "ccc"], dtype=object))

    def test_to_numpy_strings_copy_true_does_not_affect_source(self):
        arr = ArkoudaBaseArray(self.base_objs()["Strings"])
        out = arr.to_numpy(copy=True)
        np.testing.assert_array_equal(out, np.array(["a", "bb", "ccc"], dtype=object))
        # local mutation shouldn't affect a fresh materialization
        out[0] = "zzz"
        fresh = arr.to_numpy()
        assert fresh[0] == "a"

    # ---------------------------- Categorical behavior ----------------------------

    def test_to_numpy_categorical_default_to_numpy_labels(self):
        cat = self.base_objs()["Categorical"]
        arr = ArkoudaBaseArray(cat)
        out = arr.to_numpy()
        # Should materialize the labels (object dtype)
        np.testing.assert_array_equal(out, np.array(["a", "bb", "ccc"], dtype=object))

    def test_to_numpy_categorical_copy_true_isolated_result(self):
        cat = self.base_objs()["Categorical"]
        arr = ArkoudaBaseArray(cat)
        out = arr.to_numpy(copy=True)
        np.testing.assert_array_equal(out, np.array(["a", "bb", "ccc"], dtype=object))
        out[-1] = "X"
        fresh = arr.to_numpy()
        assert fresh[-1] == "ccc"

    # ----------------------------- pdarray behavior -----------------------------

    def test_pdarray_to_ndarray_basic(self):
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])
        out = arr.to_ndarray()
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.dtype("int64")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype="int64"))

    def test_pdarray_to_ndarray_isolation_on_mutation(self):
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])
        out = arr.to_ndarray()
        out[0] = 999  # mutate the client array
        fresh = arr.to_ndarray()
        assert fresh[0] == 1  # backend unchanged

    def test_pdarray_empty_to_ndarray(self):
        empty = ak.array(np.array([], dtype="int64"))
        arr = ArkoudaBaseArray(empty)
        out = arr.to_ndarray()
        assert out.size == 0
        assert out.dtype == np.dtype("int64")

    # ----------------------------- Strings behavior -----------------------------

    def test_strings_to_ndarray_basic(self):
        arr = ArkoudaBaseArray(self.base_objs()["Strings"])
        out = arr.to_ndarray()
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.array(["a", "bb", "ccc"], dtype=object))

    def test_strings_to_ndarray_isolation_on_mutation(self):
        arr = ArkoudaBaseArray(self.base_objs()["Strings"])
        out = arr.to_ndarray()
        out[0] = "ZZZ"
        fresh = arr.to_ndarray()
        assert fresh[0] == "a"

    # --------------------------- Categorical behavior ---------------------------

    def test_categorical_to_ndarray_labels(self):
        arr = ArkoudaBaseArray(self.base_objs()["Categorical"])
        out = arr.to_ndarray()
        # Materializes labels as object dtype
        np.testing.assert_array_equal(out, np.array(["a", "bb", "ccc"], dtype=object))

    def test_categorical_to_ndarray_isolation_on_mutation(self):
        arr = ArkoudaBaseArray(self.base_objs()["Categorical"])
        out = arr.to_ndarray()
        out[-1] = "X"
        fresh = arr.to_ndarray()
        assert fresh[-1] == "ccc"

    # ------------------------------ Sanity / smoke ------------------------------

    def test_to_ndarray_length_matches(self):
        for key, backend in self.base_objs().items():
            arr = ArkoudaBaseArray(backend)
            assert len(arr.to_ndarray()) == len(arr)

    def test_to_ndarray_largeish_numeric_smoke(self):
        a = ArkoudaBaseArray(ak.arange(0, 10_000))
        out = a.to_ndarray()
        assert out[0] == 0
        assert out[-1] == 9_999
        assert out.dtype == np.dtype("int64")

    # --------------------------- Negative cases / validation ------------------------------------------

    def test_concat_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            ArkoudaArray._concat_same_type([])

    def test_concat_mixed_subclass_types_raises_value_error(self):
        a = ArkoudaArray(ak.array([1, 2]))
        b = ArkoudaStringArray(ak.array(["x", "y"]))
        with pytest.raises(ValueError, match="instances of the same class"):
            ArkoudaArray._concat_same_type([a, b])

    def test_concat_dtype_mismatch_pdarray_raises_value_error(self):
        a = ArkoudaArray(ak.array([1, 2, 3]))  # int64
        b = ArkoudaArray(ak.array(np.array([1.0, 2.0])))  # float64
        with pytest.raises(ValueError, match="All pdarrays must have same dtype"):
            ArkoudaArray._concat_same_type([a, b])

    # ------------------------------- Fast path: single item -------------------------------------------

    def test_single_item_fast_path_identity(self):
        for arr in [
            ArkoudaArray(ak.array([1, 2, 3])),
            ArkoudaStringArray(ak.array(["a", "b"])),
            ArkoudaCategoricalArray(ak.Categorical(ak.array(["a", "b"]))),
        ]:
            out = type(arr)._concat_same_type([arr])
            # Should be the same object (no new allocation)
            assert out._data is arr._data

    # -------------------------------------- Happy paths -----------------------------------------------

    def test_concat_pdarray_happy_path_contents_and_length(self):
        a = ArkoudaArray(ak.array([1, 2, 3]))
        b = ArkoudaArray(ak.array([4, 5]))
        c = ArkoudaArray(ak.array([], dtype="int64"))

        out = ArkoudaArray._concat_same_type([a, b, c])
        assert isinstance(out, ArkoudaArray)
        # Length should add up
        assert len(out) == len(a) + len(b) + len(c)
        # Contents should match numpy concatenation

        assert_equal(out._data, ak.array([1, 2, 3, 4, 5], dtype="int64"))

    def test_concat_strings_happy_path_contents_and_length(self):
        a = ArkoudaStringArray(ak.array(["a", "bb"]))
        b = ArkoudaStringArray(ak.array(["ccc"]))
        out = ArkoudaStringArray._concat_same_type([a, b])
        assert isinstance(out, ArkoudaStringArray)
        assert len(out) == len(a) + len(b)
        assert_equal(out._data, ak.array(["a", "bb", "ccc"]))

    def test_concat_categorical_happy_path_contents_and_length(self):
        s1 = ak.array(["r", "g", "b"])
        s2 = ak.array(["g", "r"])
        a = ArkoudaCategoricalArray(ak.Categorical(s1))
        b = ArkoudaCategoricalArray(ak.Categorical(s2))

        out = ArkoudaCategoricalArray._concat_same_type([a, b])
        assert isinstance(out, ArkoudaCategoricalArray)
        assert len(out) == len(a) + len(b)
        # Compare as numpy arrays of labels
        assert_equal(out._data, ak.Categorical(ak.array(["r", "g", "b", "g", "r"])))

    # ------------------------------ Mixed content sanity checks ---------------------------------------

    def test_concat_with_zero_length_segments(self):
        a = ArkoudaArray(ak.array([], dtype="int64"))
        b = ArkoudaArray(ak.array([10, 20]))
        c = ArkoudaArray(ak.array([], dtype="int64"))
        out = ArkoudaArray._concat_same_type([a, b, c])
        assert len(out) == 2
        assert_equal(out._data, ak.array([10, 20], dtype="int64"))

    def test_concat_largeish_segments_length_only_smoke(self):
        # Keep it lightweight but non-trivial
        from arkouda.pandas.extension._arkouda_array import ArkoudaArray

        a = ArkoudaArray(ak.arange(0, 1000))
        b = ArkoudaArray(ak.arange(1000, 1500))
        out = ArkoudaArray._concat_same_type([a, b])
        assert len(out) == 1500
        # Spot check a few values to avoid moving too much data to the client

        assert out[0] == 0
        assert out[999] == 999
        assert out[1000] == 1000
        assert out[-1] == 1499

    @pytest.mark.parametrize("sort", [False, True])
    @pytest.mark.parametrize("use_na_sentinel", [True, False])
    def test_factorize_int_basic(self, sort, use_na_sentinel):
        """
        Int array has no NAs; first-appearance order vs sorted uniques;
        NA sentinel only affects behavior if there are NAs (there aren't here).
        """
        a = ArkoudaArray(ak.array([1, 2, 1, 3]))
        codes, uniques = a.factorize(sort=sort, use_na_sentinel=use_na_sentinel)

        if not sort:
            # First appearance: uniques [1, 2, 3]
            assert_array_equal(uniques, np.array([1, 2, 3]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))
        else:
            # Sorted: uniques [1, 2, 3] (same here, but codes recomputed from sorted order)
            assert_array_equal(uniques, np.array([1, 2, 3]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))

    @pytest.mark.parametrize("sort", [False, True])
    def test_factorize_float_with_nan_default_sentinel(self, sort):
        """
        Float array treats NaN as missing -> -1 sentinel by default.
        """
        a = ArkoudaArray(ak.array([1.0, np.nan, 1.0, 2.0]))
        codes, uniques = a.factorize(sort=sort)

        if not sort:
            # First appearance uniques: [1.0, 2.0]
            assert_array_equal(uniques, np.array([1.0, 2.0]))
            assert_array_equal(codes, np.array([0, -1, 0, 1]))
        else:
            # Sorted uniques: [1.0, 2.0] (same set)
            assert_array_equal(uniques, np.array([1.0, 2.0]))
            assert_array_equal(codes, np.array([0, -1, 0, 1]))

    def test_factorize_float_with_nan_no_sentinel(self):
        """
        With use_na_sentinel=False, NaNs get a valid code == len(uniques).
        """
        a = ArkoudaArray(ak.array([1.0, np.nan, 1.0, 2.0]))
        codes, uniques = a.factorize(sort=False, use_na_sentinel=False)
        # uniques from first appearance: [1.0, 2.0]; NaN code == 2
        assert_array_equal(uniques, np.array([1.0, 2.0]))
        assert_array_equal(codes, np.array([0, 2, 0, 1]))

    def test_factorize_float_all_nan(self):
        """
        Edge case: all values are NaN -> codes all sentinel, uniques empty.
        """
        a = ArkoudaArray(ak.array([np.nan, np.nan]))
        codes, uniques = a.factorize()
        assert_array_equal(uniques, np.array([], dtype=float))
        assert_array_equal(codes, np.array([-1, -1], dtype=np.int64))

    @pytest.mark.parametrize("sort", [False, True])
    def test_factorize_strings_basic(self, sort):
        """
        Strings: no NA handling; empty strings are treated as normal values.
        """
        s = ak.array(["a", "b", "a", "c"])
        a = ArkoudaStringArray(s)
        codes, uniques = a.factorize(sort=sort)

        if not sort:
            assert_array_equal(uniques, np.array(["a", "b", "c"]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))
        else:
            # Sorted: ["a", "b", "c"] -> same result for this set
            assert_array_equal(uniques, np.array(["a", "b", "c"]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))

    def test_factorize_strings_with_empty_string(self):
        """
        Explicitly ensure "" is treated as a normal value (not missing).
        """
        s = ak.array(["", "x", "", "y"])
        a = ArkoudaStringArray(s)
        codes, uniques = a.factorize(sort=False)
        assert_array_equal(uniques, np.array(["", "x", "y"]))
        assert_array_equal(codes, np.array([0, 1, 0, 2]))

    @pytest.mark.parametrize("sort", [False, True])
    def test_factorize_categorical_basic(self, sort):
        """
        Categorical: factorization operates over observed values (not categories table),
        honoring first-appearance vs sorted order semantics of the observed data.
        """
        s = ak.array(["red", "blue", "red", "green"])
        cat = ak.Categorical(s)  # construct from Strings
        a = ArkoudaCategoricalArray(cat)
        codes, uniques = a.factorize(sort=sort)

        if not sort:
            # first appearance uniques: ["red", "blue", "green"]
            assert_array_equal(uniques, np.array(["red", "blue", "green"]))
            assert_array_equal(codes, np.array([0, 1, 0, 2]))
        else:
            # sorted uniques: ["blue", "green", "red"]
            assert_array_equal(uniques, np.array(["blue", "green", "red"]))
            # remapped codes according to sorted order:
            # red->2, blue->0, green->1
            assert_array_equal(codes, np.array([2, 0, 2, 1]))

    def test_factorize_stability_first_appearance_vs_sorted(self):
        """
        Sanity check that switching sort changes code assignments consistently.
        """
        x = ak.array([2, 1, 3, 2])
        a = ArkoudaArray(x)

        codes_unsorted, uniques_unsorted = a.factorize(sort=False)
        codes_sorted, uniques_sorted = a.factorize(sort=True)

        # First appearance uniques: [2, 1, 3]
        assert_array_equal(uniques_unsorted, np.array([2, 1, 3]))
        assert_array_equal(codes_unsorted, np.array([0, 1, 2, 0]))

        # Sorted uniques: [1, 2, 3]
        assert_array_equal(uniques_sorted, np.array([1, 2, 3]))
        # mapping old->new: 2->1, 1->0, 3->2  => [1,0,2,1]
        assert_array_equal(codes_sorted, np.array([1, 0, 2, 1]))
