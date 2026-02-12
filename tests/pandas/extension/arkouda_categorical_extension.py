import numpy as np
import pandas as pd
import pytest

import arkouda as ak

from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.pandas.categorical import Categorical
from arkouda.pandas.extension import (
    ArkoudaArray,
    ArkoudaCategoricalArray,
    ArkoudaCategoricalDtype,
    ArkoudaExtensionArray,
    ArkoudaStringArray,
)
from arkouda.testing import assert_equal, assert_equivalent


class TestArkoudaCategoricalExtension:
    def test_base_categorical_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _arkouda_categorical_array

        result = doctest.testmod(
            _arkouda_categorical_array, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.fixture
    def cat_arr(self):
        """
        Construct a small categorical EA with categories ['a','b','c'] and values:
        ['a','b','a','c','b']
        """
        values = ["a", "b", "a", "c", "b"]
        # Prefer a `_from_sequence` constructor if your EA supports it.
        # Otherwise, adapt to your categorical EA's builder API (e.g., from_codes).
        return ArkoudaCategoricalArray._from_sequence(values)

    def test_init_from_categorical_reuses_underlying(self):
        base = Categorical(ak.array(["a", "b", "a"]))
        arr = ArkoudaCategoricalArray(base)
        assert arr._data is base
        assert np.array_equal(arr.to_ndarray(), np.array(["a", "b", "a"], dtype=object))

    @pytest.mark.parametrize(
        "payload, expected",
        [
            (np.array(["x", "y", "x"]), np.array(["x", "y", "x"], dtype=object)),
            (["apple", "banana", "apple"], np.array(["apple", "banana", "apple"], dtype=object)),
            (("cat", "dog", "cat"), np.array(["cat", "dog", "cat"], dtype=object)),
        ],
    )
    def test_init_converts_numpy_and_python_sequences(self, payload, expected):
        arr = ArkoudaCategoricalArray(payload)
        out = arr.to_ndarray()
        assert np.array_equal(out, expected)

    def test_init_from_arkouda_categorical_array_reuses_backing_data(self):
        base = Categorical(ak.array(["r", "g"]))
        a1 = ArkoudaCategoricalArray(base)
        a2 = ArkoudaCategoricalArray(a1)
        assert a2._data is a1._data
        assert np.array_equal(a2.to_ndarray(), np.array(["r", "g"], dtype=object))

    def test_dtype_property_is_arkouda_categorical_dtype(self):
        c = Categorical(ak.array(["hi", "bye"]))
        arr = ArkoudaCategoricalArray(c)
        assert isinstance(arr.dtype, ArkoudaCategoricalDtype)

    def test_init_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            ArkoudaCategoricalArray({"bad": "type"})

    def test_take_categorical_no_allow_fill(self, cat_arr):
        out = cat_arr.take([0, 2, 4], allow_fill=False)
        # Expect original values at those positions
        np_out = out.to_numpy().tolist()
        assert np_out == ["a", "a", "b"]

    # TODO: Implement this test after Issue #4878 is resolved
    # def test_take_categorical_negative_indices_no_allow_fill(self, cat_arr):
    #     out = cat_arr.take([-1, -3, 0], allow_fill=False)
    #     np_out = np.asarray(out.to_numpy()).tolist()
    #     assert np_out == ["b", "a", "a"]

    #   TODO:  Include this test when Issue #4881 is resolved
    # def test_take_categorical_allow_fill_default_is_na(self, cat_arr):
    #     """
    #     When allow_fill=True and fill_value=None, categorical EAs should fill with NA.
    #     We'll assert that isna() reflects the masked positions.
    #     """
    #     out = cat_arr.take([0, -1, 2, -1], allow_fill=True, fill_value=None)
    #     # Positions 1 and 3 should be NA
    #     isna = out.isna()
    #     # Try Arkouda boolean -> numpy for assertion
    #     try:
    #         isna_np = isna.to_ndarray()
    #     except Exception:
    #         isna_np = np.asarray(isna)
    #     assert isna_np.tolist() == [False, True, False, True]

    def test_take_categorical_allow_fill_with_existing_category(self, cat_arr):
        """Fill with a known category value ('b')."""
        out = cat_arr.take([0, -1, 3, -1], allow_fill=True, fill_value="b")
        np_out = out.to_numpy().tolist()
        assert np_out == ["a", "b", "c", "b"]

    def test_take_categorical_allow_fill_invalid_negative_raises(self, cat_arr):
        with pytest.raises(ValueError):
            cat_arr.take([0, -2, 1], allow_fill=True, fill_value="a")

    def test_take_categorical_preserves_dtype_and_categories(self, cat_arr):
        """
        Ensure categories are preserved after take (common categorical EA invariant).
        We try to check categories either on `dtype.categories` or on the instance.
        """
        from arkouda import Categorical

        # Snapshot categories before
        cats_before = None
        if hasattr(cat_arr, "categories"):
            cats_before = cat_arr.categories.tolist()

        out = cat_arr.take([4, 3, 2, 1, 0], allow_fill=False)

        # Same categories after (order preserved)
        cats_after = None
        if hasattr(out, "categories"):
            cats_after = out.categories.tolist()

        assert cats_before == cats_after
        assert len(out.to_numpy()) == 5
        assert isinstance(out, ArkoudaCategoricalArray)
        assert isinstance(out._data, Categorical)

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_take_categorical_scaling(self, prob_size):
        strings = ak.array(["a", "b"])
        pda = ak.Categorical(strings[ak.tile(ak.array([0, 1]), prob_size // 2)])
        arr = ArkoudaCategoricalArray(pda)
        s = pd.Series(pda.to_ndarray())
        idx1 = ak.arange(prob_size, dtype=ak.int64) // 2
        assert_equivalent(arr.take(idx1)._data.to_strings(), s.take(idx1.to_ndarray()).to_numpy())


class TestArkoudaCategoricalArrayAsType:
    def test_categorical_array_astype_category_stays_extension(
        self,
    ):
        c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["x", "y", "x"])))
        out = c.astype("category")
        assert isinstance(out, ArkoudaCategoricalArray)
        assert_equal(out._data, c._data)

    def test_categorical_array_astype_object_returns_numpy_labels(
        self,
    ):
        c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["x", "y", "x"])))
        out = c.astype(object)
        assert isinstance(out, np.ndarray)
        assert out.dtype == object
        assert out.tolist() == ["x", "y", "x"]

    @pytest.mark.parametrize("dtype", ["string", "str", "str_"])
    def test_categorical_array_astype_string_targets_return_string_array(self, dtype):
        c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["x", "y", "x"])))
        out = c.astype(dtype)
        assert isinstance(out, ArkoudaStringArray)
        assert out.to_ndarray().tolist() == ["x", "y", "x"]

    def test_categorical_array_astype_other_returns_extension_array_not_numpy(self):
        # New behavior: does NOT fall back to NumPy; returns an Arkouda-backed EA
        c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["1", "2", "3"])))
        out = c.astype("int64")

        assert isinstance(out, ArkoudaExtensionArray)
        assert not isinstance(out, np.ndarray)

        # Values should match numeric cast of labels
        np.testing.assert_array_equal(out.to_ndarray(), np.array([1, 2, 3], dtype=np.int64))

    def test_categorical_array_astype_other_uses_labels_once(self):
        # (Optional sanity) ensure it is casting labels, not codes/categories
        c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["10", "20", "10"])))
        out = c.astype("int64")
        np.testing.assert_array_equal(out.to_ndarray(), np.array([10, 20, 10], dtype=np.int64))

    def test_categorical_array_astype_extensiondtype_categoricaldtype_copy_false_returns_self(self):
        c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["x", "y", "x"])))
        out = c.astype(pd.CategoricalDtype(), copy=False)
        assert out is c

    def test_categorical_array_astype_extensiondtype_categoricaldtype_copy_true_returns_new_array(self):
        c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["x", "y", "x"])))
        out = c.astype(pd.CategoricalDtype(), copy=True)

        assert isinstance(out, ArkoudaCategoricalArray)
        assert out is not c
        assert out.to_ndarray().tolist() == ["x", "y", "x"]

    def test_categorical_array_astype_extensiondtype_stringdtype_returns_string_array(self):
        c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["x", "y", "x"])))
        out = c.astype(pd.StringDtype())  # ExtensionDtype path

        assert isinstance(out, ArkoudaStringArray)
        assert out.to_ndarray().tolist() == ["x", "y", "x"]

    def test_categorical_array_astype_extensiondtype_numeric_casts_labels_and_returns_extension_array(
        self,
    ):
        c = ArkoudaCategoricalArray(ak.Categorical(ak.array(["1", "2", "3"])))
        out = c.astype(pd.Int64Dtype())  # ExtensionDtype path

        assert isinstance(out, ArkoudaExtensionArray)
        assert not isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out.to_ndarray(), np.array([1, 2, 3], dtype=np.int64))


class TestArkoudaCategoricalArrayEq:
    def _make(self, values):
        """Helper to construct an ArkoudaCategoricalArray from Python/NumPy values."""
        cats = ak.Categorical(ak_array(values))
        return ArkoudaCategoricalArray(cats)

    def test_eq_categorical_same_length_all_equal(self):
        left = self._make(["a", "b", "c"])
        right = self._make(["a", "b", "c"])

        result = left == right

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 3
        assert result._data.dtype == "bool"
        assert result._data.all()

    def test_eq_categorical_same_length_some_unequal(self):
        # ["a", "b", "c", "d", "e"]
        left = self._make(["a", "b", "c", "d", "e"])
        # ["a", "x", "c", "y", "e"] -> True, False, True, False, True
        right = self._make(["a", "x", "c", "y", "e"])

        result = left == right

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 5
        assert result._data.dtype == "bool"

        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(result._data.to_ndarray(), expected)
        assert result._data.sum() == 3

    def test_eq_categorical_length_mismatch_raises(self):
        left = self._make(["a", "b", "c"])
        right = self._make(["a", "b", "c", "d"])

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = left == right

    def test_eq_scalar_broadcast_label(self):
        arr = self._make(["foo", "bar", "foo", "baz"])

        result = arr == "foo"

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 4
        assert result._data.dtype == "bool"

        expected = np.array([True, False, True, False])
        np.testing.assert_array_equal(result._data.to_ndarray(), expected)
        assert result._data.sum() == 2

    def test_eq_with_numpy_array(self):
        arr = self._make(["a", "b", "c"])
        other = np.array(["a", "x", "c"], dtype=object)

        result = arr == other

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 3
        assert result._data.dtype == "bool"

        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result._data.to_ndarray(), expected)
        assert result._data.sum() == 2

    def test_eq_with_numpy_array_length_mismatch_raises(self):
        arr = self._make(["a", "b", "c"])
        other = np.array(["a", "b"], dtype=object)

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = arr == other

    def test_eq_with_python_sequence(self):
        arr = self._make(["a", "b", "c", "d"])
        other = ["a", "x", "c", "y"]

        result = arr == other

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 4
        assert result._data.dtype == "bool"

        expected = np.array([True, False, True, False])
        np.testing.assert_array_equal(result._data.to_ndarray(), expected)
        assert result._data.sum() == 2

    def test_eq_with_python_sequence_length_mismatch_raises(self):
        arr = self._make(["a", "b", "c"])
        other = ["a", "b"]

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = arr == other

    def test_eq_with_unsupported_type_returns_all_false(self):
        arr = self._make(["a", "b", "c"])

        result = arr == {"not": "comparable"}

        assert result is False

    def test_eq_with_python_sequence_len1_broadcasts_categorical(self):
        arr = ArkoudaCategoricalArray(Categorical(ak.array(["a", "b", "c", "d"])))
        result = arr == ["c"]
        assert result._data.sum() == 1  # only index 2

    def test_eq_with_numpy_array_len1_broadcasts_categorical(self):
        arr = ArkoudaCategoricalArray(Categorical(ak.array(["a", "b", "c", "d"])))
        result = arr == np.array(["c"], dtype=object)
        assert result._data.sum() == 1

    def test_eq_with_python_sequence_length_mismatch_raises_categorical(self):
        arr = ArkoudaCategoricalArray(Categorical(ak.array(["a", "b", "c"])))
        with pytest.raises(ValueError, match="Lengths must match"):
            _ = arr == ["a", "b"]  # len 2, not 1 and not len(arr)


class TestArkoudaCategoricalArrayGetitem:
    def _make_array(self):
        data = ak.Categorical(ak.array(["a", "b", "c", "d"]))
        return ArkoudaCategoricalArray(data)

    # --- scalar indexing -------------------------------------------------

    def test_getitem_scalar_returns_python_scalar(self):
        arr = self._make_array()

        result = arr[1]

        assert isinstance(result, str)
        assert result == "b"

    # --- list / sequence indexers ----------------------------------------

    def test_getitem_python_list_of_int_returns_categorical_array(self):
        arr = self._make_array()
        idx = [1, 3]

        result = arr[idx]

        assert isinstance(result, ArkoudaCategoricalArray)
        # values
        np.testing.assert_array_equal(result.to_ndarray(), np.array(["b", "d"]))

    def test_getitem_empty_list_returns_empty_categorical_array(self):
        arr = self._make_array()
        idx: list[int] = []

        result = arr[idx]

        assert isinstance(result, ArkoudaCategoricalArray)
        assert len(result) == 0

    # --- numpy / arkouda indexers ----------------------------------------

    def test_getitem_numpy_int_array_returns_categorical_array(self):
        arr = self._make_array()
        idx = np.array([0, 2], dtype=np.int64)

        result = arr[idx]

        assert isinstance(result, ArkoudaCategoricalArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array(["a", "c"]))

    def test_getitem_numpy_bool_mask_returns_categorical_array(self):
        arr = self._make_array()
        mask = np.array([True, False, True, False], dtype=bool)

        result = arr[mask]

        assert isinstance(result, ArkoudaCategoricalArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array(["a", "c"]))

    def test_getitem_arkouda_pdarray_int_indexer_returns_categorical_array(self):
        arr = self._make_array()
        idx = ak.array([0, 3])

        result = arr[idx]

        assert isinstance(result, ArkoudaCategoricalArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array(["a", "d"]))

    # --- slices -----------------------------------------------------------

    def test_getitem_slice_returns_arkouda_categorical_array(self):
        arr = self._make_array()

        result = arr[1:3]

        assert isinstance(result, ArkoudaCategoricalArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array(["b", "c"]))

    def test_getitem_full_slice_returns_shallow_copy_categorical_array(self):
        arr = self._make_array()

        result = arr[:]

        assert isinstance(result, ArkoudaCategoricalArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array(["a", "b", "c", "d"]))
        # not required, but nice to assert we didn't mutate original
        np.testing.assert_array_equal(arr.to_ndarray(), np.array(["a", "b", "c", "d"]))

    # --- error behavior ---------------------------------------------------

    def test_getitem_raises_index_error_on_out_of_bounds_scalar(self):
        arr = self._make_array()
        with pytest.raises(IndexError):
            _ = arr[10]


class TestArkoudaCategoricalValueCounts:
    def _series_to_pycounts(self, s: pd.Series) -> dict:
        """
        Convert the returned Series to a plain Python {value: count} mapping.

        Works whether the index / values are Arkouda-backed or NumPy-backed.
        """
        idx = list(s.index.to_numpy())
        vals = list(s.to_numpy())
        return {idx[i]: int(vals[i]) for i in range(len(s))}

    def test_categorical_value_counts_basic(self):
        a = ArkoudaCategoricalArray(["a", "b", "a", "c", "b", "a"])
        out = a.value_counts()

        got = self._series_to_pycounts(out)
        assert got == {"a": 3, "b": 2, "c": 1}

    def test_categorical_value_counts_single_category(self):
        a = ArkoudaCategoricalArray(["x", "x", "x"])
        out = a.value_counts()

        got = self._series_to_pycounts(out)
        assert got == {"x": 3}

    def test_categorical_value_counts_empty(self):
        a = ArkoudaCategoricalArray(ak.array([], dtype="str_"))
        out = a.value_counts()

        assert isinstance(out, pd.Series)
        assert len(out) == 0

    def test_categorical_value_counts_matches_pandas_as_multiset(self):
        """Cross-check correctness against pandas value_counts, ignoring ordering."""
        data = ["blue", "red", "blue", "green", "blue", "red"]
        a = ArkoudaCategoricalArray(data)
        out = a.value_counts()

        got = self._series_to_pycounts(out)
        expected = pd.Series(pd.Categorical(data)).value_counts(dropna=True).to_dict()

        # pandas returns counts as numpy ints; normalize to python ints
        assert got == {str(k): int(v) for k, v in expected.items()}

    def test_categorical_value_counts_dropna_true_drops_na_value(self):
        """
        With the current implementation, dropna=True filters the result down to
        categories != cat.na_value.
        """
        a = ArkoudaCategoricalArray(["a", "b", "a"])
        out = a.value_counts(dropna=True)

        got = self._series_to_pycounts(out)

        # It should not contain the na value
        na = a._data.na_value
        assert na not in set(got.keys())

    def test_categorical_value_counts_dropna_false_includes_non_na_categories(self):
        """dropna=False should not apply the na_value filter, so normal categories appear."""
        a = ArkoudaCategoricalArray(["a", "b", "a"])
        out = a.value_counts(dropna=False)

        got = self._series_to_pycounts(out)

        assert got.get("a", 0) == 2
        assert got.get("b", 0) == 1

    def test_categorical_value_counts_dropna(self):
        c = Categorical([])
        a = ArkoudaCategoricalArray(["x", "y", "x", c.na_value])
        na = a._data.na_value

        out1 = a.value_counts(dropna=True)
        assert na not in set(out1.index)

        out2 = a.value_counts(dropna=False)
        assert na in set(out2.index)
