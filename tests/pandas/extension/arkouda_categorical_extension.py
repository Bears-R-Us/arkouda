import pandas as pd
import pytest

import arkouda as ak
from arkouda.pandas.extension import ArkoudaCategoricalArray
from arkouda.testing import assert_equivalent


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
        """
        Fill with a known category value ('b').
        """
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
