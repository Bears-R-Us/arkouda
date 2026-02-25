import numpy as np
import pandas as pd
import pytest

import arkouda as ak

from arkouda import Series
from arkouda.numpy.strings import Strings
from arkouda.pandas.extension import ArkoudaExtensionArray
from arkouda.pandas.extension._index_accessor import ArkoudaIndexAccessor
from arkouda.pandas.extension._series_accessor import (
    ArkoudaSeriesAccessor,
    _ak_array_to_pandas_series,
    _pandas_series_to_ak_array,
)
from arkouda.pandas.index import MultiIndex
from arkouda.pandas.series import Series as ak_Series
from tests.apply_test import supports_apply


def _assert_series_equal_values(s: pd.Series, values):
    """Helper: assert pandas Series values == iterable `values`."""
    assert list(s.tolist()) == list(values)


class TestArkoudaSeriesAccessor:
    @pytest.mark.requires_chapel_module("In1dMsg")
    def test_series_accessor_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _series_accessor

        result = doctest.testmod(
            _series_accessor, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_to_ak_simple_series_roundtrip(self):
        s = pd.Series([1, 2, 3], name="nums")

        # Initially not Arkouda-backed
        assert not s.ak.is_arkouda

        # Convert to Arkouda-backed pandas.Series
        s_ak = s.ak.to_ak()
        assert isinstance(s_ak, pd.Series)
        assert s_ak.name == "nums"
        assert s_ak.ak.is_arkouda

        # Underlying array is ArkoudaExtensionArray
        arr = s_ak.array
        assert isinstance(arr, ArkoudaExtensionArray)
        akcol = arr._data
        assert isinstance(akcol, ak.pdarray)
        _assert_series_equal_values(s_ak, akcol.tolist())

        # Collect back to plain NumPy-backed Series
        s_local = s_ak.ak.collect()
        assert isinstance(s_local, pd.Series)
        assert not isinstance(s_local.array, ArkoudaExtensionArray)
        assert not s_local.ak.is_arkouda
        assert s_local.name == "nums"
        assert s_local.equals(s)

    def test_collect_on_plain_series_is_noop_semantics(self):
        """collect() on a non-Arkouda Series should behave like a simple copy."""
        s = pd.Series([10, 20, 30], name="vals")
        assert not s.ak.is_arkouda

        s2 = s.ak.collect()
        assert isinstance(s2, pd.Series)
        assert not isinstance(s2.array, ArkoudaExtensionArray)
        assert not s2.ak.is_arkouda
        assert s2.equals(s)
        assert s2.name == "vals"

    def test_pandas_series_to_ak_legacy_and_back(self):
        s = pd.Series(["a", "b", "c"], name="letters")

        # pandas.Series -> legacy Arkouda array
        ak_arr = _pandas_series_to_ak_array(s)
        # For non-Arkouda-backed string input, we should get an Arkouda Strings
        assert isinstance(ak_arr, Strings)
        assert ak_arr.tolist() == ["a", "b", "c"]

        # legacy Arkouda array -> Arkouda-backed pandas.Series (EA-backed)
        s_back = _ak_array_to_pandas_series(ak_arr, name="letters")
        assert isinstance(s_back, pd.Series)
        assert isinstance(s_back.array, ArkoudaExtensionArray)
        assert s_back.name == "letters"
        _assert_series_equal_values(s_back, ["a", "b", "c"])

    def test_to_ak_legacy_simple_series_and_accessor_roundtrip(self):
        s = pd.Series([4, 5, 6], name="foo")

        # pandas.Series -> legacy Arkouda Series
        ak_s = s.ak.to_ak_legacy()
        assert isinstance(ak_s, Series)
        assert ak_s.size == len(s)
        # Assume Arkouda pandas Series exposes list-like values
        assert ak_s.tolist() == [4, 5, 6]

        # pandas.Series -> legacy Arkouda array (underlying column)
        ak_arr = _pandas_series_to_ak_array(s)

        # ArkoudaSeriesAccessor.from_ak_legacy should just wrap the legacy array via EA
        s_ea = ArkoudaSeriesAccessor.from_ak_legacy(ak_arr, name="foo")
        assert isinstance(s_ea, pd.Series)
        assert isinstance(s_ea.array, ArkoudaExtensionArray)
        assert s_ea.name == "foo"
        _assert_series_equal_values(s_ea, [4, 5, 6])

    def test_is_arkouda_flag_series(self):
        s = pd.Series([1, 2, 3], name="nums")

        # Initially false
        assert not s.ak.is_arkouda

        s_ak = s.ak.to_ak()
        assert s_ak.ak.is_arkouda

        s_local = s_ak.ak.collect()
        assert not s_local.ak.is_arkouda

    def test_is_arkouda_true_for_to_ak_series_with_default_rangeindex(self):
        s = pd.Series([1, 2, 3])  # default RangeIndex
        ak_s = s.ak.to_ak()

        assert isinstance(getattr(ak_s, "array", None), ArkoudaExtensionArray)
        assert ak_s.ak.is_arkouda is True

    def test_is_arkouda_true_for_to_ak_series_with_nontrivial_index(self):
        s = pd.Series([10, 20, 30], index=pd.Index([100, 200, 300], name="i"))
        ak_s = s.ak.to_ak()

        assert ak_s.ak.is_arkouda is True

    def test_is_arkouda_true_for_to_ak_series_with_multiindex(self):
        mi = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["l0", "l1"])
        s = pd.Series([0, 1, 2, 3], index=mi)

        ak_s = s.ak.to_ak()
        assert ak_s.ak.is_arkouda is True

    def test_to_ak_preserves_index_and_name(self):
        idx = pd.Index([100, 200, 300], name="id")
        s = pd.Series([10, 20, 30], index=idx, name="values")

        s_ak = s.ak.to_ak()
        assert s_ak.name == "values"
        assert s_ak.index.name == "id"
        assert s_ak.ak.is_arkouda

        # Collect should preserve index and name too
        s_local = s_ak.ak.collect()
        assert s_local.name == "values"
        assert s_local.index.name == "id"
        assert s_local.equals(s)

    def test_to_ak_on_already_arkouda_series_idempotent_semantics(self):
        """Calling to_ak() on an already Arkouda-backed Series should keep it Arkouda-backed."""
        s = pd.Series([1, 2, 3], name="nums")
        s_ak = s.ak.to_ak()
        assert s_ak.ak.is_arkouda

        s_ak2 = s_ak.ak.to_ak()
        assert s_ak2.ak.is_arkouda
        assert isinstance(s_ak2.array, ArkoudaExtensionArray)
        _assert_series_equal_values(s_ak2, [1, 2, 3])
        assert s_ak2.name == "nums"
        assert s_ak2.index.equals(s_ak.index)

    def test_to_ak_idempotent_identity_and_zero_copy(self):
        """
        Reproducer for review comment:

            s = self._obj
            arr = s.array
            if isinstance(arr, ArkoudaExtensionArray):
                return s
            s_ak1 = s.ak.to_ak()
            s_ak2 = s_ak1.ak.to_ak()
            assert s_ak1 is s_ak2
        """
        s = pd.Series([100, 200, 300], name="values")

        s_ak1 = s.ak.to_ak()
        assert isinstance(s_ak1.array, ArkoudaExtensionArray)
        assert s_ak1.ak.is_arkouda

        # Capture underlying Arkouda column object for zero-copy assertion
        col1 = getattr(s_ak1.array, "_data", None)
        assert col1 is not None, "Expected ArkoudaExtensionArray to expose _data"

        # Idempotence: calling to_ak again returns the exact same Series object
        s_ak2 = s_ak1.ak.to_ak()
        assert s_ak2 is s_ak1

        # Zero-copy: underlying column should be the same object too
        col2 = getattr(s_ak2.array, "_data", None)
        assert col2 is col1

    def test_to_ak_legacy_preserves_named_index(self):
        """
        Reproducer for review comment / snippet:

            idx = pd.Index([10, 20, 30], name="id")
            s = pd.Series([100, 200, 300], index=idx, name="values")
            ak_s = s.ak.to_ak_legacy()
            assert ak_s.index ... preserves name + values
        """
        idx = pd.Index([10, 20, 30], name="id")
        s = pd.Series([100, 200, 300], index=idx, name="values")

        ak_s = s.ak.to_ak_legacy()
        assert isinstance(ak_s, ak_Series)

        # Values preserved
        assert ak_s.tolist() == [100, 200, 300]

        # Index preserved (values + name)
        # (Assumes Arkouda Series index is an Arkouda Index-like object with .to_list/.tolist and .name)
        assert ak_s.index.tolist() == [10, 20, 30]
        assert getattr(ak_s.index, "name", None) == "id"
        assert ak_s.name == "values"

        # If legacy Series has a pandas roundtrip API, verify it too, but don't require it.
        if hasattr(ak_s, "to_pandas"):
            s_back = ak_s.to_pandas()
            assert isinstance(s_back, pd.Series)
            assert s_back.index.tolist() == [10, 20, 30]
            assert s_back.index.name == "id"
            assert s_back.tolist() == [100, 200, 300]

    def test_ak_array_to_pandas_series_default_index_is_arkouda(self):
        # legacy arkouda array (pdarray / Strings / Categorical all go through the same helper)
        akarr = ak.array([1, 3, 4, 1])

        from arkouda.pandas.extension._series_accessor import _ak_array_to_pandas_series

        s = _ak_array_to_pandas_series(akarr, name="x")

        # Should not silently create a NumPy RangeIndex
        assert not isinstance(s.index, pd.RangeIndex)

        # Index should be Arkouda-backed
        assert ArkoudaIndexAccessor(s.index).is_arkouda is True

        # Values should match 0..n-1
        assert np.array_equal(s.index.to_numpy(), np.arange(len(s)))

        # Data should be Arkouda-backed and correct
        assert np.array_equal(s.to_numpy(), np.array([1, 3, 4, 1]))

    def test_ak_array_to_pandas_series_preserves_provided_index_and_makes_it_arkouda(self):
        akarr = ak.array([10, 20, 30, 40])

        from arkouda.pandas.extension._series_accessor import _ak_array_to_pandas_series

        idx = pd.Index([100, 101, 102, 103], name="id")
        assert ArkoudaIndexAccessor(idx).is_arkouda is False

        s = _ak_array_to_pandas_series(akarr, name="x", index=idx)

        # Should preserve index values + name
        assert s.index.name == "id"
        assert np.array_equal(s.index.to_numpy(), np.array([100, 101, 102, 103]))

        # Index should be Arkouda-backed (either via internal conversion or via caller passing ak index)
        assert ArkoudaIndexAccessor(s.index).is_arkouda is True

        # Data correct
        assert np.array_equal(s.to_numpy(), np.array([10, 20, 30, 40]))

    def test_series_locate_multiindex_accepts_per_level_pdarray_keys(self):
        # Build MultiIndex Series
        lvl0 = ak.array([0, 0, 1, 1])
        lvl1 = ak.array([10, 11, 10, 11])
        mi = MultiIndex([lvl0, lvl1], names=["a", "b"])

        vals = ak.array([100, 101, 102, 103])
        s = Series(vals, index=mi)

        # Per-level keys (pdarrays) — should be supported
        k0 = ak.array([0, 1])
        k1 = ak.array([10, 11])

        # Expected: select (0,10) and (1,11) in that order -> [100, 103]
        out = s.locate((k0, k1))
        out = s.locate([k0, k1])

        # Compare values (and optionally index)
        assert out.tolist() == [100, 103]
        assert out.index.nlevels == 2
        assert out.index.names == ["a", "b"]


class TestArkoudaSeriesGroupby:
    def test_series_ak_groupby_raises_if_not_arkouda_backed(self):
        s = pd.Series([80, 443, 80], name="Destination Port")  # plain pandas Series
        with pytest.raises(
            TypeError, match=r"Series must be Arkouda-backed\. Call \.ak\.to_ak\(\) first\."
        ):
            _ = s.ak.groupby()

    def test_series_ak_groupby_returns_ak_groupby_and_size_matches_pandas_value_counts(self):
        s = pd.Series([80, 443, 80, 22, 443], name="Destination Port").ak.to_ak()

        g = s.ak.groupby()
        assert isinstance(g, ak.GroupBy)

        keys, counts = g.size()

        # Convert results to python lists for comparison
        keys_py = keys.tolist()
        counts_py = counts.tolist()

        # Series.groupby().size() is equivalent to Series.value_counts()
        # The grouped values become the index of the returned series.
        # We sort so the order matches.
        expected = pd.Series([80, 443, 80, 22, 443]).value_counts().sort_index()

        assert keys_py == expected.index.to_list()
        assert counts_py == expected.to_list()

    def test_series_ak_groupby_raises_if_underlying_array_missing_data(self):
        s = pd.Series([1, 1, 2, 3]).ak.to_ak()

        # Keep Series "arkouda-backed" but make _data unavailable
        # so we hit the second error branch.
        s.array._data = None  # type: ignore[attr-defined]

        with pytest.raises(TypeError, match=r"Arkouda-backed Series array does not expose '_data'"):
            _ = s.ak.groupby()

    # ------------------------------------------------------------------
    # locate
    # ------------------------------------------------------------------

    @pytest.mark.requires_chapel_module("In1dMsg")
    @pytest.mark.parametrize("dtype", ["int64", "uint64", "bool_", "bigint"])
    @pytest.mark.parametrize("dtype_index", ["ak_int64", "ak_uint64"])
    def test_locate(self, dtype, dtype_index):
        pda = pd.array(ak.arange(3, dtype=dtype), dtype="ak." + dtype)
        pda2 = pd.array(ak.array(["A", "B", "C"]), dtype="ak_str")
        idx = pd.array(ak.arange(3), dtype=dtype_index)
        for val in pda, pda2:
            s = pd.Series(val, index=idx).ak.to_ak()

            for key in (
                1,
                pd.Index([1], dtype=dtype_index),
                pd.Index([0, 2], dtype=dtype_index),
            ):
                lk = s.ak.locate(key)
                assert isinstance(lk, pd.Series)
                key = ak.array(key) if not isinstance(key, int) else key
                assert (lk.index == s.index[key]).all()
                assert (lk.values == s.values[key]).all()

            # testing multi-index lookup
            mi = pd.MultiIndex.from_arrays([pda, pda[::-1]])
            s = pd.Series(data=val, index=mi)
            lk = s.ak.locate(mi[0])
            assert isinstance(lk, pd.Series)
            assert lk.values[0] == val[0]

            # ensure error with scalar and multi-index
            with pytest.raises(TypeError):
                s.ak.locate(0)

            with pytest.raises(TypeError):
                s.ak.locate([0, 2])


class TestArkoudaSeriesAccessorArgsort:
    def test_argsort_returns_arkoudaarray_and_matches_numpy_int(self):
        s = pd.Series([3, 1, 2]).ak.to_ak()
        perm = s.ak.argsort()

        assert perm.ak.is_arkouda
        assert np.array_equal(perm.to_numpy(), np.array([1, 2, 0]))

    def test_argsort_descending(self):
        s = pd.Series([3, 1, 2]).ak.to_ak()
        perm = s.ak.argsort(ascending=False)

        assert perm.ak.is_arkouda
        assert np.array_equal(perm.to_numpy(), np.array([0, 2, 1]))

    def test_argsort_float_nan_default_last(self):
        s = pd.Series([3.0, np.nan, 1.0]).ak.to_ak()
        perm = s.ak.argsort()

        assert perm.ak.is_arkouda
        # values: [3.0, nan, 1.0] -> sorted non-nan indices [2,0] then nan [1]
        assert np.array_equal(perm.to_numpy(), np.array([2, 0, 1]))

    def test_argsort_float_nan_first(self):
        s = pd.Series([3.0, np.nan, 1.0]).ak.to_ak()
        perm = s.ak.argsort(na_position="first")

        assert perm.ak.is_arkouda
        # nan first, then sorted non-nan
        assert np.array_equal(perm.to_numpy(), np.array([1, 2, 0]))

    def test_argsort_invalid_na_position_raises(self):
        s = pd.Series([3.0, np.nan, 1.0]).ak.to_ak()
        with pytest.raises(ValueError, match="na_position must be 'first' or 'last'"):
            s.ak.argsort(na_position="middle")

    def test_argsort_non_arkouda_series_raises(self):
        s = pd.Series([3, 1, 2])
        with pytest.raises(TypeError, match="Arkouda-backed"):
            s.ak.argsort()

    def test_argsort_strings(self):
        s = pd.Series(["b", "a", "c"]).ak.to_ak()
        perm = s.ak.argsort()

        assert perm.ak.is_arkouda
        assert np.array_equal(perm.to_numpy(), np.array([1, 0, 2]))


@pytest.mark.requires_chapel_module("ApplyMsg")
class TestArkoudaSeriesApply:
    @classmethod
    def setup_class(cls):
        if not supports_apply():
            pytest.skip("apply not supported")

    def test_series_accessor_apply_requires_arkouda_backed(self):
        s = pd.Series([1, 2, 3], name="x")
        with pytest.raises(TypeError, match="Series must be Arkouda-backed"):
            s.ak.apply(lambda x: x + 1)

    def test_series_accessor_apply_callable_preserves_index_and_name(self):
        idx = pd.Index([10, 20, 30], name="id")
        s = pd.Series([1, 2, 3], index=idx, name="x").ak.to_ak()

        out = s.ak.apply(lambda v: v + 1)

        # still distributed
        assert out.ak.is_arkouda

        # preserves metadata
        assert out.name == "x"
        assert out.index.equals(s.index)
        assert out.index.name == "id"

        # correct values (materialize for assertion)
        expected = pd.Series([2, 3, 4], index=idx, name="x")
        got = out.ak.collect()
        assert got.equals(expected)

    def test_series_accessor_apply_callable_dtype_change(self):
        idx = pd.Index([0, 1, 2], name="i")
        s = pd.Series([2, 4, 6], index=idx, name="y").ak.to_ak()

        out = s.ak.apply(lambda v: v * 0.5, result_dtype="float64")

        assert out.ak.is_arkouda
        assert out.name == "y"
        assert out.index.equals(s.index)

        expected = pd.Series([1.0, 2.0, 3.0], index=idx, name="y")
        got = out.ak.collect()
        # use allclose-like semantics for float comparisons
        assert got.index.equals(expected.index)
        assert got.name == expected.name
        assert pytest.approx(got.to_numpy().tolist()) == expected.to_numpy().tolist()

    def test_series_accessor_apply_string_lambda_same_dtype(self):
        idx = pd.Index([5, 6, 7], name="row")
        s = pd.Series([1, 2, 3], index=idx, name="z").ak.to_ak()

        out = s.ak.apply("lambda x,: x+2")

        assert out.ak.is_arkouda
        assert out.name == "z"
        assert out.index.equals(s.index)

        expected = pd.Series([3, 4, 5], index=idx, name="z")
        got = out.ak.collect()
        assert got.equals(expected)

    def test_series_accessor_apply_string_lambda_rejects_result_dtype_change(self):
        s = pd.Series([1, 2, 3], name="a").ak.to_ak()

        # apply.py enforces: for string funcs, result_dtype must match input dtype
        with pytest.raises(TypeError, match="result_dtype must match"):
            _ = s.ak.apply("lambda x,: x+1", result_dtype="float64")

    def test_series_accessor_apply_rejects_strings(self):
        s = pd.Series(["a", "b", "c"]).ak.to_ak()
        with pytest.raises(TypeError, match="only supports numeric pdarray"):
            s.ak.apply(lambda x: x)

    def test_series_accessor_apply_rejects_categorical(self):
        ak_cat = ak.Categorical(ak.array(["red", "blue", "red"]))
        s = pd.Series.ak.from_ak_legacy(ak_cat, name="color")

        with pytest.raises(TypeError, match="only supports numeric pdarray"):
            s.ak.apply(lambda x: x)
