import numpy as np
import pandas as pd
import pytest

import arkouda as ak

from arkouda.index import Index as ak_Index
from arkouda.index import MultiIndex as ak_MultiIndex
from arkouda.pandas.extension import ArkoudaExtensionArray
from arkouda.pandas.extension._index_accessor import (
    ArkoudaIndexAccessor,
    _ak_index_to_pandas_no_copy,
    _pandas_index_to_ak,
)


def _assert_index_equal_values(p_idx: pd.Index, values):
    """Helper: assert pandas Index values == iterable `values`."""
    assert list(p_idx.tolist()) == list(values)


class TestArkoudaIndexAccessor:
    def test_index_accessor_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _index_accessor

        result = doctest.testmod(
            _index_accessor, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_to_ak_simple_index_roundtrip(self):
        idx = pd.Index([1, 2, 3], name="id")

        # Initially not Arkouda-backed
        assert not idx.ak.is_arkouda

        # Convert to Arkouda-backed pandas.Index
        idx_ak = idx.ak.to_ak()
        assert isinstance(idx_ak, pd.Index)
        assert idx_ak.name == "id"
        assert idx_ak.ak.is_arkouda

        # Underlying array is ArkoudaExtensionArray
        arr = idx_ak.array
        assert isinstance(arr, ArkoudaExtensionArray)
        akcol = arr._data
        assert isinstance(akcol, ak.pdarray)
        _assert_index_equal_values(idx_ak, akcol.tolist())

        # Collect back to plain NumPy-backed Index
        idx_local = idx_ak.ak.collect()
        assert isinstance(idx_local, pd.Index)
        assert not isinstance(idx_local.array, ArkoudaExtensionArray)
        assert not idx_local.ak.is_arkouda
        assert idx_local.name == "id"
        assert idx_local.equals(idx)

    def test_collect_on_plain_index_is_noop_semantics(self):
        """collect() on a non-Arkouda Index should behave like a simple copy."""
        idx = pd.Index([10, 20, 30], name="nums")
        assert not idx.ak.is_arkouda

        idx2 = idx.ak.collect()
        assert isinstance(idx2, pd.Index)
        assert not isinstance(idx2.array, ArkoudaExtensionArray)
        assert not idx2.ak.is_arkouda
        assert idx2.equals(idx)
        assert idx2.name == "nums"

    @pytest.mark.parametrize("arkouda_backed", [False, True])
    def test_collect_multiindex_preserves_all_rows(self, arkouda_backed):
        # Ground-truth MultiIndex (plain pandas, with duplicates and ordering)
        arrays = [[1, 1, 2], ["red", "blue", "red"]]
        orig_midx = pd.MultiIndex.from_arrays(arrays, names=["num", "color"])

        # Optionally convert to Arkouda-backed representation
        if arkouda_backed:
            midx = orig_midx.ak.to_ak()
            assert midx.ak.is_arkouda
        else:
            midx = orig_midx
            assert not midx.ak.is_arkouda

        # Operation under test
        collected = midx.ak.collect()

        # Always get a plain pandas MultiIndex back
        assert isinstance(collected, pd.MultiIndex)

        # Length and names must match the original
        assert len(collected) == len(orig_midx)
        assert list(collected.names) == list(orig_midx.names)

        # Logical row tuples must be preserved exactly
        assert list(collected.to_list()) == list(orig_midx.to_list())

        # For the plain-pandas case, pandas' own `.equals` should also hold
        if not arkouda_backed:
            assert collected.equals(orig_midx)

    def test_to_ak_legacy_simple_index_and_helper_roundtrip(self):
        idx = pd.Index([4, 5, 6], name="foo")

        # pandas -> ak.Index
        ak_idx = idx.ak.to_ak_legacy()
        assert isinstance(ak_idx, ak_Index)
        assert ak_idx.name == "foo"
        assert ak_idx.size == len(idx)
        assert ak_idx.values.tolist() == [4, 5, 6]

        # ak.Index -> pandas via helper (zero-copy-ish EA)
        idx_ea = _ak_index_to_pandas_no_copy(ak_idx)
        assert isinstance(idx_ea, pd.Index)
        assert isinstance(idx_ea.array, ArkoudaExtensionArray)
        assert idx_ea.name == "foo"
        _assert_index_equal_values(idx_ea, [4, 5, 6])

    def test_pandas_index_to_ak_and_back(self):
        idx = pd.Index(["a", "b", "c"], name="letters")

        # pandas.Index -> ak.Index
        ak_idx = _pandas_index_to_ak(idx)
        assert isinstance(ak_idx, ak_Index)
        assert ak_idx.name == "letters"
        assert ak_idx.size == 3
        assert ak_idx.values.tolist() == ["a", "b", "c"]

        # ak.Index -> pandas.Index (EA-backed)
        idx_back = _ak_index_to_pandas_no_copy(ak_idx)
        assert isinstance(idx_back, pd.Index)
        assert isinstance(idx_back.array, ArkoudaExtensionArray)
        assert idx_back.name == "letters"
        _assert_index_equal_values(idx_back, ["a", "b", "c"])

    def test_to_ak_multiindex_roundtrip(self):
        arrays = [[1, 2, 3], ["a", "b", "c"]]
        midx = pd.MultiIndex.from_arrays(arrays, names=["num", "char"])

        assert not midx.ak.is_arkouda

        # pandas.MultiIndex -> Arkouda-backed MultiIndex (pandas object)
        midx_ak = midx.ak.to_ak()
        assert isinstance(midx_ak, pd.MultiIndex)
        assert list(midx_ak.names) == ["num", "char"]
        assert midx_ak.ak.is_arkouda

        # Each level should be ArkoudaExtensionArray-backed
        for lvl in midx_ak.levels:
            assert isinstance(lvl.array, ArkoudaExtensionArray)

        # Collect back to plain NumPy-backed MultiIndex
        midx_local = midx_ak.ak.collect()
        assert isinstance(midx_local, pd.MultiIndex)
        assert not midx_local.ak.is_arkouda
        assert midx_local.equals(midx)

        # Levels are no longer ArkoudaExtensionArray-backed
        for lvl in midx_local.levels:
            assert not isinstance(lvl.array, ArkoudaExtensionArray)

    def test_to_ak_legacy_multiindex_and_helper_roundtrip(self):
        arrays = [[1, 1, 2], ["red", "blue", "red"]]
        midx = pd.MultiIndex.from_arrays(arrays, names=["num", "color"])

        # pandas.MultiIndex -> ak.MultiIndex
        ak_midx = midx.ak.to_ak_legacy()
        assert isinstance(ak_midx, ak_MultiIndex)
        assert ak_midx.nlevels == 2
        assert list(ak_midx.names) == ["num", "color"]
        assert ak_midx.size == len(midx)

        # ak.MultiIndex -> pandas.MultiIndex with EA-backed levels
        midx_ea = _ak_index_to_pandas_no_copy(ak_midx)
        assert isinstance(midx_ea, pd.MultiIndex)
        assert list(midx_ea.names) == ["num", "color"]
        for lvl in midx_ea.levels:
            assert isinstance(lvl.array, ArkoudaExtensionArray)

        # Values should match original (up to pandas' internal coding)
        assert list(midx_ea) == list(midx)

    def test_is_arkouda_flag_index_and_multiindex(self):
        idx = pd.Index([1, 2, 3])
        midx = pd.MultiIndex.from_arrays([[1, 2], ["a", "b"]])

        # Initially false
        assert not idx.ak.is_arkouda
        assert not midx.ak.is_arkouda

        idx_ak = idx.ak.to_ak()
        midx_ak = midx.ak.to_ak()

        assert idx_ak.ak.is_arkouda
        assert midx_ak.ak.is_arkouda

        idx_local = idx_ak.ak.collect()
        midx_local = midx_ak.ak.collect()

        assert not idx_local.ak.is_arkouda
        assert not midx_local.ak.is_arkouda

    def test_from_ak_legacy_roundtrip_index(self):
        """Round-trip using to_ak_legacy() + from_ak_legacy()."""
        idx = pd.Index([7, 8, 9], name="nums")

        ak_idx = idx.ak.to_ak_legacy()
        assert isinstance(ak_idx, ak_Index)

        # NOTE: from_ak_legacy should *not* call akidx.to_ak();
        # it should just wrap via _ak_index_to_pandas_no_copy.
        idx_back = ak.pandas.extension.ArkoudaIndexAccessor.from_ak_legacy(ak_idx)
        assert isinstance(idx_back, pd.Index)
        assert isinstance(idx_back.array, ArkoudaExtensionArray)
        assert idx_back.name == "nums"
        _assert_index_equal_values(idx_back, [7, 8, 9])

    def test_akcol_to_series_default_index_is_arkouda(self):
        # Arkouda column (server-side)
        akcol = ak.array([1, 3, 4, 1])

        # This calls your updated helper
        from arkouda.pandas.extension._dataframe_accessor import _akcol_to_series

        s = _akcol_to_series("x", akcol)

        # Index should NOT be pandas RangeIndex (NumPy-backed)
        assert not isinstance(s.index, pd.RangeIndex)

        # Index should be Arkouda-backed
        assert ArkoudaIndexAccessor(s.index).is_arkouda is True

        # And values should match 0..n-1
        assert np.array_equal(s.index.to_numpy(), np.arange(len(s)))

    def test_akcol_to_series_converts_provided_index_to_arkouda(self):
        akcol = ak.array([10, 20, 30, 40])

        from arkouda.pandas.extension._dataframe_accessor import _akcol_to_series

        # Provide a normal NumPy-backed pandas Index
        idx = pd.Index([100, 101, 102, 103], name="id")
        assert ArkoudaIndexAccessor(idx).is_arkouda is False

        s = _akcol_to_series("x", akcol, index=idx)

        # Should now be Arkouda-backed, but preserve values & name
        assert ArkoudaIndexAccessor(s.index).is_arkouda is True
        assert s.index.name == "id"
        assert np.array_equal(s.index.to_numpy(), np.array([100, 101, 102, 103]))
