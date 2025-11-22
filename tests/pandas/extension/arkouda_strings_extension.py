import pandas as pd
import pytest

import arkouda as ak

from arkouda.pandas.extension import ArkoudaStringArray
from arkouda.testing import assert_equivalent


class TestArkoudaStringsExtension:
    def test_strings_extension_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _arkouda_string_array

        result = doctest.testmod(
            _arkouda_string_array, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.fixture
    def str_arr(self):
        data = ak.array(["a", "b", "c", "d", "e"])
        return ArkoudaStringArray(data)

    def test_take_strings_no_allow_fill(self, str_arr):
        out = str_arr.take([0, 2, 4], allow_fill=False)
        assert isinstance(out, ArkoudaStringArray)
        assert out.to_numpy().tolist() == ["a", "c", "e"]

    # TODO: Implement this test after Issue #4878 is resolved
    # def test_take_strings_negative_indices_no_allow_fill(self,str_arr):
    #     # negative indexes wrap when allow_fill=False
    #     out = str_arr.take([-1, -3, 0], allow_fill=False)
    #     assert out.to_numpy().tolist() == ["e", "c", "a"]

    def test_take_strings_allow_fill_explicit_fill(self, str_arr):
        # -1 becomes fill_value when allow_fill=True
        out = str_arr.take([0, -1, 2, -1], allow_fill=True, fill_value="MISSING")
        assert out.to_numpy().tolist() == ["a", "MISSING", "c", "MISSING"]

    def test_take_strings_allow_fill_default_fill_none(self, str_arr, monkeypatch):
        """
        When fill_value=None and allow_fill=True, expect the string default.
        Most implementations use "" (empty string) for string dtype sentinel fill.
        """
        # If your EA uses a class attr for default, ensure it's "" for strings,
        # or your take() detects dtype "str" and chooses "".
        indexer = [1, -1, 3]
        out = str_arr.take(indexer, allow_fill=True, fill_value=None)
        assert out.to_numpy().tolist() == ["b", "", "d"]

    def test_take_strings_allow_fill_invalid_negative_raises(self, str_arr):
        with pytest.raises(ValueError):
            str_arr.take([0, -2, 1], allow_fill=True, fill_value="X")

    def test_take_strings_preserves_dtype_and_length(self, str_arr):
        idx = [4, 3, 2, 1, 0]
        out = str_arr.take(idx, allow_fill=False)
        assert isinstance(out, ArkoudaStringArray)
        assert len(out) == len(idx)
        # kind should be 'U' (unicode) for numpy strings in most builds
        assert out.to_numpy().dtype.kind in ("U", "S", "O")

    def test_take_strings_sparse_fill_only_masks(self, str_arr):
        indexer = [0, 1, -1, 2, -1]
        out = str_arr.take(indexer, allow_fill=True, fill_value="FILL")
        assert out.to_numpy().tolist() == ["a", "b", "FILL", "c", "FILL"]

    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_take_strings_scaling(self, prob_size):
        strings = ak.array(["a", "b"])
        pda = strings[ak.tile(ak.array([0, 1]), prob_size // 2)]
        arr = ArkoudaStringArray(pda)
        s = pd.Series(pda.to_ndarray())
        idx1 = ak.arange(prob_size, dtype=ak.int64) // 2
        assert_equivalent(arr.take(idx1)._data, s.take(idx1.to_ndarray()).to_numpy())
