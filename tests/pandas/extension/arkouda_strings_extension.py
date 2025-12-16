import numpy as np
import pandas as pd
import pytest

import arkouda as ak

from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.pandas.extension import ArkoudaArray, ArkoudaStringArray, ArkoudaStringDtype
from arkouda.testing import assert_equivalent


@pytest.mark.requires_chapel_module("EncodingMsg")
class TestArkoudaStringsExtension:
    @pytest.fixture
    def str_arr(self):
        data = ak.array(["a", "b", "c", "d", "e"])
        return ArkoudaStringArray(data)

    def test_strings_extension_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _arkouda_string_array

        result = doctest.testmod(
            _arkouda_string_array, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_init_from_strings_uses_directly(self):
        s = ak.array(["a", "b", "c"])
        arr = ArkoudaStringArray(s)
        # Underlying should be exactly the same object
        assert arr._data is s
        assert np.array_equal(arr.to_ndarray(), np.array(["a", "b", "c"], dtype=object))

    @pytest.mark.parametrize(
        "payload, expected",
        [
            (np.array(["x", "y", "z"]), np.array(["x", "y", "z"], dtype=object)),
            (["cat", "dog"], np.array(["cat", "dog"], dtype=object)),
            (("red", "green", "blue"), np.array(["red", "green", "blue"], dtype=object)),
        ],
    )
    def test_init_converts_numpy_and_python_sequences(self, payload, expected):
        arr = ArkoudaStringArray(payload)
        out = arr.to_ndarray()
        assert np.array_equal(out, expected)

    def test_init_from_arkouda_string_array_reuses_backing_data(self):
        s = ak.array(["aa", "bb"])
        a1 = ArkoudaStringArray(s)
        a2 = ArkoudaStringArray(a1)
        # Should reuse the exact same Strings instance
        assert a2._data is a1._data
        assert np.array_equal(a2.to_ndarray(), np.array(["aa", "bb"], dtype=object))

    def test_dtype_property_is_arkouda_string_dtype(self):
        s = ak.array(["one"])
        arr = ArkoudaStringArray(s)
        assert isinstance(arr.dtype, ArkoudaStringDtype)

    def test_init_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            ArkoudaStringArray({"not": "valid"})  # dicts are not supported

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


class TestArkoudaStringArrayEq:
    def _make(self, values):
        """Helper to construct an ArkoudaStringArray from Python/NumPy values."""
        # ak_array will give a Strings object, which ArkoudaStringArray accepts
        return ArkoudaStringArray(ak_array(values))

    def test_eq_string_array_same_length_all_equal(self):
        left = self._make(["a", "b", "c"])
        right = self._make(["a", "b", "c"])

        result = left == right

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 3
        assert result._data.dtype == "bool"
        assert result._data.all()

    def test_eq_string_array_same_length_some_unequal(self):
        # [ "a", "b", "c", "d", "e" ]
        left = self._make(["a", "b", "c", "d", "e"])
        # [ "a", "x", "c", "y", "e" ] -> True, False, True, False, True
        right = self._make(["a", "x", "c", "y", "e"])

        result = left == right

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 5
        assert result._data.dtype == "bool"

        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(result._data.to_ndarray(), expected)
        assert result._data.sum() == 3

    def test_eq_string_array_length_mismatch_raises(self):
        left = self._make(["a", "b", "c"])
        right = self._make(["a", "b", "c", "d"])

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = left == right

    def test_eq_scalar_broadcast_string(self):
        arr = self._make(["foo", "bar", "foo", "baz"])

        result = arr == "foo"

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 4
        assert result._data.dtype == "bool"

        # positions 0 and 2 are "foo"
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

    def test_eq_with_python_sequence_len1_broadcasts_strings(self):
        arr = ArkoudaStringArray(ak.array(["a", "b", "c", "d"]))
        result = arr == ["c"]
        assert result._data.sum() == 1  # only index 2

    def test_eq_with_numpy_array_len1_broadcasts_strings(self):
        arr = ArkoudaStringArray(ak.array(["a", "b", "c", "d"]))
        result = arr == np.array(["c"], dtype=object)
        assert result._data.sum() == 1

    def test_eq_with_python_sequence_length_mismatch_raises_strings(self):
        arr = ArkoudaStringArray(ak.array(["a", "b", "c"]))
        with pytest.raises(ValueError, match="Lengths must match"):
            _ = arr == ["a", "b"]  # len 2, not 1 and not len(arr)
