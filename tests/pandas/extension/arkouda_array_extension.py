import numpy as np
import pytest

import arkouda as ak
from arkouda import numeric_and_bool_scalars
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.pandas.extension._arkouda_array import ArkoudaArray, ArkoudaDtype


class TestArkoudaArrayExtension:
    def test_array_extension_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _arkouda_array

        result = doctest.testmod(
            _arkouda_array, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_constructor_from_pdarray(self):
        arr = ArkoudaArray(ak.arange(5))
        assert isinstance(arr, ArkoudaArray)
        assert len(arr) == 5

    def test_constructor_from_numpy(self):
        arr = ArkoudaArray(np.array([10, 20, 30]))
        assert isinstance(arr, ArkoudaArray)
        assert len(arr) == 3

    def test_getitem_scalar(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        assert arr[1] == 1

    def test_getitem_slice(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        sub = arr[2:5]
        assert isinstance(sub, ArkoudaArray)
        assert sub.to_numpy().tolist() == [2, 3, 4]

    def test_setitem_scalar(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data[:])  # avoid modifying fixture
        arr[1] = 42
        assert arr[1] == 42

    def test_setitem_array(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data[:])
        # arr[[0, 2]] = [99, 88]
        arr[ak.array([0, 2])] = [99, 88]
        assert arr[0] == 99
        assert arr[2] == 88

    def test_len(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        assert len(arr) == 10

    def test_isna(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        na = arr.isna()
        assert ak.all(na == False)  # noqa: E712

    def test_isna_with_nan(self):
        from arkouda.testing import assert_equal

        ak_data = ak.array([1, np.nan, 2])
        arr = ArkoudaArray(ak_data)
        na = arr.isna()
        expected = ak.array([False, True, False])
        assert_equal(na, expected)

    def test_copy(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        cpy = arr.copy()
        assert arr.equals(cpy)
        assert arr is not cpy

    def test_dtype(self):
        arr = ArkoudaArray(ak.array([1]))
        assert isinstance(arr.dtype, ArkoudaDtype)
        assert arr.dtype.name == "arkouda"

    def test_nbytes(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        assert isinstance(arr.nbytes, int)
        assert arr.nbytes == 80

    def test_to_numpy(self):
        ak_data = ak.arange(5)
        arr = ArkoudaArray(ak_data)
        np_arr = arr.to_numpy()
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.tolist() == [0, 1, 2, 3, 4]

    def test_astype(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        casted = arr.astype(np.float64)
        assert isinstance(casted, ArkoudaArray)
        assert isinstance(casted._data, pdarray)
        assert casted._data.dtype == np.float64

    def test_equals_true(self):
        ak_data = ak.arange(10)
        arr1 = ArkoudaArray(ak_data)
        arr2 = ArkoudaArray(ak_data[:])
        assert arr1.equals(arr2)

    def test_equals_false(self):
        ak_data = ak.arange(10)
        arr1 = ArkoudaArray(ak_data)
        arr2 = ArkoudaArray(ak.array([5, 4, 3, 2, 1]))
        assert not arr1.equals(arr2)

    def test_argsort(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        perm = arr.argsort()
        sorted_vals = arr._data[perm]
        assert ak.is_sorted(sorted_vals)

    @pytest.mark.parametrize("reduction", ["all", "any", "sum", "prod", "min", "max"])
    def test_reduce_ops(self, reduction):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        result = arr._reduce(reduction)
        assert isinstance(result, numeric_and_bool_scalars)

    def test_reduce_invalid(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        with pytest.raises(TypeError):
            arr._reduce("mean")

    def test_concat_same_type(self):
        a1 = ArkoudaArray(ak.array([1, 2]))
        a2 = ArkoudaArray(ak.array([3, 4]))
        out = ArkoudaArray._concat_same_type([a1, a2])
        assert isinstance(out, ArkoudaArray)
        assert out.to_numpy().tolist() == [1, 2, 3, 4]

    def test_eq_operator(self):
        a1 = ArkoudaArray(ak.array([1, 2]))
        a2 = ArkoudaArray(ak.array([1, 2]))
        result = a1 == a2
        assert ak.all(result)

    def test_repr(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        assert "ArkoudaArray" in repr(arr)

    def test_factorize(self):
        arr = ArkoudaArray(ak.array([1, 2, 1, 3]))
        codes, uniques = arr.factorize()
        assert set(codes.tolist()) == {0, 1, 2}
        assert sorted(uniques.tolist()) == [1, 2, 3]

    def test_from_factorized(self):
        values = [10, 20, 10]
        orig = ArkoudaArray(ak.array([10, 20, 10]))
        new_arr = ArkoudaArray._from_factorized(values, orig)
        assert isinstance(new_arr, ArkoudaArray)
        assert new_arr.to_numpy().tolist() == values
