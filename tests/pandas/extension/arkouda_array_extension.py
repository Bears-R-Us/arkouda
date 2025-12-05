import numpy as np
import pandas as pd
import pytest

import arkouda as ak

from arkouda import numeric_and_bool_scalars
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.pandas.extension import ArkoudaArray, ArkoudaCategoricalArray, ArkoudaStringArray
from arkouda.pandas.extension._arkouda_array import ArkoudaArray
from arkouda.pandas.extension._dtypes import ArkoudaBoolDtype, ArkoudaFloat64Dtype, ArkoudaInt64Dtype
from arkouda.testing import assert_equivalent


SUPPORTED_TYPES = [ak.bool_, ak.uint64, ak.int64, ak.bigint, ak.float64]


class TestArkoudaArrayExtension:
    @pytest.fixture
    def base_arr(self):
        # Small, distinct values to make indexing obvious
        data = ak.array([10, 20, 30, 40, 50])
        return ArkoudaArray(data)

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

    def test_init_from_pdarray_reuses_underlying(self):
        base = ak.arange(5)
        arr = ArkoudaArray(base)
        assert isinstance(arr._data, pdarray)
        # Reuse: same object identity is the clearest signal
        assert arr._data is base
        # Round-trip equality
        assert np.array_equal(arr.to_ndarray(), np.arange(5))

    def test_init_from_pdarray_copy_when_requested(self):
        base = ak.arange(5)
        arr = ArkoudaArray(base, copy=True)
        assert isinstance(arr._data, pdarray)
        # Should not be the same object when copy=True
        assert arr._data is not base
        assert base.name != arr._data.name
        assert np.array_equal(arr.to_ndarray(), np.arange(5))

    @pytest.mark.parametrize(
        "payload, expected",
        [
            (np.array([1, 2, 3], dtype=np.int64), np.array([1, 2, 3])),
            (np.array([1.5, 2.5, 3.5], dtype=np.float64), np.array([1.5, 2.5, 3.5])),
            ([True, False, True], np.array([True, False, True])),
            ((10, 20, 30), np.array([10, 20, 30])),
        ],
    )
    def test_init_converts_numpy_and_python_sequences(self, payload, expected):
        arr = ArkoudaArray(payload)
        out = arr.to_ndarray()
        assert isinstance(arr._data, pdarray)
        assert out.dtype == expected.dtype
        assert np.array_equal(out, expected)

    def test_init_from_arkoudaarray_reuses_backing_pdarray(self):
        base = ak.arange(4)
        a1 = ArkoudaArray(base)
        a2 = ArkoudaArray(a1)  # should reuse pdarray, not wrap twice
        assert a2._data is a1._data
        assert np.array_equal(a2.to_ndarray(), np.arange(4))

    @pytest.mark.parametrize(
        "src,dtype_cls,values",
        [
            ([1, 2, 3], ArkoudaInt64Dtype, np.array([1, 2, 3], dtype=np.int64)),
            ([1.0, 2.0], ArkoudaFloat64Dtype, np.array([1.0, 2.0], dtype=np.float64)),
            ([1, 0, 1], ArkoudaBoolDtype, np.array([True, False, True], dtype=bool)),
        ],
    )
    def test_init_with_explicit_dtype_casts(self, src, dtype_cls, values):
        # dtype may be provided as NumPy dtype strings or objects
        # We test a representative set: int64, float64, bool
        if dtype_cls is ArkoudaInt64Dtype:
            dtype = np.int64
        elif dtype_cls is ArkoudaFloat64Dtype:
            dtype = np.float64
        else:
            dtype = bool

        arr = ArkoudaArray(src, dtype=dtype)
        assert isinstance(arr.dtype, dtype_cls)
        assert np.array_equal(arr.to_ndarray(), values)

    def test_init_rejects_2d_input(self):
        two_d = np.array([[1, 2], [3, 4]], dtype=np.int64)
        with pytest.raises(ValueError):
            ArkoudaArray(two_d)

    def test_init_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            ArkoudaArray({"a": 1, "b": 2})

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
        from arkouda.pandas.extension._dtypes import _ArkoudaBaseDtype

        assert isinstance(arr.dtype, _ArkoudaBaseDtype)
        assert arr.dtype.name == "int64"

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

    @pytest.mark.parametrize(
        "indexer_factory",
        [
            lambda: [0, 2, 4],  # Python list
            lambda: np.array([0, 2, 4], dtype=np.int64),  # NumPy int64
            lambda: np.array([0, 2, 4], dtype=np.int32),  # NumPy int32
            lambda: ak.array(np.array([0, 2, 4], dtype=np.int64)),  # Arkouda pdarray
        ],
    )
    def test_take_no_allow_fill_various_indexers(self, base_arr, indexer_factory):
        indexer = indexer_factory()
        out = base_arr.take(indexer, allow_fill=False)
        assert isinstance(out, ArkoudaArray)
        assert out.to_numpy().tolist() == [10, 30, 50]

    #   TODO: Implement this test after Issue #4878 is resolved
    # def test_take_negative_indices_no_allow_fill(self,base_arr):
    #     # Negative indices are valid when allow_fill=False (wrap from end)
    #     indexer = [-1, -2, 0]  # 50, 30, 10
    #     out = base_arr.take(indexer, allow_fill=False)
    #     assert out.to_numpy().tolist() == [50, 30, 10]

    def test_take_allow_fill_replaces_minus_one_with_fill_value(self, base_arr):
        indexer = [0, -1, 2, -1]
        out = base_arr.take(indexer, allow_fill=True, fill_value=999)
        # indices: 0 -> 10, -1 -> fill, 2 -> 30, -1 -> fill
        assert out.to_numpy().tolist() == [10, 999, 30, 999]

    def test_take_allow_fill_uses_default_fill_when_none(self, base_arr, monkeypatch):
        # Ensure class default is used when fill_value=None
        # If you've changed default_fill_value, update expectation accordingly.
        monkeypatch.setattr(ArkoudaArray, "default_fill_value", -1, raising=False)

        indexer = [1, -1, 3]
        out = base_arr.take(indexer, allow_fill=True, fill_value=None)
        assert out.to_numpy().tolist() == [20, -1, 40]

    def test_take_allow_fill_invalid_negative_raises(self, base_arr):
        # Pandas semantics: when allow_fill=True, only -1 is allowed as sentinel
        indexer = [0, -2, 1]
        with pytest.raises(ValueError):
            base_arr.take(indexer, allow_fill=True, fill_value=0)

    def test_take_preserves_dtype_and_length(self, base_arr):
        indexer = [4, 3, 2, 1, 0]
        out = base_arr.take(indexer, allow_fill=False)
        assert isinstance(out, ArkoudaArray)
        assert len(out) == len(indexer)
        # round-trip dtype through numpy for a quick check
        assert out._data.dtype == base_arr._data.dtype

    def test_take_with_pdarray_indexer(self, base_arr):
        idx_pd = ak.array(np.array([0, 4, 1], dtype=np.int64))
        assert isinstance(idx_pd, pdarray)
        out = base_arr.take(idx_pd, allow_fill=False)
        assert out.to_numpy().tolist() == [10, 50, 20]

    def test_take_allow_fill_large_mask_sparse_fill(self, base_arr):
        # Mixed valid and -1 entries; checks that only masked positions are filled
        indexer = [0, 1, -1, 2, -1]
        out = base_arr.take(indexer, allow_fill=True, fill_value=777)
        assert out.to_numpy().tolist() == [10, 20, 777, 30, 777]

    @pytest.mark.parametrize("fill_value", [0, 123, -5])
    def test_take_allow_fill_explicit_fill_castable(self, base_arr, fill_value):
        indexer = [-1, 0, -1]
        out = base_arr.take(indexer, allow_fill=True, fill_value=fill_value)
        assert out.to_numpy().tolist() == [fill_value, 10, fill_value]

    def test_take_allow_fill_all_filled(self, base_arr):
        indexer = [-1, -1, -1]
        out = base_arr.take(indexer, allow_fill=True, fill_value=42)
        assert out.to_numpy().tolist() == [42, 42, 42]

    @pytest.mark.parametrize("dtype", SUPPORTED_TYPES)
    @pytest.mark.parametrize("prob_size", pytest.prob_size)
    def test_take_numeric_scaling(self, dtype, prob_size):
        pda = ak.arange(prob_size, dtype=dtype)
        arr = ArkoudaArray(pda)
        s = pd.Series(pda.to_ndarray())
        idx1 = ak.arange(prob_size, dtype=ak.int64) // 2
        assert_equivalent(arr.take(idx1)._data, s.take(idx1.to_ndarray()).to_numpy())
