import numpy as np
import pandas as pd
import pytest

import arkouda as ak

from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.pandas.extension import ArkoudaCategoricalArray, ArkoudaStringArray
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

    def test_copy_shallow_creates_new_wrapper_but_shares_data(self, ea):
        """
        deep=False should:
          * return a new ExtensionArray wrapper,
          * but share the same underlying Arkouda object in _data.
        """
        shallow = ea.copy(deep=False)

        # New wrapper instance, same concrete subclass
        assert shallow is not ea
        assert type(shallow) is type(ea)

        # Same underlying Arkouda object (no server-side copy)
        assert shallow._data is ea._data

        # Values are equal
        np.testing.assert_array_equal(shallow.to_numpy(), ea.to_numpy())

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

    def test_len(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        assert len(arr) == 10

    def test_isna(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data)
        na = arr.isna()
        assert np.all(na == False)

    def test_isna_with_nan(self):
        from arkouda.testing import assert_equal

        ak_data = ak.array([1, np.nan, 2])
        arr = ArkoudaArray(ak_data)
        na = arr.isna()
        expected = np.array([False, True, False])
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
        assert ak.all(result._data)

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

    @pytest.fixture(params=["numeric", "strings", "categorical"])
    def ea(self, request):
        """
        Parametrized fixture that yields one instance of each Arkouda-backed EA:

        - "numeric"      -> ArkoudaArray
        - "strings"      -> ArkoudaStringsArray
        - "categorical"  -> ArkoudaCategoricalArray
        """
        kind = request.param

        if kind == "numeric":
            data = ak.arange(5)
            arr = ArkoudaArray(data)

        elif kind == "strings":
            data = ak.array(["a", "b", "c", "a", "b"])
            arr = ArkoudaStringArray(data)

        elif kind == "categorical":
            base = ak.array(["a", "b", "c", "a", "b"])
            cat = ak.Categorical(base)
            arr = ArkoudaCategoricalArray(cat)

        else:  # pragma: no cover - defensive
            raise ValueError(f"Unexpected kind: {kind}")

        # Attach kind so tests can use it as an id if needed
        arr._test_kind = kind
        return arr

    def test_copy_deep_creates_independent_underlying_data(self, ea):
        """
        deep=True should:
          * return a new ExtensionArray wrapper,
          * with a different underlying Arkouda object in _data,
          * but with identical values.
        """
        deep = ea.copy(deep=True)

        # New wrapper instance, same concrete subclass
        assert deep is not ea
        assert type(deep) is type(ea)

        # Different underlying Arkouda object (server-side copy)
        assert deep._data is not ea._data

        # Values preserved
        np.testing.assert_array_equal(deep.to_numpy(), ea.to_numpy())

    def test_copy_default_behaves_like_deep_true(self, ea):
        """
        The default copy() call (no explicit deep argument) should behave like
        deep=True: a deep copy of the backing data.
        """
        default_copy = ea.copy()

        # New wrapper instance, same concrete subclass
        assert default_copy is not ea
        assert type(default_copy) is type(ea)

        # Different underlying Arkouda object
        assert default_copy._data is not ea._data

        # Values preserved
        np.testing.assert_array_equal(default_copy.to_numpy(), ea.to_numpy())


class TestArkoudaArrayEq:
    def test_eq_arkouda_array_same_length_all_equal(self):
        left = ArkoudaArray(ak.arange(5))
        right = ArkoudaArray(ak.arange(5))

        result = left == right

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 5
        assert result._data.dtype == "bool"
        assert result._data.all()

    def test_eq_arkouda_array_same_length_some_unequal(self):
        left = ArkoudaArray(ak.arange(5))
        right = ArkoudaArray(ak.arange(5) + 1)

        result = left == right

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 5
        assert result._data.dtype == "bool"
        assert result._data.sum() == 0

    def test_eq_arkouda_array_length_mismatch_raises(self):
        left = ArkoudaArray(ak.arange(3))
        right = ArkoudaArray(ak.arange(4))

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = left == right

    def test_eq_scalar_broadcast_int(self):
        data = ArkoudaArray(ak.arange(5))  # [0, 1, 2, 3, 4]
        result = data == 2

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 5
        assert result._data.dtype == "bool"
        assert result._data.sum() == 1  # only index 2

    def test_eq_scalar_broadcast_bool(self):
        data = ArkoudaArray(ak_array([True, False, True], dtype=bool))
        result = data == True

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 3
        assert result._data.dtype == "bool"
        assert result._data.sum() == 2

    def test_eq_with_pdarray_same_length(self):
        base = ak.arange(4)
        arr = ArkoudaArray(base)

        other = ak_array([0, 10, 2, 30])
        result = arr == other

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 4
        assert result._data.dtype == "bool"
        # indices 0 and 2 are equal
        assert result._data.sum() == 2

    def test_eq_with_pdarray_length_mismatch_raises(self):
        base = ak.arange(3)
        arr = ArkoudaArray(base)

        other = ak_array([0, 1])  # length 2

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = arr == other

    def test_eq_with_numpy_array(self):
        arr = ArkoudaArray(ak.arange(3))
        other = np.array([0, 99, 2], dtype=np.int64)

        result = arr == other

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 3
        assert result._data.dtype == "bool"
        # indices 0 and 2 are equal
        assert result._data.sum() == 2

    def test_eq_with_numpy_array_length_mismatch_raises(self):
        arr = ArkoudaArray(ak.arange(3))
        other = np.array([0, 1], dtype=np.int64)

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = arr == other

    def test_eq_with_python_sequence(self):
        arr = ArkoudaArray(ak.arange(4))
        other = [0, 10, 2, 30]

        result = arr == other

        assert isinstance(result, ArkoudaArray)
        assert result._data.size == 4
        assert result._data.dtype == "bool"
        assert result._data.sum() == 2  # indices 0 and 2

    def test_eq_with_unsupported_type_returns_all_false(self):
        arr = ArkoudaArray(ak.arange(5))

        result = arr == {"not": "comparable"}
        assert result is False

    def test_eq_with_python_sequence_len1_broadcasts(self):
        arr = ArkoudaArray(ak.arange(4))  # [0,1,2,3]
        result = arr == [2]
        assert result._data.sum() == 1  # only index 2

    def test_eq_with_numpy_array_len1_broadcasts(self):
        arr = ArkoudaArray(ak.arange(4))
        result = arr == np.array([2], dtype=np.int64)
        assert result._data.sum() == 1

    def test_eq_with_python_sequence_length_mismatch_raises(self):
        arr = ArkoudaArray(ak.arange(3))
        with pytest.raises(ValueError, match="Lengths must match"):
            _ = arr == [0, 1]  # len 2, not 1 and not len(arr)


class TestArkoudaArrayAllAny:
    def test_all_true_for_all_true_bool_array(self):
        arr = ArkoudaArray(ak.array([True, True, True]))

        result = arr.all()

        assert isinstance(result, bool)
        assert result is True

    def test_all_false_for_array_with_false(self):
        arr = ArkoudaArray(ak.array([True, False, True]))

        result = arr.all()

        assert isinstance(result, bool)
        assert result is False

    def test_any_true_for_array_with_true(self):
        arr = ArkoudaArray(ak.array([False, True, False]))

        result = arr.any()

        assert isinstance(result, bool)
        assert result is True

    def test_any_false_for_all_false_bool_array(self):
        arr = ArkoudaArray(ak.array([False, False, False]))

        result = arr.any()

        assert isinstance(result, bool)
        assert result is False

    def test_all_and_any_on_singleton_true(self):
        arr = ArkoudaArray(ak.array([True]))

        assert arr.all() is True
        assert arr.any() is True

    def test_all_and_any_on_singleton_false(self):
        arr = ArkoudaArray(ak.array([False]))

        assert arr.all() is False
        assert arr.any() is False


class TestArkoudaArrayOr:
    # -----------------------------
    # Helpers
    # -----------------------------
    def make_bool(self, vals):
        return ArkoudaArray(ak_array(vals, dtype=bool))

    # -----------------------------
    # ArkoudaArray | ArkoudaArray
    # -----------------------------
    def test_or_two_bool_arrays(self):
        a = self.make_bool([True, False, True])
        b = self.make_bool([False, False, True])

        result = a | b

        assert isinstance(result, ArkoudaArray)
        np.testing.assert_array_equal(result._data.to_ndarray(), np.array([True, False, True]))

    def test_or_length_mismatch_raises(self):
        a = self.make_bool([True, False])
        b = self.make_bool([True, False, True])

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = a | b

    # -----------------------------
    # ArkoudaArray | pdarray
    # -----------------------------
    def test_or_with_pdarray_same_length(self):
        a = self.make_bool([True, False, False])
        b = ak_array([False, True, False], dtype=bool)

        result = a | b

        assert isinstance(result, ArkoudaArray)
        np.testing.assert_array_equal(result._data.to_ndarray(), np.array([True, True, False]))

    def test_or_with_pdarray_length_mismatch_raises(self):
        a = self.make_bool([True, False])  # length 2
        b = ak_array([True, False, True], dtype=bool)  # length 3

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = a | b

    # -----------------------------
    # ArkoudaArray | scalar bool
    # -----------------------------
    def test_or_scalar_true(self):
        a = self.make_bool([False, False, True])

        result = a | True

        np.testing.assert_array_equal(result._data.to_ndarray(), np.array([True, True, True]))

    def test_or_scalar_false(self):
        a = self.make_bool([True, False, True])

        result = a | False

        np.testing.assert_array_equal(result._data.to_ndarray(), np.array([True, False, True]))

    # -----------------------------
    # scalar bool | ArkoudaArray  (__ror__)
    # -----------------------------
    def test_ror_scalar_true(self):
        a = self.make_bool([False, True, False])

        result = True | a  # triggers __ror__

        np.testing.assert_array_equal(result._data.to_ndarray(), np.array([True, True, True]))

    def test_ror_scalar_false(self):
        a = self.make_bool([False, True, False])

        result = False | a  # triggers __ror__

        np.testing.assert_array_equal(result._data.to_ndarray(), np.array([False, True, False]))

    # -----------------------------
    # ArkoudaArray | numpy/list
    # -----------------------------
    def test_or_with_numpy_array(self):
        a = self.make_bool([True, False, True])
        b = np.array([False, True, False])

        result = a | b

        np.testing.assert_array_equal(result._data.to_ndarray(), np.array([True, True, True]))

    def test_or_with_python_list(self):
        a = self.make_bool([True, False, False])
        b = [False, False, True]

        result = a | b

        np.testing.assert_array_equal(result._data.to_ndarray(), np.array([True, False, True]))

    def test_or_numpy_length_mismatch_raises(self):
        a = self.make_bool([True, False])
        b = np.array([True, False, True])

        with pytest.raises(ValueError, match="Lengths must match"):
            _ = a | b

    # -----------------------------
    # Unsupported types
    # -----------------------------
    def test_or_unsupported_rhs_returns_notimplemented(self):
        a = self.make_bool([True, False])

        class Weird:
            pass

        # Python should then try Weird.__ror__, which also doesn't exist,
        # resulting in a TypeError for unsupported operands.
        with pytest.raises(TypeError):
            _ = a | Weird()

    def test_or_numeric_array_returns_notimplemented(self):
        # Since you only support OR on bool dtype
        a = ArkoudaArray(ak_array([1, 2, 3], dtype=ak.int64))
        b = self.make_bool([True, False, True])

        assert (a.__or__(b)) is NotImplemented

    def test_or_with_python_sequence_len1_broadcasts(self):
        arr = ArkoudaArray(ak.array([True, False, True]))
        result = arr | [True]
        assert result._data.sum() == 3  # all True

    def test_or_with_numpy_array_len1_broadcasts(self):
        arr = ArkoudaArray(ak.array([True, False, True]))
        result = arr | np.array([False], dtype=bool)
        assert result._data.sum() == 2  # unchanged: True, False, True

    def test_or_with_python_sequence_length_mismatch_raises(self):
        arr = ArkoudaArray(ak.array([True, False, True]))
        with pytest.raises(ValueError, match="Lengths must match"):
            _ = arr | [True, False]  # len 2, not 1 and not len(arr)


class TestArkoudaArrayReduce:
    @pytest.mark.parametrize(
        "name",
        ["sum", "prod", "min", "max", "mean", "var", "std"],
    )
    def test_reduce_numeric_matches_pandas(self, name):
        data = np.array([1, 2, 3, 2], dtype=np.float64)
        arr = ArkoudaArray(ak.array(data))

        got = arr._reduce(name)

        s = pd.Series(data)
        # pandas methods; ddof=1 for var/std by default
        exp = getattr(s, name)()

        assert np.isfinite(got)
        assert got == pytest.approx(exp)

    @pytest.mark.parametrize(
        "data, exp_count",
        [
            ([1, 2, 3, 2], 4),
            ([10], 1),
            ([5, 5, 5], 3),
        ],
    )
    def test_reduce_count(self, data, exp_count):
        arr = ArkoudaArray(ak.array(data))
        assert arr._reduce("count") == exp_count

    @pytest.mark.parametrize(
        "vals, name, exp",
        [
            ([True, True], "all", True),
            ([True, False], "all", False),
            ([False, False], "any", False),
            ([False, True], "any", True),
        ],
    )
    def test_reduce_any_all_bool(self, vals, name, exp):
        arr = ArkoudaArray(ak.array(vals))
        assert bool(arr._reduce(name)) == exp

    def test_reduce_or_and_bool(self):
        arr = ArkoudaArray(ak.array([True, False, True]))
        assert bool(arr._reduce("or")) is True
        assert bool(arr._reduce("and")) is False

    @pytest.mark.parametrize(
        "vals, op, exp",
        [
            ([1, 2, 3], "all", True),
            ([1, 0, 3], "all", False),
            ([0, 0, 0], "any", False),
            ([0, 2, 0], "any", True),
            ([1, 2, 3], "and", True),  # alias for all
            ([1, 0, 3], "and", False),
            ([0, 0, 0], "or", False),  # alias for any
            ([0, 2, 0], "or", True),
        ],
    )
    def test_reduce_truthy_ops_on_int(self, vals, op, exp):
        arr = ArkoudaArray(ak.array(vals, dtype=ak.int64))
        assert bool(arr._reduce(op)) == exp

    def test_reduce_argmin_argmax_matches_numpy_first_tie(self):
        data = np.array([5, 1, 1, 9, 9], dtype=np.int64)
        arr = ArkoudaArray(ak.array(data))

        assert arr._reduce("argmin") == int(np.argmin(data))
        assert arr._reduce("argmax") == int(np.argmax(data))

    def test_reduce_first(self):
        arr = ArkoudaArray(ak.array([10, 20, 30]))
        assert arr._reduce("first") == 10

    def test_reduce_first_empty_raises(self):
        arr = ArkoudaArray(ak.array([], dtype=ak.int64))
        with pytest.raises((IndexError, ValueError)):
            _ = arr._reduce("first")

    def test_reduce_unknown_name_raises_typeerror(self):
        arr = ArkoudaArray(ak.arange(5))
        with pytest.raises(TypeError):
            arr._reduce("does_not_exist")

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_accepts_skipna_flag(self, skipna):
        arr = ArkoudaArray(ak.array([1.0, np.nan, 2.0]))
        # whichever semantics you currently implement, it should not error
        _ = arr._reduce("sum", skipna=skipna)


class TestArkoudaArraySetitem:
    def test_setitem_scalar_integer_position(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data[:])
        arr[1] = 42
        assert arr[1] == 42

    def test_setitem_arkouda_int_indexer(self):
        ak_data = ak.arange(10)
        arr = ArkoudaArray(ak_data[:])
        arr[ak.array([0, 2])] = [99, 88]
        assert arr[0] == 99
        assert arr[2] == 88

    def test_scalar_setitem_numpy_integer_indexer(self):
        data = ak.arange(5)
        arr = ArkoudaArray(data)

        idx = np.array([1, 3], dtype=np.int64)
        arr[idx] = 99

        assert np.array_equal(
            arr.to_ndarray(),
            np.array([0, 99, 2, 99, 4]),
        )

    def test_scalar_setitem_numpy_boolean_mask(self):
        data = ak.arange(5)
        arr = ArkoudaArray(data)

        mask = arr.to_ndarray() % 2 == 0  # True at positions 0, 2, 4
        arr[mask] = -1

        assert np.array_equal(
            arr.to_ndarray(),
            np.array([-1, 1, -1, 3, -1]),
        )

    def test_setitem_with_python_sequence_value(self):
        data = ak.arange(5)
        arr = ArkoudaArray(data)

        idx = np.array([1, 3, 4], dtype=np.int64)
        arr[idx] = [10, 20, 30]

        assert np.array_equal(
            arr.to_ndarray(),
            np.array([0, 10, 2, 20, 30]),
        )

    def test_setitem_with_arkoudaarray_value(self):
        data = ak.arange(5)
        arr = ArkoudaArray(data)

        other = ArkoudaArray(ak.arange(10, 15))
        idx = np.array([1, 3, 4], dtype=np.int64)

        arr[idx] = other[idx]

        # other[idx] is [11, 13, 14]
        assert np.array_equal(
            arr.to_ndarray(),
            np.array([0, 11, 2, 13, 14]),
        )

    def test_setitem_with_pdarray_value(self):
        data = ak.arange(5)
        arr = ArkoudaArray(data)

        values = ak.arange(100, 105)  # pdarray
        idx = np.array([0, 2, 4], dtype=np.int64)

        arr[idx] = values[idx]

        # values[idx] is [100, 102, 104]
        assert np.array_equal(
            arr.to_ndarray(),
            np.array([100, 1, 102, 3, 104]),
        )

    def test_scalar_fast_path_does_not_wrap_pdarray(self):
        """
        A bit white-box: make sure scalar assignment works without requiring
        array conversion for the value (no crash, correct result).
        """
        data = ak.arange(3)
        arr = ArkoudaArray(data)

        # This should go through the scalar fast path in __setitem__
        arr[1] = 777

        assert isinstance(arr._data, pdarray)
        assert np.array_equal(
            arr.to_ndarray(),
            np.array([0, 777, 2]),
        )

    def test_setitem_empty_list_noop(self):
        arr = ArkoudaArray(ak.arange(5))

        before = arr.to_numpy().copy()

        # setitem with empty list should do nothing
        arr[[]] = 99

        after = arr.to_numpy()

        assert (after == before).all()

    def test_setitem_python_list_of_ints_indexer(self):
        arr = ArkoudaArray(ak.arange(5))

        arr[[1, 3]] = 99

        np.testing.assert_array_equal(arr.to_ndarray(), np.array([0, 99, 2, 99, 4]))

    def test_setitem_python_list_of_bools_indexer(self):
        arr = ArkoudaArray(ak.arange(5))

        arr[[True, False, True, False, True]] = -1

        np.testing.assert_array_equal(arr.to_ndarray(), np.array([-1, 1, -1, 3, -1]))

    def test_setitem_numpy_uint64_indexer(self):
        arr = ArkoudaArray(ak.arange(5))

        idx = np.array([0, 4], dtype=np.uint64)
        arr[idx] = 777

        np.testing.assert_array_equal(arr.to_ndarray(), np.array([777, 1, 2, 3, 777]))

    def test_setitem_rejects_unsupported_list_element_type(self):
        arr = ArkoudaArray(ak.arange(5))

        with pytest.raises(TypeError):
            arr[["nope"]] = 1

    def test_setitem_mixed_index_dtype_not_supported(self):
        arr = ArkoudaArray(ak.arange(5))

        # Mixed list: bool + int should be rejected (mirror getitem behavior)
        with pytest.raises(NotImplementedError):
            arr[[True, 1, 2]] = 9

    def test_setitem_empty_numpy_int_indexer_noop(self):
        arr = ArkoudaArray(ak.arange(5))
        before = arr.to_ndarray().copy()

        empty = np.array([], dtype=np.int64)
        arr[empty] = 123  # should be a no-op

        after = arr.to_ndarray()
        np.testing.assert_array_equal(after, before)

    def test_setitem_slice_scalar_value(self):
        arr = ArkoudaArray(ak.arange(6))

        # slices are currently allowed by __setitem__ (passed through to pdarray)
        arr[2:5] = 9

        np.testing.assert_array_equal(arr.to_ndarray(), np.array([0, 1, 9, 9, 9, 5]))


class TestArkoudaArrayGetitem:
    def _make_array(self):
        # Small, simple fixture for all tests
        data = ak.arange(5)  # array([0, 1, 2, 3, 4])
        return ArkoudaArray(data)

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

    def test_getitem_scalar_returns_python_scalar(self):
        arr = self._make_array()

        result = arr[2]
        # Should be a scalar, not an ArkoudaArray
        assert not isinstance(result, ArkoudaArray)
        assert isinstance(result, (int, np.integer))
        assert result == 2

        # Negative index also returns scalar
        result_neg = arr[-1]
        assert isinstance(result_neg, (int, np.integer))
        assert result_neg == 4

    def test_getitem_slice_returns_arkouda_array(self):
        arr = self._make_array()

        result = arr[1:4]
        assert isinstance(result, ArkoudaArray)

        np.testing.assert_array_equal(result.to_ndarray(), np.array([1, 2, 3]))

    def test_getitem_numpy_int_array_indexer(self):
        arr = self._make_array()
        idx = np.array([0, 3], dtype=np.int64)

        result = arr[idx]
        assert isinstance(result, ArkoudaArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array([0, 3]))

    def test_getitem_numpy_bool_array_indexer(self):
        arr = self._make_array()
        mask = np.array([True, False, True, False, True])

        result = arr[mask]
        assert isinstance(result, ArkoudaArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array([0, 2, 4]))

    def test_getitem_python_list_of_ints(self):
        arr = self._make_array()

        result = arr[[1, 4]]
        assert isinstance(result, ArkoudaArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array([1, 4]))

    def test_getitem_python_list_of_bools(self):
        arr = self._make_array()

        result = arr[[True, False, True, False, True]]
        assert isinstance(result, ArkoudaArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array([0, 2, 4]))

    def test_getitem_empty_list_returns_empty_array(self):
        arr = self._make_array()

        result = arr[[]]
        assert isinstance(result, ArkoudaArray)
        # Underlying pdarray should be empty
        assert result._data.size == 0
        # And round-trip to NumPy should be empty as well
        assert result.to_ndarray().size == 0

    def test_getitem_arkouda_int_indexer(self):
        arr = self._make_array()
        ak_idx = ak.array([4, 0, 2])

        result = arr[ak_idx]
        assert isinstance(result, ArkoudaArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array([4, 0, 2]))

    def test_getitem_rejects_unsupported_list_element_type(self):
        arr = self._make_array()

        with pytest.raises(TypeError):
            _ = arr[["not", "ints", "or", "bools"]]

    def test_getitem_numpy_unsigned_int_indexer(self):
        arr = self._make_array()
        idx = np.array([1, 3], dtype=np.uint64)

        result = arr[idx]
        assert isinstance(result, ArkoudaArray)
        np.testing.assert_array_equal(result.to_ndarray(), np.array([1, 3]))

    def test_mixed_index_dtype_not_supported(self):
        arr = ArkoudaArray(ak.arange(5))

        # Mixed list: bool + int → mixed dtypes
        idx = [True, 1, 2, 0, 3]

        with pytest.raises(NotImplementedError):
            _ = arr[idx]
