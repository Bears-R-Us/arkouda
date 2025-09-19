import numpy as np
import pytest

import arkouda as ak
from arkouda.pandas.extension._arkouda_base_array import ArkoudaBaseArray


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

    @pytest.fixture(scope="module")
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
    def test_len_matches_expected(self, data_triplets, label_idx):
        label, ak_obj, expected = data_triplets[label_idx]
        arr = ArkoudaBaseArray(ak_obj)
        assert len(arr) == expected, f"len failed for {label}"

    @pytest.mark.parametrize("label_idx", [0, 1, 2])
    def test_len_matches_backend_len(self, data_triplets, label_idx):
        label, ak_obj, _ = data_triplets[label_idx]
        arr = ArkoudaBaseArray(ak_obj)
        assert len(arr) == len(ak_obj), f"wrapper len mismatch for {label}"

    @pytest.mark.parametrize("label_idx", [0, 1, 2])
    def test_len_consistency_after_concat(self, data_triplets, label_idx):
        """
        Sanity check that length tracks when we build a bigger backend object
        and wrap it again (helps catch off-by-one issues in __len__).
        """
        label, ak_obj, expected = data_triplets[label_idx]
        # duplicate and concatenate
        if hasattr(ak_obj, "concatenate"):
            bigger = ak_obj.concatenate([ak_obj])
        else:
            # generic fallback using ak.concatenate on pdarray-like
            from arkouda.numpy.pdarraysetops import concatenate

            bigger = concatenate([ak_obj, ak_obj])

        arr_big = ArkoudaBaseArray(bigger)
        assert len(arr_big) == 2 * len(ak_obj), f"concat len wrong for {label}"

    def test_len_zero_length_cases(self):
        """
        Edge cases: zero-length arrays across pdarray and Strings
        (Categorical empty creation via empty Strings).
        """
        empty_pd = ak.array(np.asarray([], dtype="int64"))
        empty_s = ak.array([], dtype="str_")
        empty_cat = ak.Categorical(empty_s)

        for label, obj in [("pdarray", empty_pd), ("Strings", empty_s), ("Categorical", empty_cat)]:
            arr = ArkoudaBaseArray(obj)
            assert len(arr) == 0, f"expected zero length for empty {label}"

    # --- Fast-path: input already Arkouda -------------------------------------------------------------

    @pytest.mark.parametrize("label", ["pdarray", "Strings"])
    def test_from_sequence_fastpath_identity_when_no_copy_and_no_dtype(self, label):
        ak_obj = self.base_objs()[label]
        arr = ArkoudaBaseArray._from_sequence(ak_obj, dtype=None, copy=False)
        assert isinstance(arr, ArkoudaBaseArray)
        # Should not allocate a new backend object
        assert arr._data is ak_obj

    @pytest.mark.parametrize("label", ["pdarray", "Strings"])
    def test_from_sequence_fastpath_copy_when_copy_true(self, label):
        ak_obj = self.base_objs()[label]
        arr = ArkoudaBaseArray._from_sequence(ak_obj, copy=True)
        assert isinstance(arr, ArkoudaBaseArray)
        # A physical copy should be made
        assert arr._data is not ak_obj
        # But contents should be equal
        np.testing.assert_array_equal(arr.to_numpy(), ak_obj.to_ndarray())

    def test_from_sequence_fastpath_categorical_identity(self):
        ak_obj = self.base_objs()["Categorical"]
        arr = ArkoudaBaseArray._from_sequence(ak_obj, copy=False)
        assert isinstance(arr, ArkoudaBaseArray)
        assert arr._data is ak_obj
        # Length preserved
        assert len(arr) == len(ak_obj)

    def test_from_sequence_fastpath_dtype_cast_pdarray(self):
        nums = self.base_objs()["pdarray"]  # int64
        # Cast to float64
        arr = ArkoudaBaseArray._from_sequence(nums, dtype="float64", copy=False)
        assert isinstance(arr, ArkoudaBaseArray)
        assert arr._data is not nums, "astype should produce a new backend object"
        assert str(arr._data.dtype) == "float64"
        np.testing.assert_allclose(arr.to_numpy(), nums.to_ndarray().astype("float64"))

    def test_from_sequence_fastpath_dtype_same_is_noop_pdarray(self):
        nums = self.base_objs()["pdarray"]
        arr = ArkoudaBaseArray._from_sequence(nums, dtype="int64", copy=False)
        # dtype matches; should avoid extra cast
        assert arr._data is nums

    # --- Fallback path: non-Arkouda inputs ------------------------------------------------------------

    def test_from_sequence_with_python_list(self):
        data = [10, 20, 30]
        arr = ArkoudaBaseArray._from_sequence(data)
        assert isinstance(arr, ArkoudaBaseArray)
        np.testing.assert_array_equal(arr.to_numpy(), np.array(data))

    def test_from_sequence_with_numpy_array_and_dtype(self):
        data = np.array([1, 2, 3], dtype="uint8")
        arr = ArkoudaBaseArray._from_sequence(data, dtype="int64")
        assert str(arr._data.dtype) == "int64"
        np.testing.assert_array_equal(arr.to_numpy(), data.astype("int64"))

    def test_from_sequence_scalar_normalizes_to_len1(self):
        arr = ArkoudaBaseArray._from_sequence(42)
        assert len(arr) == 1
        np.testing.assert_array_equal(arr.to_numpy(), np.array([42]))

    # --- pandas interop -------------------------------------------------------------------------------

    @pytest.mark.parametrize("box", ["Series", "Index"])
    def test_from_sequence_pandas_box(self, box):
        pd = pytest.importorskip("pandas")
        data = pd.Series([5, 6, 7]) if box == "Series" else pd.Index([5, 6, 7])
        arr = ArkoudaBaseArray._from_sequence(data)
        assert isinstance(arr, ArkoudaBaseArray)
        np.testing.assert_array_equal(arr.to_numpy(), np.array([5, 6, 7]))

    # --- Sanity / structural checks -------------------------------------------------------------------

    @pytest.mark.parametrize(
        "payload",
        [
            [1, 2, 3],
            np.array([1.5, 2.5, 3.5], dtype="float64"),
        ],
    )
    def test_from_sequence_returns_cls_instance(self, payload):
        arr = ArkoudaBaseArray._from_sequence(payload)
        assert isinstance(arr, ArkoudaBaseArray)

    def test_from_sequence_len_matches_input_len(self):
        payload = list(range(11))
        arr = ArkoudaBaseArray._from_sequence(payload)
        assert len(arr) == len(payload)

    # ------------------------ pdarray: dtype + copy semantics ------------------------

    def test_pdarray_default_no_args(self):
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])
        out = arr.to_numpy()
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.dtype("int64")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype="int64"))

    def test_pdarray_dtype_cast_to_float64(self):
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])  # int64 backend
        out = arr.to_numpy(dtype="float64")
        assert out.dtype == np.dtype("float64")
        np.testing.assert_allclose(out, np.array([1, 2, 3, 4], dtype="float64"))

    def test_pdarray_dtype_same_with_copy_true(self):
        """If dtype matches and copy=True, ensure result is equal and of same dtype."""
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])
        out = arr.to_numpy(dtype="int64", copy=True)
        assert out.dtype == np.dtype("int64")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype="int64"))
        # Sanity: local mutation doesn't change a freshly fetched array
        out[0] = 999
        fresh = arr.to_numpy()
        assert fresh[0] == 1

    def test_pdarray_copy_true_no_dtype(self):
        """copy=True with no dtype specified should still return equal values."""
        arr = ArkoudaBaseArray(self.base_objs()["pdarray"])
        out = arr.to_numpy(copy=True)
        assert out.dtype == np.dtype("int64")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4], dtype="int64"))
        out[1] = -123
        fresh = arr.to_numpy()
        assert fresh[1] == 2

    def test_pdarray_empty_edge_case(self):
        empty = ak.array(np.array([], dtype="int64"))
        arr = ArkoudaBaseArray(empty)
        out = arr.to_numpy()
        assert out.dtype == np.dtype("int64")
        assert out.size == 0

    # ----------------------------- Strings behavior -----------------------------

    def test_strings_default_to_numpy(self):
        arr = ArkoudaBaseArray(self.base_objs()["Strings"])
        out = arr.to_numpy()
        np.testing.assert_array_equal(out, np.array(["a", "bb", "ccc"], dtype=object))

    def test_strings_copy_true_does_not_affect_source(self):
        arr = ArkoudaBaseArray(self.base_objs()["Strings"])
        out = arr.to_numpy(copy=True)
        np.testing.assert_array_equal(out, np.array(["a", "bb", "ccc"], dtype=object))
        # local mutation shouldn't affect a fresh materialization
        out[0] = "zzz"
        fresh = arr.to_numpy()
        assert fresh[0] == "a"

    # ---------------------------- Categorical behavior ----------------------------

    def test_categorical_default_to_numpy_labels(self):
        cat = self.base_objs()["Categorical"]
        arr = ArkoudaBaseArray(cat)
        out = arr.to_numpy()
        # Should materialize the labels (object dtype)
        np.testing.assert_array_equal(out, np.array(["a", "bb", "ccc"], dtype=object))

    def test_categorical_copy_true_isolated_result(self):
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
