import numpy as np
import pandas as pd
import pytest

from pandas.api.extensions import ExtensionArray as PandasExtensionArray

import arkouda as ak

from arkouda import Categorical
from arkouda.pandas.extension import (
    ArkoudaArray,
    ArkoudaCategoricalArray,
    ArkoudaStringArray,
)


class TestArkoudaArrayExplodeInherited:
    def test_explode_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override `_explode`; it should inherit the
        default implementation from pandas.api.extensions.ExtensionArray.

        This locks in the fact that explode semantics are coming from pandas,
        not Arkouda-specific code.
        """
        # Not defined directly on ArkoudaArray
        assert "_explode" not in ArkoudaArray.__dict__

        # Method object is exactly the base ExtensionArray implementation
        assert ArkoudaArray._explode is PandasExtensionArray._explode

    @pytest.mark.parametrize(
        "ea_cls, values",
        [
            # Numeric ArkoudaArray examples
            (ArkoudaArray, [1, 2, 3]),
            (ArkoudaArray, [10.5, 20.5]),
            # String-backed example
            (ArkoudaStringArray, ["a", "b", "c"]),
            # Categorical-backed example
            (ArkoudaCategoricalArray, ["red", "blue", "red"]),
        ],
    )
    def test_explode_scalar_data_roundtrip(self, ea_cls, values):
        """
        For arrays that do NOT contain list-like elements, the default
        ExtensionArray._explode implementation should:

        - return a copy of the array (same values, same type),
        - return lengths as an ndarray of ones with shape (len(arr),).

        This should hold for ArkoudaArray, ArkoudaStringsArray, and
        ArkoudaCategoricalArray when they wrap scalar (non-list-like) data.
        """
        # Construct the underlying Arkouda data depending on EA type
        if ea_cls is ArkoudaCategoricalArray:
            # Categorical typically wraps an Arkouda Categorical built from values
            ak_data = Categorical(ak.array(values))
        else:
            # Numeric or string cases: ak.array should produce the right Arkouda type
            ak_data = ak.array(values)

        arr = ea_cls(ak_data)

        exploded, lengths = arr._explode()

        # 1) Same EA type
        assert isinstance(exploded, type(arr))

        # 2) Values unchanged
        orig_np = arr.to_numpy()
        exploded_np = exploded.to_numpy()
        np.testing.assert_array_equal(exploded_np, orig_np)

        # 3) lengths: ndarray of ones, one entry per original element
        assert isinstance(lengths, np.ndarray)
        assert lengths.shape == (len(arr),)
        assert np.issubdtype(lengths.dtype, np.integer)
        assert np.all(lengths == 1)

    @pytest.mark.parametrize(
        "ea_cls, values",
        [
            (ArkoudaArray, [1, 2, 3]),
            (ArkoudaStringArray, ["x", "y", "z"]),
            (ArkoudaCategoricalArray, ["low", "med", "high"]),
        ],
    )
    def test_explode_does_not_change_series_values(self, ea_cls, values):
        """
        Series.explode for non-list-like, non-object extension dtypes is
        effectively a no-op: it returns a Series copy with identical data
        and index. Arkouda-backed Series of all three EA types should
        follow that behavior.
        """
        if ea_cls is ArkoudaCategoricalArray:
            ak_data = Categorical(ak.array(values))
        else:
            ak_data = ak.array(values)

        arr = ea_cls(ak_data)
        s = pd.Series(arr)

        result = s.explode()

        # Expect a value-wise identical Series.
        # (dtype may differ depending on pandas internals, so we focus on
        # values and index equality.)
        assert list(result.index) == list(s.index)
        np.testing.assert_array_equal(result.to_numpy(), s.to_numpy())


class TestArkoudaArrayFormatterInherited:
    def test_formatter_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override _formatter; it should inherit the
        default implementation from pandas.api.extensions.ExtensionArray.
        """
        # Not defined directly on ArkoudaArray
        assert "_formatter" not in ArkoudaArray.__dict__

        # Method object comes from the pandas ExtensionArray base class
        assert ArkoudaArray._formatter is PandasExtensionArray._formatter

    # ------------------------------------------------------------------
    # Core tests for ArkoudaArray
    # ------------------------------------------------------------------

    def test_formatter_unboxed_uses_repr(self):
        """
        When boxed=False (the default), the returned formatter should behave
        like repr(x) for arbitrary Python scalars, per pandas docs.
        """
        a = ArkoudaArray(ak.arange(3))
        fmt = a._formatter(boxed=False)

        assert callable(fmt)

        value = "foo"
        result = fmt(value)

        assert isinstance(result, str)
        assert result == repr(value)  # "'foo'"
        assert result != str(value)  # "foo"

    def test_formatter_boxed_uses_str(self):
        """
        When boxed=True (array is being printed inside a Series/DataFrame),
        the returned formatter should behave like str(x).
        """
        a = ArkoudaArray(ak.arange(3))
        fmt = a._formatter(boxed=True)

        assert callable(fmt)

        value = "foo"
        result = fmt(value)

        assert isinstance(result, str)
        assert result == str(value)  # "foo"
        assert result != repr(value)  # "'foo'"

    def test_formatter_default_is_equivalent_to_boxed_false(self):
        """
        Calling _formatter() without arguments should be equivalent to
        _formatter(boxed=False).
        """
        a = ArkoudaArray(ak.arange(3))

        fmt_default = a._formatter()
        fmt_unboxed = a._formatter(boxed=False)

        for val in (0, 1.5, "foo"):
            assert fmt_default(val) == fmt_unboxed(val)

    # ------------------------------------------------------------------
    # NEW: Run the same formatter semantics tests for:
    #   - ArkoudaArray
    #   - ArkoudaStringArray
    #   - ArkoudaCategoricalArray
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaArray, lambda: ak.arange(3)),
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z"])),
            ),
        ],
    )
    def test_formatter_unboxed_repr_for_all_EAs(self, EA, make_data):
        """All Arkouda EAs should inherit pandas' unboxed formatter: repr(x)."""
        arr = EA(make_data())
        fmt = arr._formatter(boxed=False)

        for value in [0, 1.5, "foo"]:
            result = fmt(value)
            # Unboxed formatter should behave like repr(...)
            assert result == repr(value)
            # Only enforce inequality for strings, since for numbers
            # str(x) == repr(x) in Python.
            if isinstance(value, str):
                assert result != str(value)

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaArray, lambda: ak.arange(3)),
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z"])),
            ),
        ],
    )
    def test_formatter_boxed_str_for_all_EAs(self, EA, make_data):
        """All Arkouda EAs should inherit pandas' boxed formatter: str(x)."""
        arr = EA(make_data())
        fmt = arr._formatter(boxed=True)

        for value in [0, 1.5, "foo"]:
            result = fmt(value)
            # Boxed formatter should behave like str(...)
            assert result == str(value)
            # Only enforce inequality with repr for strings, since
            # for numbers str(x) == repr(x) in Python.
            if isinstance(value, str):
                assert result != repr(value)

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaArray, lambda: ak.arange(3)),
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z"])),
            ),
        ],
    )
    def test_formatter_default_equivalent_to_unboxed_for_all_EAs(self, EA, make_data):
        """Default call _formatter() should match _formatter(boxed=False) for all EAs."""
        arr = EA(make_data())
        fmt_default = arr._formatter()
        fmt_unboxed = arr._formatter(boxed=False)

        for value in [0, 1.5, "foo"]:
            assert fmt_default(value) == fmt_unboxed(value)


class TestArkoudaArrayFromScalarsInherited:
    def test_from_scalars_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not define _from_scalars itself; it should be
        inherited from pandas.api.extensions.ExtensionArray.
        """
        assert "_from_scalars" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "_from_scalars" in base.__dict__)
        assert owner is PandasExtensionArray

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            # numeric
            (ArkoudaArray, lambda: ak.arange(5)),
            # strings
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d", "e"])),
            # categorical
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y", "x"])),
            ),
        ],
    )
    def test_from_scalars_roundtrip_for_all_EAs(self, EA, make_data):
        """
        For all Arkouda extension arrays, _from_scalars should reconstruct
        a new EA instance from the list of Python scalars.
        """
        arr = EA(make_data())
        scalars = arr.tolist()

        b = EA._from_scalars(scalars, dtype=arr.dtype)

        # type matches
        assert isinstance(b, EA)

        # dtype matches
        assert b.dtype == arr.dtype

        # roundtrip scalars match original list
        assert b.tolist() == scalars

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaArray, lambda: ak.arange(3)),
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z"])),
            ),
        ],
    )
    def test_from_scalars_requires_dtype_kwonly_argument(self, EA, make_data):
        """
        _from_scalars has a required keyword-only 'dtype' argument. Omitting it
        should raise TypeError from Python's argument binding.
        """
        arr = EA(make_data())
        scalars = arr.tolist()

        with pytest.raises(TypeError):
            EA._from_scalars(scalars)  # missing dtype


class TestArkoudaArrayGetReprFooterInherited:
    def test_get_repr_footer_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not override _get_repr_footer; it should inherit
        the implementation from pandas.api.extensions.ExtensionArray.
        """
        assert "_get_repr_footer" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "_get_repr_footer" in base.__dict__)
        assert owner is PandasExtensionArray

    def _footer_text(self, footer):
        """Normalize return value: may be list[str] or str."""
        if isinstance(footer, list):
            assert all(isinstance(line, str) for line in footer)
            return " ".join(footer)
        assert isinstance(footer, str)
        return footer

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            # numeric
            (ArkoudaArray, lambda: ak.arange(5)),
            # strings
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d", "e"])),
            # categorical
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y", "x"])),
            ),
        ],
    )
    def test_get_repr_footer_contains_length_and_dtype(self, EA, make_data):
        """Footer from _get_repr_footer should include length and dtype for ALL EAs."""
        arr = EA(make_data())

        footer = arr._get_repr_footer()
        text = self._footer_text(footer)

        assert "Length" in text or "length" in text
        assert str(len(arr)) in text
        assert "dtype" in text
        assert str(arr.dtype) in text

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            # numeric empty
            (ArkoudaArray, lambda: ak.array([], dtype=ak.int64)),
            # empty strings
            (ArkoudaStringArray, lambda: ak.array([], dtype=ak.str_)),
            # empty categorical (must supply empty categorical)
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array([], dtype=ak.str_)),
            ),
        ],
    )
    def test_get_repr_footer_handles_empty_array(self, EA, make_data):
        """Footer for empty arrays should still report length 0 and the dtype."""
        arr = EA(make_data())

        footer = arr._get_repr_footer()
        text = self._footer_text(footer)

        assert "Length" in text or "length" in text
        assert "0" in text  # must show length=0 somewhere
        assert "dtype" in text
        assert str(arr.dtype) in text


class TestArkoudaArrayHashInherited:
    def test_hash_pandas_object_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not override _hash_pandas_object; it should inherit
        the implementation from pandas.api.extensions.ExtensionArray.
        """
        # Not defined directly on ArkoudaArray
        assert "_hash_pandas_object" not in ArkoudaArray.__dict__

        # Find which class in the MRO actually defines _hash_pandas_object
        owner = next(base for base in ArkoudaArray.mro() if "_hash_pandas_object" in base.__dict__)
        assert owner is PandasExtensionArray

    @pytest.mark.xfail(
        reason=("Fails because Depends on ArkoudaCategoricalArray.to_factorize_view #5101")
    )
    @pytest.mark.parametrize(
        "EA, make_data",
        [
            # numeric EA
            (ArkoudaArray, lambda: ak.array([10, 20, 10, 30])),
            # string EA
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "a", "c"])),
            # categorical EA
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "x", "z"])),
            ),
        ],
    )
    def test_hash_pandas_object_equal_values_have_equal_hash(self, EA, make_data):
        """
        For any Arkouda EA, hash_pandas_object on a Series should give equal
        hashes for equal values within that Series.

        We *do not* require the hashes to match a NumPy-backed Series; only
        internal consistency is required.
        """
        arr = EA(make_data())
        s = pd.Series(arr)

        hashes = pd.util.hash_pandas_object(s, index=False)
        # Basic shape / dtype checks
        assert len(hashes) == len(arr)
        assert hashes.dtype == "uint64"

        hashes_np = hashes.to_numpy(dtype="uint64")
        values = arr.tolist()

        # Group positions by value and check that all positions with the same
        # logical value have identical hash.
        groups: dict[object, set[int]] = {}
        for i, v in enumerate(values):
            groups.setdefault(v, set()).add(hashes_np[i])

        for v, hset in groups.items():
            # Each logical value should map to a single hash
            assert len(hset) == 1, f"value {v!r} had multiple hashes: {hset}"

    @pytest.mark.xfail(
        reason=("Fails because Depends on ArkoudaCategoricalArray.to_factorize_view #5101")
    )
    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaArray, lambda: ak.array([1, 2, 3, 4])),
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["p", "q", "r", "q"])),
            ),
        ],
    )
    def test_hash_pandas_object_is_stable_for_same_series(self, EA, make_data):
        """
        Hashing the same Arkouda-backed Series twice should give identical
        results for any Arkouda EA: numeric, string, or categorical.
        """
        arr = EA(make_data())
        s = pd.Series(arr)

        h1 = pd.util.hash_pandas_object(s, index=False)
        h2 = pd.util.hash_pandas_object(s, index=False)

        np.testing.assert_array_equal(
            h1.to_numpy(dtype="uint64"),
            h2.to_numpy(dtype="uint64"),
        )


class TestArkoudaArrayPutmask:
    def test_putmask_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override _putmask; it should inherit the
        implementation from pandas.api.extensions.ExtensionArray.
        """
        assert "_putmask" not in ArkoudaArray.__dict__
        owner = next(base for base in ArkoudaArray.mro() if "_putmask" in base.__dict__)
        assert owner is PandasExtensionArray

    #
    # ---- Numeric ArkoudaArray Tests (existing) ----
    #

    def test_putmask_scalar_value_matches_numpy(self):
        data = np.array([0, 1, 2, 3, 4], dtype="int64")
        mask = np.array([False, True, False, True, False], dtype=bool)
        value = 99

        expected = data.copy()
        expected[mask] = value

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.copy()

        ret = result._putmask(mask, value)
        assert ret is None

        out = result.to_numpy()
        assert out.dtype == expected.dtype
        assert np.array_equal(out, expected)

    def test_putmask_array_value_matches_numpy(self):
        data = np.array([0, 1, 2, 3, 4], dtype="int64")
        mask = np.array([True, False, True, False, True], dtype=bool)
        value = np.array([10, 11, 12, 13, 14], dtype="int64")

        expected = data.copy()
        expected[mask] = value[mask]

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.copy()

        result._putmask(mask, value)

        out = result.to_numpy()
        assert out.dtype == expected.dtype
        assert np.array_equal(out, expected)

    def test_putmask_empty_mask_is_noop(self):
        data = np.array([0, 1, 2, 3, 4], dtype=int)
        mask = np.zeros_like(data, dtype=bool)

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.copy()

        result._putmask(mask, 999)
        assert np.array_equal(result.to_numpy(), data)

    #
    # ---- Strings EA Tests ----
    #

    def test_putmask_strings(self):
        """
        _putmask should correctly update string values if ArkoudaStringArray
        implements __setitem__. Otherwise, it should raise NotImplementedError.
        """
        data = np.array(["a", "b", "c", "d"], dtype=object)
        mask = np.array([False, True, False, True], dtype=bool)
        value = "X"

        ak_arr = ArkoudaStringArray(ak.array(data))
        result = ak_arr.copy()

        try:
            result._putmask(mask, value)
        except NotImplementedError:
            pytest.skip("ArkoudaStringArray does not yet implement __setitem__.")

        expected = data.copy()
        expected[mask] = value

        assert result.tolist() == expected.tolist()

    #
    # ---- Categorical EA Tests ----
    #

    def test_putmask_categorical(self):
        """
        _putmask should update categorical values if ArkoudaCategoricalArray
        implements __setitem__. Otherwise, skip until implemented.
        """
        data = np.array(["x", "y", "x", "z"], dtype=object)
        mask = np.array([True, False, True, False], dtype=bool)
        value = "Q"

        ak_cat = ak.Categorical(ak.array(data))
        arr = ArkoudaCategoricalArray(ak_cat)
        result = arr.copy()

        try:
            result._putmask(mask, value)
        except NotImplementedError:
            pytest.skip("ArkoudaCategoricalArray does not yet implement __setitem__.")

        expected = data.copy()
        expected[mask] = value

        assert result.tolist() == expected.tolist()


class TestArkoudaArrayRank:
    def test_rank_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not override _rank; it should inherit the
        implementation from pandas.api.extensions.ExtensionArray.
        """
        # Not defined directly on ArkoudaArray
        assert "_rank" not in ArkoudaArray.__dict__

        # The owner in the MRO should be the pandas ExtensionArray base class
        owner = next(base for base in ArkoudaArray.mro() if "_rank" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_rank_default_matches_pandas_series(self):
        """
        Series.rank() on a Series backed by ArkoudaArray should give the same
        numeric results as rank() on a plain NumPy-backed Series with the
        same data. This exercises the inherited _rank implementation.
        """
        values = np.array([10, 20, 10, 30], dtype="int64")

        # Expected: plain pandas Series
        s_expected = pd.Series(values)
        expected = s_expected.rank()  # default: method='average', ascending=True

        # Arkouda-backed Series
        ak_arr = ArkoudaArray(ak.array(values))
        s_ak = pd.Series(ak_arr)
        result = s_ak.rank()

        # Same index
        assert list(result.index) == list(expected.index)

        # Same numeric ranks (float dtype)
        np.testing.assert_allclose(result.to_numpy(), expected.to_numpy())

    def test_rank_with_ties_and_descending(self):
        """Check that rank with ties and ascending=False matches pandas."""
        values = np.array([5, 1, 5, 2], dtype="int64")

        s_expected = pd.Series(values)
        expected = s_expected.rank(ascending=False, method="average")

        ak_arr = ArkoudaArray(ak.array(values))
        s_ak = pd.Series(ak_arr)
        result = s_ak.rank(ascending=False, method="average")

        assert list(result.index) == list(expected.index)
        np.testing.assert_allclose(result.to_numpy(), expected.to_numpy())

    # ------------------------------------------------------------------
    # New: String and Categorical examples (plus numeric for symmetry)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            # numeric EA
            (ArkoudaArray, lambda: ak.array([10, 20, 10, 30])),
            # string EA
            (ArkoudaStringArray, lambda: ak.array(["b", "a", "b", "c"])),
            # categorical EA
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "x", "z"])),
            ),
        ],
    )
    def test_rank_default_matches_pandas_for_all_eas(self, EA, make_data):
        """
        For numeric, string, and categorical Arkouda EAs, Series.rank()
        should agree with pandas when applied to the logical values.

        We compare the Arkouda-backed Series to a baseline pandas Series
        constructed from arr.tolist(), so we're testing the semantics of
        the inherited _rank implementation rather than any Arkouda-specific
        code paths.
        """
        # Arkouda EA
        arr = EA(make_data())
        s_ak = pd.Series(arr)

        # Baseline: plain pandas Series built from the logical values
        baseline_values = arr.tolist()
        s_expected = pd.Series(baseline_values)
        expected = s_expected.rank()  # default args

        result = s_ak.rank()

        # Same index
        assert list(result.index) == list(expected.index)

        # Same numeric ranks
        np.testing.assert_allclose(
            result.to_numpy(),
            expected.to_numpy(),
        )


class TestArkoudaArrayRepr2D:
    def test_repr_2d_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not override _repr_2d; it should inherit the
        implementation from pandas.api.extensions.ExtensionArray.
        """
        assert "_repr_2d" not in ArkoudaArray.__dict__
        owner = next(base for base in ArkoudaArray.mro() if "_repr_2d" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_repr_2d_currently_raises_typeerror_for_numeric_arkoudaarray(self):
        """
        Calling _repr_2d() directly on a *numeric* ArkoudaArray currently fails
        with a TypeError, because the base implementation passes a scalar to
        format_object_summary, which then tries to call len(...) on it.

        We lock that in as the current behavior for ArkoudaArray so that any
        future fix (e.g. overriding _repr_2d) can update this test.
        """
        a = ArkoudaArray(ak.arange(5))

        with pytest.raises(TypeError):
            a._repr_2d()

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d", "e"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y", "x"])),
            ),
        ],
    )
    def test_repr_2d_strings_and_categoricals_do_not_error(self, EA, make_data):
        """
        For string and categorical Arkouda EAs, calling _repr_2d() should
        succeed (no exception). Depending on pandas / EA implementation
        details, it may return either a boolean flag or a string
        representation. We only lock in that it does not crash, and that
        any string result looks like a normal EA repr with length/dtype.
        """
        arr = EA(make_data())

        result = arr._repr_2d()

        # Type may be bool (classic pandas flag) or str (full repr); both are OK.
        assert isinstance(result, (bool, str))

        if isinstance(result, str):
            # Basic sanity checks that it looks like a repr footer/body.
            assert "Length" in result or "length" in result
            assert "dtype" in result

    @pytest.mark.parametrize(
        "EA, make_data, expected_strings",
        [
            # numeric
            (ArkoudaArray, lambda: ak.arange(5), ["0", "1", "2", "3", "4"]),
            # strings
            (
                ArkoudaStringArray,
                lambda: ak.array(["a", "b", "c", "d", "e"]),
                ["a", "b", "c", "d", "e"],
            ),
            # categorical
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y", "x"])),
                ["x", "y", "z"],
            ),
        ],
    )
    def test_series_repr_sensible_for_all_EAs(self, EA, make_data, expected_strings):
        """
        A Series backed by any Arkouda EA should have a sensible repr that
        includes the values and a dtype line. This is the user-facing path
        and should not crash even though _repr_2d itself may fail for
        numeric ArkoudaArray.
        """
        arr = EA(make_data())
        s = pd.Series(arr, name="foo")

        rep = repr(s)

        # repr shouldn't be empty and should contain the Series name
        assert "foo" in rep
        # Expect logical values to appear
        for v in expected_strings:
            assert str(v) in rep

        # Just require that some dtype footer is present; exact text is an impl detail
        assert "dtype:" in rep


class TestArkoudaArrayValuesForArgsort:
    def test_values_for_argsort_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not override _values_for_argsort; it should inherit
        the implementation from pandas.api.extensions.ExtensionArray.
        """
        # Not defined directly on ArkoudaArray
        assert "_values_for_argsort" not in ArkoudaArray.__dict__

        # The owner in the MRO should be the pandas ExtensionArray base class
        owner = next(base for base in ArkoudaArray.mro() if "_values_for_argsort" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_values_for_argsort_simple_ints_matches_raw_values(self):
        """
        For simple integer data without missing values, _values_for_argsort()
        should essentially be the underlying values as a NumPy array, so that
        np.argsort on it matches the natural sort order.
        """
        values = np.array([3, 1, 4, 1, 5], dtype="int64")

        ak_arr = ArkoudaArray(ak.array(values))
        vfa = ak_arr._values_for_argsort()

        # Should be a NumPy array of same dtype and values
        assert isinstance(vfa, np.ndarray)
        assert vfa.dtype == values.dtype
        np.testing.assert_array_equal(vfa, values)

        # argsort on _values_for_argsort should match argsort on the raw data
        idx_vfa = np.argsort(vfa)
        idx_expected = np.argsort(values)
        np.testing.assert_array_equal(idx_vfa, idx_expected)

    def test_values_for_argsort_with_nans_agrees_with_numpy_order(self):
        """
        When values contain NaN, _values_for_argsort() should produce an array
        such that np.argsort on it induces the same sorted order (including
        NaN placement) as sorting the original NumPy array.
        """
        values = np.array([3.0, np.nan, 1.0, np.nan, 2.0], dtype="float64")

        ak_arr = ArkoudaArray(ak.array(values))
        vfa = ak_arr._values_for_argsort()

        # Should be a NumPy array of float dtype
        assert isinstance(vfa, np.ndarray)
        assert np.issubdtype(vfa.dtype, np.floating)

        # Compare sorted orders rather than raw vfa, to allow for internal
        # NaN-handling tricks in pandas' implementation
        idx_vfa = np.argsort(vfa, kind="mergesort")
        idx_expected = np.argsort(values, kind="mergesort")

        sorted_from_vfa = values[idx_vfa]
        sorted_expected = values[idx_expected]

        # The sorted values including NaNs should match
        np.testing.assert_array_equal(sorted_from_vfa, sorted_expected)

    @pytest.mark.xfail(
        reason=(
            "Series.sort_values() not yet supported for ArkoudaArray because "
            "isna(mask) → pdarray<bool> cannot be np.asarray-ed"
        )
    )
    def test_values_for_argsort_used_by_series_sort_values(self):
        values = np.array([4, 2, 5, 1, 3], dtype="int64")

        s_expected = pd.Series(values)
        expected = s_expected.sort_values()

        ak_arr = ArkoudaArray(ak.array(values))
        s_ak = pd.Series(ak_arr)
        result = s_ak.sort_values()

        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    # ------------------------------------------------------------------
    # New: Strings + Categorical examples
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "EA, make_data",
        [
            # numeric ArkoudaArray (for completeness under the same test)
            (ArkoudaArray, lambda: ak.array([3, 1, 4, 1, 5])),
            # string EA
            (ArkoudaStringArray, lambda: ak.array(["c", "a", "d", "a", "b"])),
            # categorical EA
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["z", "x", "y", "x", "z"])),
            ),
        ],
    )
    def test_values_for_argsort_induces_correct_sort_order_across_eas(self, EA, make_data):
        """
        For ArkoudaArray, ArkoudaStringArray, and ArkoudaCategoricalArray,
        _values_for_argsort() should return a NumPy array whose argsort order,
        when applied back to the logical Python-level values (tolist()), gives
        the same sorted values as normal Python sorting.

        This checks that the *ordering induced by vfa* is correct, without
        assuming anything about the exact dtype or representation (e.g. codes
        vs labels for categoricals).
        """
        # Construct EA and call the inherited implementation
        arr = EA(make_data())
        vfa = arr._values_for_argsort()

        assert isinstance(vfa, np.ndarray)
        assert len(vfa) == len(arr)

        # Argsort on the values-for-argsort
        idx_vfa = np.argsort(vfa, kind="mergesort")

        # Logical values as Python objects
        labels = np.array(arr.tolist(), dtype=object)

        # Sorted values according to the EA's argsort keys
        sorted_from_vfa = labels[idx_vfa]

        # Baseline: sort the labels with normal Python / NumPy ordering
        expected_sorted = np.array(sorted(labels.tolist()), dtype=object)

        np.testing.assert_array_equal(sorted_from_vfa, expected_sorted)


class TestArkoudaArrayDelete:
    def test_delete_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override delete; it should inherit the
        implementation from pandas.api.extensions.ExtensionArray.
        """
        # Not defined directly on ArkoudaArray
        assert "delete" not in ArkoudaArray.__dict__

        # The owner in the MRO should be the pandas ExtensionArray base class
        owner = next(base for base in ArkoudaArray.mro() if "delete" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_delete_single_position(self):
        """
        delete(loc) with loc as an int should remove that position and return
        a new ArkoudaArray, matching np.delete semantics on the underlying
        values.
        """
        values = np.array([0, 1, 2, 3, 4], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.delete(2)  # drop the "2"

        assert isinstance(result, type(ak_arr))
        expected = np.delete(values, 2)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_delete_multiple_positions_list(self):
        """
        delete(loc) with a list of indices should remove those positions and
        return a new ArkoudaArray.
        """
        values = np.array([0, 1, 2, 3, 4], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        # Remove positions 1 and 3
        result = ak_arr.delete([1, 3])

        assert isinstance(result, type(ak_arr))
        expected = np.delete(values, [1, 3])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_delete_with_slice(self):
        """delete(loc) with a slice should behave like np.delete with the same slice."""
        values = np.array([0, 1, 2, 3, 4, 5], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        # Remove positions 1, 2, 3
        result = ak_arr.delete(slice(1, 4))

        assert isinstance(result, type(ak_arr))
        expected = np.delete(values, slice(1, 4))
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_delete_with_negative_index(self):
        """
        Negative indices should be interpreted in the usual Python way, just
        like np.delete.
        """
        values = np.array([10, 20, 30, 40], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        # -1 deletes the last element
        result = ak_arr.delete(-1)

        assert isinstance(result, type(ak_arr))
        expected = np.delete(values, -1)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_delete_all_elements_results_in_empty_array(self):
        """
        Deleting all positions should return an empty ArkoudaArray with the
        same dtype.
        """
        values = np.array([0, 1, 2], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.delete([0, 1, 2])

        assert isinstance(result, type(ak_arr))
        out = result.to_numpy()
        assert out.shape == (0,)
        assert out.dtype == values.dtype

    # ------------------------------------------------------------------
    # New: Strings + Categorical (and numeric again) using tolist()
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "EA, make_data, loc",
        [
            # numeric
            (ArkoudaArray, lambda: ak.array([0, 1, 2, 3, 4]), 2),
            # strings
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d"]), 1),
            # categorical
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y"])),
                3,
            ),
        ],
    )
    def test_delete_single_position_across_eas(self, EA, make_data, loc):
        """
        delete(loc) should drop the given position for all Arkouda EAs
        (numeric, string, categorical), when viewed at the Python level
        via tolist().
        """
        arr = EA(make_data())
        result = arr.delete(loc)

        # Baseline expected via Python list semantics
        baseline = arr.tolist()
        expected = baseline[:loc] + baseline[loc + 1 :]

        assert isinstance(result, type(arr))
        assert result.tolist() == expected

    @pytest.mark.parametrize(
        "EA, make_data, positions",
        [
            # numeric
            (ArkoudaArray, lambda: ak.array([0, 1, 2, 3, 4]), [1, 3]),
            # strings
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d", "e"]), [0, 4]),
            # categorical
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y", "x"])),
                [1, 2],
            ),
        ],
    )
    def test_delete_multiple_positions_across_eas(self, EA, make_data, positions):
        """
        delete(loc) with a list of positions should remove the same logical
        entries when compared against Python list deletion on tolist().
        """
        arr = EA(make_data())
        result = arr.delete(positions)

        baseline = arr.tolist()
        # Emulate np.delete/list deletion: drop these indices
        mask = np.ones(len(baseline), dtype=bool)
        mask[positions] = False
        expected = [v for v, keep in zip(baseline, mask) if keep]

        assert isinstance(result, type(arr))
        assert result.tolist() == expected

    @pytest.mark.parametrize(
        "EA, make_data, slc",
        [
            # numeric
            (ArkoudaArray, lambda: ak.array([0, 1, 2, 3, 4, 5]), slice(1, 4)),
            # strings
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d", "e"]), slice(1, 3)),
            # categorical
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y", "x"])),
                slice(2, 5),
            ),
        ],
    )
    def test_delete_slice_across_eas(self, EA, make_data, slc):
        """
        delete(loc) with a slice should remove the same logical run of
        elements for all Arkouda EAs, compared to deleting that slice
        from the Python list produced by tolist().
        """
        arr = EA(make_data())
        result = arr.delete(slc)

        baseline = arr.tolist()
        expected = baseline.copy()
        del expected[slc]

        assert isinstance(result, type(arr))
        assert result.tolist() == expected


class TestArkoudaArrayDropna:
    def test_dropna_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override dropna; it should inherit the
        implementation from pandas.api.extensions.ExtensionArray.
        """
        # Not defined directly on ArkoudaArray
        assert "dropna" not in ArkoudaArray.__dict__

        # The owner in the MRO should be the pandas ExtensionArray base class
        owner = next(base for base in ArkoudaArray.mro() if "dropna" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_dropna_removes_nans_matches_pandas_ea(self):
        """
        dropna() on ArkoudaArray with float+NaN data should match the behavior
        of dropna() on a pandas nullable Float64 ExtensionArray with the same
        values (NaNs removed, order preserved).
        """
        values = [0.0, np.nan, 2.0, np.nan, 4.0]

        # pandas reference using an EA, not a plain Series
        pd_ea = pd.array(values, dtype="Float64")
        expected_ea = pd_ea.dropna()
        expected = expected_ea.to_numpy(dtype="float64")

        ak_arr = ArkoudaArray(ak.array(values))
        result = ak_arr.dropna()

        assert isinstance(result, type(ak_arr))
        out = result.to_numpy()
        np.testing.assert_array_equal(out, expected)

    def test_dropna_no_missing_is_noop(self):
        """
        When there are no missing values, dropna() should return an
        ArkoudaArray with the same values.
        """
        values = [0.0, 1.0, 2.0, 3.0]
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.dropna()

        assert isinstance(result, type(ak_arr))
        np.testing.assert_array_equal(result.to_numpy(), np.array(values, dtype=float))

    def test_dropna_all_missing_returns_empty_array(self):
        """
        When all values are missing, dropna() should return an empty
        ArkoudaArray of the same dtype.
        """
        values = [np.nan, np.nan]
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.dropna()

        assert isinstance(result, type(ak_arr))
        out = result.to_numpy()

        assert out.shape == (0,)
        # dtype should still be floating
        assert np.issubdtype(out.dtype, np.floating)

    # ------------------------------------------------------------------
    # New: Strings and Categorical examples
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "EA, make_data, pandas_dtype",
        [
            # string EA: no missings, dropna should be a no-op
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d"]), "string[python]"),
            # categorical EA: no missings, dropna should be a no-op
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y"])),
                "category",
            ),
        ],
    )
    def test_dropna_no_missing_is_noop_for_strings_and_categoricals(self, EA, make_data, pandas_dtype):
        """
        For string and categorical Arkouda-backed EAs with no missing values,
        dropna() should behave like pandas' dropna() on the corresponding
        pandas ExtensionArray: no rows are removed and values are preserved.
        """
        arr = EA(make_data())
        result = arr.dropna()

        assert isinstance(result, type(arr))

        # Baseline via pandas EA on the Python-level values
        baseline_values = arr.tolist()
        pd_ea = pd.array(baseline_values, dtype=pandas_dtype)
        expected = pd_ea.dropna().tolist()

        # In the no-missing case, this is effectively a no-op, but we rely
        # on pandas' behavior as the ground truth.
        assert result.tolist() == expected


class TestArkoudaArrayInsert:
    def test_insert_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override insert; it should inherit the
        implementation from pandas.api.extensions.ExtensionArray.
        """
        assert "insert" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "insert" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_insert_at_beginning(self):
        """
        insert(0, value) should prepend the scalar and return a new
        ArkoudaArray whose values match np.insert on the underlying data.
        """
        values = np.array([1, 2, 3], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.insert(0, 99)

        assert isinstance(result, type(ak_arr))
        expected = np.insert(values, 0, 99)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_insert_in_middle(self):
        """
        insert(loc, value) with 0 < loc < len should insert before that
        position, matching np.insert.
        """
        values = np.array([0, 1, 2, 3, 4], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.insert(2, 99)  # before the element "2"

        assert isinstance(result, type(ak_arr))
        expected = np.insert(values, 2, 99)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_insert_at_end(self):
        """insert(len(arr), value) should append the scalar, matching np.insert."""
        values = np.array([10, 20, 30], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.insert(len(ak_arr), 99)

        assert isinstance(result, type(ak_arr))
        expected = np.insert(values, len(values), 99)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    # ------------------------------------------------------------------
    # New: Strings and Categorical examples
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "EA, make_data, loc, value, expected",
        [
            # numeric EA
            (ArkoudaArray, lambda: ak.arange(3), 0, 99, [99, 0, 1, 2]),
            # string EA
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c"]), 0, "z", ["z", "a", "b", "c"]),
            # categorical EA (insert a value already in the categories)
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z"])),
                0,
                "y",
                ["y", "x", "y", "z"],
            ),
        ],
    )
    def test_insert_at_beginning_for_all_eas(self, EA, make_data, loc, value, expected):
        """
        insert(0, value) should work consistently for numeric, string,
        and categorical Arkouda-backed EAs.
        """
        arr = EA(make_data())
        result = arr.insert(loc, value)

        assert isinstance(result, type(arr))
        assert result.tolist() == expected

    @pytest.mark.parametrize(
        "EA, make_data, loc, value, expected",
        [
            # numeric EA
            (ArkoudaArray, lambda: ak.arange(5), 2, 99, [0, 1, 99, 2, 3, 4]),
            # string EA
            (
                ArkoudaStringArray,
                lambda: ak.array(["a", "b", "c", "d"]),
                2,
                "z",
                ["a", "b", "z", "c", "d"],
            ),
            # categorical EA
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y"])),
                2,
                "y",
                ["x", "y", "y", "z", "y"],
            ),
        ],
    )
    def test_insert_in_middle_for_all_eas(self, EA, make_data, loc, value, expected):
        """
        insert(loc, value) with 0 < loc < len should behave like list
        insertion on the underlying logical values for all Arkouda EAs.
        """
        arr = EA(make_data())
        result = arr.insert(loc, value)

        assert isinstance(result, type(arr))
        assert result.tolist() == expected

    @pytest.mark.parametrize(
        "EA, make_data, value, expected",
        [
            # numeric EA
            (ArkoudaArray, lambda: ak.arange(3), 99, [0, 1, 2, 99]),
            # string EA
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c"]), "z", ["a", "b", "c", "z"]),
            # categorical EA
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z"])),
                "x",
                ["x", "y", "z", "x"],
            ),
        ],
    )
    def test_insert_at_end_for_all_eas(self, EA, make_data, value, expected):
        """
        insert(len(arr), value) should append the scalar for all Arkouda
        EAs, preserving logical values.
        """
        arr = EA(make_data())
        result = arr.insert(len(arr), value)

        assert isinstance(result, type(arr))
        assert result.tolist() == expected


class TestArkoudaArrayIsin:
    def test_isin_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override isin; it should inherit the
        implementation from pandas.api.extensions.ExtensionArray.
        """
        assert "isin" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "isin" in base.__dict__)
        assert owner is PandasExtensionArray

    # ------------------------------------------------------------------
    # Numeric baseline tests
    # ------------------------------------------------------------------

    def test_isin_simple_matches_numpy(self):
        """
        Direct call to ArkoudaArray.isin should return a NumPy boolean array
        whose values match numpy.isin for the same input.
        """
        data = np.array([10, 20, 30, 40], dtype="int64")
        test_vals = [20, 40]

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.isin(test_vals)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

        expected = np.isin(data, test_vals)
        np.testing.assert_array_equal(result, expected)

    def test_isin_with_no_matches(self):
        data = np.array([1, 2, 3, 4], dtype="int64")
        test_vals = [99, 100]

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.isin(test_vals)

        expected = np.isin(data, test_vals)
        np.testing.assert_array_equal(result, expected)

    def test_isin_with_all_matches(self):
        data = np.array([5, 5, 5], dtype="int64")
        test_vals = [5]

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.isin(test_vals)

        expected = np.isin(data, test_vals)
        np.testing.assert_array_equal(result, expected)

    # ------------------------------------------------------------------
    # String and Categorical ExtensionArray tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "EA, make_data, test_vals",
        [
            # ArkoudaStringArray
            (
                ArkoudaStringArray,
                lambda: ak.array(["a", "b", "c", "b"]),
                ["b", "c"],
            ),
            # ArkoudaCategoricalArray
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "x", "z"])),
                ["x", "z"],
            ),
        ],
    )
    def test_isin_basic_for_strings_and_categoricals(self, EA, make_data, test_vals):
        """
        For ArkoudaStringArray and ArkoudaCategoricalArray, isin() should
        return a NumPy boolean array whose values match numpy.isin applied to
        the Python-level logical values (tolist()).
        """
        arr = EA(make_data())
        result = arr.isin(test_vals)

        # Always NumPy array
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

        # Convert logical values to numpy object array
        baseline = np.asarray(arr.tolist(), dtype=object)
        expected = np.isin(baseline, test_vals)

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "EA, make_data, test_vals",
        [
            (
                ArkoudaStringArray,
                lambda: ak.array(["a", "b", "c"]),
                ["q", "z"],
            ),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z"])),
                ["q"],
            ),
        ],
    )
    def test_isin_no_matches_for_strings_and_categoricals(self, EA, make_data, test_vals):
        """
        String and Categorical EAs: no test_vals match any element.
        Result should be an all-False NumPy boolean array.
        """
        arr = EA(make_data())
        result = arr.isin(test_vals)

        expected = np.isin(np.asarray(arr.tolist(), dtype=object), test_vals)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "EA, make_data, test_vals",
        [
            (
                ArkoudaStringArray,
                lambda: ak.array(["x", "x", "x"]),
                ["x"],
            ),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["a", "a", "a"])),
                ["a"],
            ),
        ],
    )
    def test_isin_all_matches_for_strings_and_categoricals(self, EA, make_data, test_vals):
        """
        All elements match → result should be all True for both String and
        Categorical Arkouda EAs.
        """
        arr = EA(make_data())
        result = arr.isin(test_vals)

        expected = np.isin(np.asarray(arr.tolist(), dtype=object), test_vals)
        np.testing.assert_array_equal(result, expected)


class TestArkoudaArrayMap:
    def test_map_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not override map; it should inherit the
        implementation from pandas.core.arrays.base.ExtensionArray.
        """
        assert "map" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "map" in base.__dict__)
        assert owner is PandasExtensionArray

    # ------------------------------------------------------------------
    # Numeric ArkoudaArray tests (baseline behavior)
    # ------------------------------------------------------------------

    def test_map_dict(self):
        """
        Mapping with a dictionary should behave like pandas: each value
        replaced according to the mapping dict.
        """
        data = np.array([1, 2, 3, 2], dtype="int64")
        mapping = {1: 10, 2: 20, 3: 30}

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.map(mapping)

        # pandas reference via nullable EA
        expected = pd.array(data).map(mapping)

        np.testing.assert_array_equal(
            np.asarray(result),
            np.asarray(expected),
        )

    def test_map_dict_with_missing_keys(self):
        """
        If a key is not in the mapping dict, pandas returns NaN.
        ArkoudaArray.map should follow that behavior.
        """
        data = np.array([1, 2, 99], dtype="int64")
        mapping = {1: 10, 2: 20}

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.map(mapping)

        expected = pd.array(data).map(mapping)

        result_arr = np.asarray(result)
        expected_arr = np.asarray(expected)

        # dtype will typically be float because of NaN
        assert result_arr.dtype == expected_arr.dtype
        np.testing.assert_array_equal(result_arr, expected_arr)

    def test_map_callable(self):
        """Mapping with a callable should apply the function to each element."""
        data = np.array([1, 2, 3], dtype="int64")

        def f(x):
            return x * x

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.map(f)

        expected = pd.array(data).map(f)

        np.testing.assert_array_equal(
            np.asarray(result),
            np.asarray(expected),
        )

    def test_map_callable_with_nan(self):
        """Callable mapping should receive NaN and preserve NaN in output."""
        data = np.array([1.0, np.nan, 3.0], dtype="float64")

        def f(x):
            return x + 1 if not pd.isna(x) else x

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.map(f)

        expected = pd.array(data).map(f)

        np.testing.assert_array_equal(
            np.asarray(result),
            np.asarray(expected),
        )

    def test_map_changes_dtype(self):
        """Mapping to strings should produce a string-like dtype similar to pandas."""
        data = np.array([1, 2, 3], dtype="int64")

        def f(x):
            return f"val_{x}"

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.map(f)

        expected = pd.array(data).map(f)

        result_arr = np.asarray(result)
        expected_arr = np.asarray(expected)

        assert result_arr.dtype == expected_arr.dtype
        np.testing.assert_array_equal(result_arr, expected_arr)

    # ------------------------------------------------------------------
    # String and Categorical Arkouda EAs
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "EA, values, mapping",
        [
            # Strings: map to upper-case
            (
                ArkoudaStringArray,
                ["a", "b", "c", "b"],
                {"a": "A", "b": "B", "c": "C"},
            ),
        ],
    )
    def test_map_dict_strings(self, EA, values, mapping):
        """
        For ArkoudaStringArray, dict mapping should match pandas'
        ExtensionArray.map on the logical values.
        """
        ak_data = ak.array(values)
        arr = EA(ak_data)

        result = arr.map(mapping)
        expected = pd.array(values).map(mapping)

        np.testing.assert_array_equal(
            np.asarray(result),
            np.asarray(expected),
        )

    @pytest.mark.xfail(
        reason=(
            "ArkoudaCategoricalArray.map currently fails because "
            "Index.astype(object, copy=...) calls ArkoudaCategoricalArray.astype "
            "with an unsupported 'copy' keyword."
        )
    )
    def test_map_dict_categorical_not_supported_yet(self):
        """
        Document current failure mode for ArkoudaCategoricalArray.map with dict
        mapping. Once ArkoudaCategoricalArray.astype supports 'copy=', this
        test should be updated to assert equality with pandas.
        """
        values = ["x", "y", "x", "z"]
        mapping = {"x": 1, "y": 2, "z": 3}

        ak_data = ak.Categorical(ak.array(values))
        arr = ArkoudaCategoricalArray(ak_data)

        # Currently raises TypeError deep in Index.astype(..., copy=False)
        arr.map(mapping)

    @pytest.mark.parametrize(
        "EA, values",
        [
            (ArkoudaStringArray, ["a", "b", "c"]),
        ],
    )
    def test_map_callable_strings(self, EA, values):
        """
        Callable mapping on string Arkouda EA should behave like
        pandas EA.map on the logical Python values.
        """
        ak_data = ak.array(values)
        arr = EA(ak_data)

        def f(x):
            return f"val_{x}"

        result = arr.map(f)
        expected = pd.array(values).map(f)

        np.testing.assert_array_equal(
            np.asarray(result),
            np.asarray(expected),
        )

    @pytest.mark.xfail(
        reason=(
            "ArkoudaCategoricalArray.map with callable is blocked by the same "
            "astype(copy=...) issue as the dict-mapping case."
        )
    )
    def test_map_callable_categorical_not_supported_yet(self):
        """
        Current behavior: callable mapping on ArkoudaCategoricalArray also
        fails due to astype(copy=...) on the categorical index.
        """
        values = ["x", "y", "z"]

        def f(x):
            return f"val_{x}"

        ak_data = ak.Categorical(ak.array(values))
        arr = ArkoudaCategoricalArray(ak_data)

        arr.map(f)

    @pytest.mark.parametrize(
        "EA, values, mapping",
        [
            # Some keys missing → NaNs/NA expected for strings
            (
                ArkoudaStringArray,
                ["a", "b", "c"],
                {"a": "A"},  # 'b' and 'c' missing
            ),
        ],
    )
    def test_map_dict_with_missing_keys_strings(self, EA, values, mapping):
        """
        For string EAs, missing dict keys should yield missing values in the
        same positions as pandas, though the concrete sentinel (np.nan vs <NA>)
        may differ.
        """
        ak_data = ak.array(values)
        arr = EA(ak_data)

        result = arr.map(mapping)
        expected = pd.array(values).map(mapping)

        result_arr = np.asarray(result)
        expected_arr = np.asarray(expected)

        # Dtype should still align at the NumPy level
        assert result_arr.dtype == expected_arr.dtype

        # Compare elementwise, treating ANY NA/NaN/<NA> as equivalent.
        assert result_arr.shape == expected_arr.shape

        for r, e in zip(result_arr, expected_arr):
            if pd.isna(e):
                # pandas may use <NA>, arkouda path may use np.nan; both should be "missing"
                assert pd.isna(r)
            else:
                assert r == e


class TestArkoudaArrayRavel:
    def test_ravel_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override ravel; it should inherit the
        implementation from pandas.core.arrays.base.ExtensionArray.
        """
        assert "ravel" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "ravel" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_ravel_returns_self_for_1d_array(self):
        """
        For 1D ExtensionArrays, ravel() must return self (see pandas
        ExtensionArray.ravel docstring).
        """
        a = ArkoudaArray(ak.arange(5))
        result = a.ravel()
        assert result is a

    def test_ravel_does_not_modify_values(self):
        """Ravel should be a no-op on 1D arrays; values should remain identical."""
        data = np.array([10, 20, 30, 40], dtype="int64")
        a = ArkoudaArray(ak.array(data))

        result = a.ravel()

        np.testing.assert_array_equal(result.to_numpy(), data)
        assert result is a

    def test_ravel_on_array_with_nans(self):
        """Ravel should still return self even when NaNs are present."""
        data = np.array([1.0, np.nan, 3.0], dtype="float64")
        a = ArkoudaArray(ak.array(data))

        result = a.ravel()

        assert result is a
        np.testing.assert_array_equal(result.to_numpy(), data)

    # ------------------------------------------------------------------
    # NEW TESTS FOR STRINGS + CATEGORICALS
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y"])),
            ),
        ],
    )
    def test_ravel_returns_self_for_strings_and_categoricals(self, EA, make_data):
        """ravel() must return self for all Arkouda EA subclasses."""
        arr = EA(make_data())
        result = arr.ravel()

        # pandas contract: ravel returns *the same object* for 1D EAs
        assert result is arr

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["u", "v", "u"])),
            ),
        ],
    )
    def test_ravel_preserves_values_for_strings_and_categoricals(self, EA, make_data):
        """Check that the values are unchanged by ravel() for non-numeric EAs."""
        arr = EA(make_data())
        data_list = arr.tolist()

        result = arr.ravel()

        assert result is arr
        assert result.tolist() == data_list


class TestArkoudaArrayRepeat:
    def test_repeat_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override repeat; it should inherit the
        implementation from pandas.core.arrays.base.ExtensionArray.
        """
        assert "repeat" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "repeat" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_repeat_scalar_count_matches_numpy(self):
        """
        repeat(n) with a scalar n should repeat each element n times.
        For simple numeric data, this should match numpy.repeat on the
        underlying values.
        """
        data = np.array([0, 1, 2], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(data))

        result = ak_arr.repeat(3)
        result_arr = np.asarray(result)

        expected = np.repeat(data, 3)

        np.testing.assert_array_equal(result_arr, expected)

    def test_repeat_per_element_counts_matches_numpy(self):
        """
        repeat(repeats) with an array-like 'repeats' giving per-element
        counts should match numpy.repeat semantics on the underlying data.
        """
        data = np.array([10, 20, 30, 40], dtype="int64")
        repeats = np.array([1, 2, 0, 3], dtype="int64")

        ak_arr = ArkoudaArray(ak.array(data))

        result = ak_arr.repeat(repeats)
        result_arr = np.asarray(result)

        expected = np.repeat(data, repeats)

        np.testing.assert_array_equal(result_arr, expected)

    def test_repeat_zero_times_returns_empty(self):
        """
        Repeating zero times should yield an empty array; dtype is allowed
        to change (pandas often promotes to float), so we only assert that
        it is empty and numeric-like.
        """
        data = np.array([1, 2, 3], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(data))

        result = ak_arr.repeat(0)
        result_arr = np.asarray(result)

        assert result_arr.shape == (0,)
        # Don't pin exact dtype; just require it's some numeric dtype.
        assert np.issubdtype(result_arr.dtype, np.number)

    # ------------------------------------------------------------------
    # NEW TESTS: Strings + Categoricals
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "EA, make_data, n",
        [
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c"]), 2),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "x"])),
                3,
            ),
        ],
    )
    def test_repeat_scalar_count_strings_and_categoricals(self, EA, make_data, n):
        """
        For string and categorical EAs, repeat(n) should logically repeat
        each value n times, matching simple Python list semantics.
        """
        arr = EA(make_data())
        values = arr.tolist()

        result = arr.repeat(n)
        result_list = result.tolist()

        expected = [v for v in values for _ in range(n)]

        assert isinstance(result, EA)
        assert result_list == expected

    @pytest.mark.parametrize(
        "EA, make_data, repeats",
        [
            (
                ArkoudaStringArray,
                lambda: ak.array(["a", "b", "c", "d"]),
                [1, 0, 2, 1],
            ),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y"])),
                [2, 1, 0, 1],
            ),
        ],
    )
    def test_repeat_per_element_counts_strings_and_categoricals(self, EA, make_data, repeats):
        """
        For string and categorical EAs, repeat(repeats) should replicate
        list-style semantics: each element v_i is repeated repeats[i] times.
        """
        arr = EA(make_data())
        values = arr.tolist()

        result = arr.repeat(repeats)
        result_list = result.tolist()

        expected = [v for v, r in zip(values, repeats) for _ in range(int(r))]

        assert isinstance(result, EA)
        assert result_list == expected

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "x"])),
            ),
        ],
    )
    def test_repeat_zero_times_empty_strings_and_categoricals(self, EA, make_data):
        """
        For string and categorical EAs, repeat(0) should return an empty
        EA of the same class.
        """
        arr = EA(make_data())

        result = arr.repeat(0)

        assert isinstance(result, EA)
        assert len(result) == 0
        # Logical values empty
        assert result.tolist() == []


class TestArkoudaArraySearchSorted:
    def test_searchsorted_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not define searchsorted; it should inherit the
        implementation from pandas ExtensionArray.
        """
        assert "searchsorted" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "searchsorted" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_searchsorted_scalar_left_matches_numpy(self):
        """
        searchsorted(scalar, side='left') should behave like numpy.searchsorted
        on the underlying numeric data.
        """
        data = np.array([10, 20, 30, 40], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(data))

        value = 25
        result = ak_arr.searchsorted(value, side="left")
        expected = np.searchsorted(data, value, side="left")

        assert result == expected

    def test_searchsorted_scalar_right_matches_numpy(self):
        """
        searchsorted(scalar, side='right') should behave like numpy.searchsorted
        with side='right'.
        """
        data = np.array([10, 20, 20, 30], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(data))

        value = 20
        result = ak_arr.searchsorted(value, side="right")
        expected = np.searchsorted(data, value, side="right")

        assert result == expected

    def test_searchsorted_array_like_matches_numpy(self):
        """
        searchsorted(array_like) should return a NumPy array of insertion
        positions matching numpy.searchsorted for each element.
        """
        data = np.array([0, 5, 10, 15], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(data))

        values = np.array([-1, 0, 3, 10, 20], dtype="int64")
        result = ak_arr.searchsorted(values, side="left")

        expected = np.searchsorted(data, values, side="left")

        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_searchsorted_empty_values(self):
        """Searchsorted on an empty 'values' array should return an empty array."""
        data = np.array([1, 2, 3], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(data))

        values = np.array([], dtype="int64")
        result = ak_arr.searchsorted(values)

        assert np.asarray(result).shape == (0,)

    def test_searchsorted_empty_array_base(self):
        """
        If the base array is empty, searchsorted always returns zero (or an array
        of zeros).
        """
        data = np.array([], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(data))

        # Scalar
        assert ak_arr.searchsorted(5) == 0

        # Array-like
        values = np.array([1, 2, 3])
        result = ak_arr.searchsorted(values)
        expected = np.zeros(len(values), dtype=np.intp)

        np.testing.assert_array_equal(np.asarray(result), expected)

    # ----------------------------------------------------------------------
    # Strings: searchsorted works and should match NumPy on logical values
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "make_data, value, side",
        [
            (lambda: ak.array(["a", "c", "e"]), "b", "left"),
            (lambda: ak.array(["a", "c", "e"]), "c", "right"),
        ],
    )
    def test_searchsorted_scalar_strings_matches_numpy(self, make_data, value, side):
        """
        For ArkoudaStringArray, searchsorted(scalar) should agree with
        numpy.searchsorted on the logical values (as object dtype).
        """
        arr = ArkoudaStringArray(make_data())
        baseline = np.asarray(arr.tolist(), dtype=object)

        result = arr.searchsorted(value, side=side)
        expected = np.searchsorted(baseline, value, side=side)

        assert result == expected

    @pytest.mark.parametrize(
        "make_data, values",
        [
            (
                lambda: ak.array(["a", "c", "e"]),
                ["a", "b", "e", "z"],
            ),
        ],
    )
    def test_searchsorted_array_like_strings_matches_numpy(self, make_data, values):
        """
        For ArkoudaStringArray, searchsorted(array_like) should match
        numpy.searchsorted on the logical values.
        """
        arr = ArkoudaStringArray(make_data())
        baseline = np.asarray(arr.tolist(), dtype=object)
        values_arr = np.asarray(values, dtype=object)

        result = arr.searchsorted(values_arr, side="left")
        expected = np.searchsorted(baseline, values_arr, side="left")

        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_searchsorted_empty_array_base_strings_currently_raises(self):
        """
        For an empty ArkoudaStringArray, searchsorted currently fails inside
        Strings.to_ndarray(), because the expected byte count for offsets
        is 0 but the reply has nonzero length.

        We lock that in as the current behavior so future fixes can update
        this test.
        """
        arr = ArkoudaStringArray(ak.array([], dtype=ak.str_))

        with pytest.raises(RuntimeError):
            arr.searchsorted("anything")

        with pytest.raises(RuntimeError):
            arr.searchsorted(np.array(["a", "b"], dtype=object))


class TestArkoudaArrayShift:
    def test_shift_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should *not* override shift; it should inherit the
        implementation from pandas.core.arrays.base.ExtensionArray.

        If this ever starts failing, it likely means ArkoudaArray has gained
        its own shift implementation and these tests should be revisited.
        """
        assert "shift" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "shift" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_shift_positive_periods_default_fill(self):
        """
        For positive periods, ArkoudaArray.shift should move values "down"
        and fill the leading positions with -1 by default (current Arkouda
        behavior for int64-backed arrays).
        """
        values = np.array([0, 1, 2, 3, 4], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.shift(periods=1)
        result_arr = np.asarray(result)
        expected = np.array([-1, 0, 1, 2, 3], dtype="int64")
        np.testing.assert_array_equal(result_arr, expected)

        result2 = ak_arr.shift(periods=2)
        result2_arr = np.asarray(result2)
        expected2 = np.array([-1, -1, 0, 1, 2], dtype="int64")
        np.testing.assert_array_equal(result2_arr, expected2)

    def test_shift_negative_periods_default_fill(self):
        """
        For negative periods, ArkoudaArray.shift should move values "up"
        and fill the trailing positions with -1 by default.
        """
        values = np.array([10, 20, 30, 40, 50], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.shift(periods=-1)
        result_arr = np.asarray(result)
        expected = np.array([20, 30, 40, 50, -1], dtype="int64")
        np.testing.assert_array_equal(result_arr, expected)

        result2 = ak_arr.shift(periods=-2)
        result2_arr = np.asarray(result2)
        expected2 = np.array([30, 40, 50, -1, -1], dtype="int64")
        np.testing.assert_array_equal(result2_arr, expected2)

    def test_shift_with_explicit_fill_value(self):
        """
        When a fill_value is provided, ArkoudaArray.shift should use that
        value instead of the default -1 in the newly created positions.
        """
        values = np.array([1, 2, 3], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.shift(periods=1, fill_value=99)
        result_arr = np.asarray(result)
        expected = np.array([99, 1, 2], dtype="int64")
        np.testing.assert_array_equal(result_arr, expected)

        result_neg = ak_arr.shift(periods=-1, fill_value=99)
        result_neg_arr = np.asarray(result_neg)
        expected_neg = np.array([2, 3, 99], dtype="int64")
        np.testing.assert_array_equal(result_neg_arr, expected_neg)

    def test_shift_large_periods_all_fill(self):
        """
        If abs(periods) >= len(arr), the result should be entirely filled
        with the fill_value (or -1 by default).
        """
        values = np.array([5, 6, 7], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        # Default fill_value = -1
        result = ak_arr.shift(periods=10)
        result_arr = np.asarray(result)
        expected = np.array([-1, -1, -1], dtype="int64")
        np.testing.assert_array_equal(result_arr, expected)

        # Explicit fill_value
        result2 = ak_arr.shift(periods=-10, fill_value=0)
        result2_arr = np.asarray(result2)
        expected2 = np.array([0, 0, 0], dtype="int64")
        np.testing.assert_array_equal(result2_arr, expected2)

    def test_shift_zero_periods_is_noop_on_values(self):
        """
        periods=0 should be a no-op on the values (even if the object identity
        may or may not be preserved).
        """
        values = np.array([1, 2, 3], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(values))

        result = ak_arr.shift(periods=0)
        result_arr = np.asarray(result)

        np.testing.assert_array_equal(result_arr, values)

    # ------------------------------------------------------------------
    # Strings & Categoricals
    # ------------------------------------------------------------------

    def test_shift_strings_with_fill_value_matches_pandas(self):
        """
        For ArkoudaStringArray, shift with an explicit fill_value should
        match pandas' string ExtensionArray.shift on the logical values.
        """
        values = ["a", "b", "c", "d", "e"]
        ak_data = ak.array(values)
        arr = ArkoudaStringArray(ak_data)

        result = arr.shift(periods=1, fill_value="ZZ")
        result_list = (
            arr.__class__(result)._data.to_ndarray().tolist()
            if hasattr(result, "_data")
            else result.tolist()
        )

        # pandas reference - use a string EA, not plain object
        pd_ea = pd.array(values, dtype="string[python]")
        expected = pd_ea.shift(periods=1, fill_value="ZZ")
        expected_list = list(expected.astype(object))

        assert result_list == expected_list

    def test_shift_strings_zero_periods_is_noop(self):
        """For ArkoudaStringArray, periods=0 should leave logical values unchanged."""
        values = ["a", "b", "c"]
        arr = ArkoudaStringArray(ak.array(values))

        result = arr.shift(periods=0)
        assert result.tolist() == values

    def test_shift_categorical_currently_raises_valueerror(self):
        """
        For ArkoudaCategoricalArray, shift() currently fails when pandas'
        machinery tries to construct a categorical from the fill row
        (e.g. [-1]). We lock in the ValueError as the current behavior.

        If/when shift is properly implemented for ArkoudaCategoricalArray,
        this test should be updated to assert the correct semantics instead.
        """
        cat = ak.Categorical(ak.array(["x", "y", "z", "y", "x"]))
        arr = ArkoudaCategoricalArray(cat)

        with pytest.raises(ValueError):
            arr.shift(periods=1)


class TestArkoudaArrayToList:
    def test_tolist_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not override tolist(); it should inherit the
        implementation from pandas ExtensionArray.
        """
        assert "tolist" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "tolist" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_tolist_basic_int(self):
        """
        tolist() on basic integer data should return a list whose values
        match the underlying numeric data.
        """
        data = np.array([1, 2, 3], dtype="int64")
        ak_arr = ArkoudaArray(ak.array(data))

        result = ak_arr.tolist()

        assert isinstance(result, list)
        # Don't over-specify scalar types (may be numpy.int64); just check values.
        assert result == data.tolist()

    def test_tolist_basic_float(self):
        """
        Floats should be returned as a list of numeric values matching
        the underlying data.
        """
        data = np.array([1.5, 2.25, -3.75], dtype="float64")
        ak_arr = ArkoudaArray(ak.array(data))

        result = ak_arr.tolist()

        assert isinstance(result, list)
        assert result == data.tolist()

    def test_tolist_with_nans_matches_pandas(self):
        """
        Behavior with NaNs should match pandas' ExtensionArray.tolist()
        elementwise, treating both NaN and pd.NA as "missing".
        """
        data = np.array([1.0, np.nan, 3.5], dtype="float64")

        pd_ea = pd.array(data)  # nullable EA
        expected = pd_ea.tolist()

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.tolist()

        assert isinstance(result, list)
        assert len(result) == len(expected)

        for got, exp in zip(result, expected):
            if pd.isna(exp):
                # treat any missing (NaN, None, pd.NA) on the Arkouda side as matching
                assert pd.isna(got)
            else:
                assert got == exp

    def test_tolist_empty(self):
        """Empty ArkoudaArray should return an empty Python list."""
        ak_arr = ArkoudaArray(ak.array(np.array([], dtype="int64")))

        result = ak_arr.tolist()

        assert isinstance(result, list)
        assert result == []

    def test_tolist_matches_pandas_extension_array(self):
        """
        General check: ArkoudaArray.tolist() should match the behavior of
        pandas ExtensionArray.tolist() on the same values, including NaNs,
        comparing elementwise with pd.isna for missing values.
        """
        values = np.array([5, np.nan, 7, 9], dtype="float64")

        pd_ea = pd.array(values)
        expected = pd_ea.tolist()

        ak_arr = ArkoudaArray(ak.array(values))
        result = ak_arr.tolist()

        assert isinstance(result, list)
        assert len(result) == len(expected)

        for got, exp in zip(result, expected):
            if pd.isna(exp):
                assert pd.isna(got)
            else:
                assert got == exp

    # ------------------------------------------------------------------
    # Strings & Categoricals
    # ------------------------------------------------------------------

    def test_tolist_strings_roundtrip(self):
        """
        ArkoudaStringArray.tolist() should round-trip the logical string
        values, consistent with pandas' string ExtensionArray.tolist().
        """
        values = ["a", "b", "c"]
        ak_data = ak.array(values)
        arr = ArkoudaStringArray(ak_data)

        result = arr.tolist()

        assert isinstance(result, list)

        # pandas reference
        expected = pd.array(values, dtype="string[python]").tolist()
        assert result == expected

    def test_tolist_categorical_roundtrip(self):
        """
        ArkoudaCategoricalArray.tolist() should return the logical category
        labels, consistent with pandas.Categorical.tolist().
        """
        labels = ["x", "y", "x", "z"]
        ak_cat = ak.Categorical(ak.array(labels))
        arr = ArkoudaCategoricalArray(ak_cat)

        result = arr.tolist()

        assert isinstance(result, list)

        # pandas reference
        expected = list(pd.Categorical(labels))
        assert result == expected


class TestArkoudaArrayTranspose:
    def test_transpose_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not override transpose; it should inherit the
        implementation from pandas.core.arrays.base.ExtensionArray.
        """
        assert "transpose" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "transpose" in base.__dict__)
        assert owner is PandasExtensionArray

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaArray, lambda: ak.arange(5)),
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d", "e"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y", "x"])),
            ),
        ],
    )
    def test_transpose_preserves_type_and_values_for_1d_array(self, EA, make_data):
        """
        transpose() on a 1D Arkouda-backed EA should return the same EA type
        with the same logical values in the same order.
        """
        a = EA(make_data())
        result = a.transpose()

        assert isinstance(result, EA)
        # Use tolist to compare logical values across all EA types
        assert result.tolist() == a.tolist()

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaArray, lambda: ak.arange(5)),
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d", "e"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y", "x"])),
            ),
        ],
    )
    def test_T_property_equivalent_to_transpose(self, EA, make_data):
        """
        The .T property should be equivalent to calling transpose() for all
        Arkouda-backed EAs.
        """
        a = EA(make_data())

        result_T = a.T
        result_trans = a.transpose()

        assert isinstance(result_T, EA)
        assert isinstance(result_trans, EA)
        assert result_T.tolist() == result_trans.tolist()

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaArray, lambda: ak.array([10, 20, 30, 40])),
            (ArkoudaStringArray, lambda: ak.array(["u", "v", "w", "x"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["a", "b", "a", "c"])),
            ),
        ],
    )
    def test_transpose_does_not_modify_values(self, EA, make_data):
        """transpose() must not change the underlying logical data for 1D arrays."""
        a = EA(make_data())
        before = a.tolist()

        result = a.transpose()

        assert result.tolist() == before

    @pytest.mark.parametrize(
        "EA, make_data",
        [
            (ArkoudaArray, lambda: ak.arange(4)),
            (ArkoudaStringArray, lambda: ak.array(["a", "b", "c", "d"])),
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "z", "y"])),
            ),
        ],
    )
    def test_transpose_accepts_unused_axes_argument(self, EA, make_data):
        """
        Some callers pass axes even for 1D arrays; transpose() should ignore
        them and still preserve the values.
        """
        a = EA(make_data())

        result = a.transpose(0)
        result2 = a.transpose((0,))

        assert result.tolist() == a.tolist()
        assert result2.tolist() == a.tolist()


class TestArkoudaArrayUnique:
    def test_unique_is_inherited_from_pandas_extension_array(self):
        """
        ArkoudaArray should not override unique; it should inherit the
        implementation from pandas.core.arrays.base.ExtensionArray.
        """
        assert "unique" not in ArkoudaArray.__dict__

        owner = next(base for base in ArkoudaArray.mro() if "unique" in base.__dict__)
        assert owner is PandasExtensionArray

    def test_unique_removes_duplicates_and_preserves_order(self):
        """
        unique() on ArkoudaArray should remove duplicates while preserving
        the order of first occurrence, matching pandas EA semantics.
        """
        data = np.array([1, 2, 2, 3, 1], dtype="int64")

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.unique()
        result_arr = np.asarray(result)

        # pandas reference via nullable EA
        pd_ea = pd.array(data)
        expected = pd_ea.unique()
        expected_arr = np.asarray(expected)

        np.testing.assert_array_equal(result_arr, expected_arr)

    def test_unique_all_duplicates(self):
        """If all elements are the same, unique() should return a length-1 array."""
        data = np.array([7, 7, 7, 7], dtype="int64")

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.unique()
        result_arr = np.asarray(result)

        assert result_arr.shape == (1,)
        assert result_arr[0] == 7

    def test_unique_already_unique(self):
        """
        If the data are already unique, unique() should return the same
        values in the same order.
        """
        data = np.array([10, 20, 30, 40], dtype="int64")

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.unique()
        result_arr = np.asarray(result)

        np.testing.assert_array_equal(result_arr, data)

    def test_unique_with_nans_matches_pandas_extension_array(self):
        """
        unique() on data containing NaNs should behave like pandas'
        ExtensionArray.unique(), including how missing values are handled.
        """
        data = np.array([1.0, np.nan, 1.0, np.nan, 2.0], dtype="float64")

        # pandas reference via nullable EA
        pd_ea = pd.array(data)
        expected = pd_ea.unique()
        expected_list = expected.tolist()

        ak_arr = ArkoudaArray(ak.array(data))
        result = ak_arr.unique()
        # result is an ExtensionArray; use tolist() for robust comparison
        result_list = result.tolist()

        assert len(result_list) == len(expected_list)

        for got, exp in zip(result_list, expected_list):
            if pd.isna(exp):
                # treat NaN / None / pd.NA all as "missing"
                assert pd.isna(got)
            else:
                assert got == exp

    # ------------------------------------------------------------------
    # String and Categorical examples
    # ------------------------------------------------------------------

    @pytest.mark.xfail(reason=("Fails because ArkoudaCategoricalArray.astype() is not yet implemented."))
    @pytest.mark.parametrize(
        "EA, make_data, pandas_constructor",
        [
            # Strings: order-preserving de-duplication
            (
                ArkoudaStringArray,
                lambda: ak.array(["a", "b", "a", "c", "b"]),
                lambda vals: pd.array(vals, dtype="string"),
            ),
            # Categoricals: order-preserving de-duplication on labels
            (
                ArkoudaCategoricalArray,
                lambda: ak.Categorical(ak.array(["x", "y", "x", "z", "y"])),
                lambda vals: pd.Categorical(vals),
            ),
        ],
    )
    def test_unique_strings_and_categoricals_match_pandas(self, EA, make_data, pandas_constructor):
        """
        For ArkoudaStringArray and ArkoudaCategoricalArray, unique() should
        remove duplicates while preserving the order of first occurrence,
        matching pandas' unique() on the logical values.
        """
        # Arkouda-backed EA
        ak_data = make_data()
        arr = EA(ak_data)

        result = arr.unique()
        result_list = result.tolist()

        # pandas baseline on the logical Python-level values
        logical_values = arr.tolist()
        pd_ea = pandas_constructor(logical_values)
        expected = pd_ea.unique()
        expected_list = expected.tolist()

        assert result_list == expected_list

    @pytest.mark.xfail(reason=("Fails because ArkoudaCategoricalArray.astype() is not yet implemented."))
    @pytest.mark.parametrize(
        "EA, values",
        [
            (ArkoudaStringArray, ["foo", "foo", "foo"]),
            (ArkoudaCategoricalArray, ["cat", "cat", "cat"]),
        ],
    )
    def test_unique_strings_and_categoricals_all_duplicates(self, EA, values):
        """
        For string and categorical Arkouda EAs, if all elements are the same,
        unique() should return a length-1 array with that value.
        """
        if EA is ArkoudaCategoricalArray:
            ak_data = ak.Categorical(ak.array(values))
        else:
            ak_data = ak.array(values)

        arr = EA(ak_data)
        result = arr.unique()
        result_list = result.tolist()

        assert result_list == [values[0]]

    @pytest.mark.xfail(reason=("Fails because ArkoudaCategoricalArray.astype() is not yet implemented."))
    @pytest.mark.parametrize(
        "EA, values",
        [
            (ArkoudaStringArray, ["a", "b", "c", "d"]),
            (ArkoudaCategoricalArray, ["x", "y", "z"]),
        ],
    )
    def test_unique_strings_and_categoricals_already_unique(self, EA, values):
        """
        For string and categorical Arkouda EAs, if the data are already
        unique, unique() should return the same logical values in the
        same order.
        """
        if EA is ArkoudaCategoricalArray:
            ak_data = ak.Categorical(ak.array(values))
        else:
            ak_data = ak.array(values)

        arr = EA(ak_data)
        result = arr.unique()
        result_list = result.tolist()

        assert result_list == values
