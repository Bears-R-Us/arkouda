import numpy as np
import pandas as pd
import pytest

from pandas import DataFrame as pd_DataFrame
from pandas import Series as pd_Series

from arkouda.numpy.pdarrayclass import all as ak_all
from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraycreation import arange as ak_arange
from arkouda.numpy.pdarraycreation import array as ak_array
from arkouda.numpy.strings import Strings
from arkouda.pandas.categorical import Categorical as ak_Categorical
from arkouda.pandas.dataframe import DataFrame as ak_DataFrame
from arkouda.pandas.extension import (
    ArkoudaArray,
    ArkoudaCategorical,
    ArkoudaExtensionArray,
    ArkoudaIndexAccessor,
    ArkoudaStringArray,
)
from arkouda.pandas.extension._dataframe_accessor import (
    _akdf_to_pandas_no_copy,
    _df_to_akdf_no_copy,
    _extract_ak_from_ea,
    _is_arkouda_series,
    _looks_like_ak_col,
    _series_to_akcol_no_copy,
)


class TestDataFrameAccessorInternals:
    def test_dataframe_extension_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _dataframe_accessor

        result = doctest.testmod(
            _dataframe_accessor, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_looks_like_ak_col_recognizes_core_types(self):
        ak_ints = ak_arange(5)
        ak_strings = ak_array(["a", "b", "c"])
        ak_cat = ak_Categorical(ak_array(["low", "low", "high"]))

        assert _looks_like_ak_col(ak_ints)
        assert _looks_like_ak_col(ak_strings)
        assert _looks_like_ak_col(ak_cat)

        assert not _looks_like_ak_col([1, 2, 3])
        assert not _looks_like_ak_col(np.arange(3))

    def test_extract_ak_from_ea_success_and_failure(self):
        ak_ints = ak_arange(3)
        ea = ArkoudaArray(ak_ints)

        col = _extract_ak_from_ea(ea)
        assert _looks_like_ak_col(col)
        assert ak_all(col == ak_ints)

        with pytest.raises(TypeError):
            _extract_ak_from_ea(object())

    def test_is_arkouda_series(self):
        s_ak = pd_Series(ArkoudaArray(ak_arange(3)))
        s_plain = pd_Series([1, 2, 3])

        assert _is_arkouda_series(s_ak)
        assert not _is_arkouda_series(s_plain)

    def test_series_to_akcol_no_copy_success_and_failure(self):
        ak_ints = ak_arange(4)
        s_ak = pd_Series(ArkoudaArray(ak_ints))

        col = _series_to_akcol_no_copy(s_ak)
        assert _looks_like_ak_col(col)
        assert ak_all(col == ak_ints)

        with pytest.raises(TypeError):
            _series_to_akcol_no_copy(pd_Series([1, 2, 3]))


class TestDataFrameConversion:
    def _make_arkouda_df(self) -> pd_DataFrame:
        """Helper: pandas DataFrame with Arkouda-backed EAs."""
        return pd_DataFrame(
            {
                "i": ArkoudaArray(ak_arange(5)),
                "s": ArkoudaStringArray(ak_array(["a", "b", "c", "d", "e"])),
                "c": ArkoudaCategorical(
                    ak_Categorical(ak_array(["low", "low", "high", "medium", "low"]))
                ),
            }
        )

    def test_df_to_akdf_no_copy_and_back_roundtrip(self):
        df = self._make_arkouda_df()

        akdf = _df_to_akdf_no_copy(df)
        assert isinstance(akdf, ak_DataFrame)
        assert set(akdf.columns) == {"i", "s", "c"}
        assert isinstance(akdf["c"], ak_Categorical)

        df2 = _akdf_to_pandas_no_copy(akdf)
        assert isinstance(df2, pd_DataFrame)
        assert list(df2.columns) == ["i", "s", "c"]
        assert all(_is_arkouda_series(df2[col]) for col in df2.columns)

        # Value equality (use to_list() to avoid EA internals)
        assert df2["i"].tolist() == list(range(5))
        assert df2["s"].tolist() == ["a", "b", "c", "d", "e"]

    def test_to_ak_legacy_converts_plain_pandas_df(self):
        df = pd_DataFrame(
            {
                "i": [1, 2, 3],
                "s": ["a", "b", "c"],
                "c": pd_Series(["low", "low", "high"], dtype="category"),
            }
        )

        akdf = df.ak.to_ak_legacy()
        assert isinstance(akdf, ak_DataFrame)
        assert set(akdf.columns) == {"i", "s", "c"}

        assert isinstance(akdf["i"], pdarray)
        assert isinstance(akdf["s"], Strings)
        assert isinstance(akdf["c"], ak_Categorical)

        assert ak_all(akdf["i"] == ak_array([1, 2, 3]))
        assert akdf["c"].categories.tolist() == ["high", "low", "N/A"] or akdf[
            "c"
        ].categories.tolist() == ["low", "high", "N/A"]

    def test_from_ak_legacy_produces_arkouda_backed_pandas_df(self):
        akdf = ak_DataFrame(
            {
                "i": ak_arange(3),
                "s": ak_array(["x", "y", "z"]),
                "c": ak_Categorical(ak_array(["low", "high", "low"])),
            }
        )

        # Accessor is bound to any DataFrame; only akdf matters here.
        df = pd.DataFrame.ak.from_ak_legacy(akdf)

        assert list(df.columns) == ["i", "s", "c"]
        assert all(_is_arkouda_series(df[col]) for col in df.columns)
        assert df["i"].tolist() == [0, 1, 2]
        assert df["s"].tolist() == ["x", "y", "z"]

    def test_to_ak_creates_arkouda_backed_dataframe(self):
        df = pd_DataFrame({"x": [10, 20, 30], "y": [1.0, 2.0, 3.0]})

        df_ak = df.ak.to_ak()
        assert isinstance(df_ak, pd_DataFrame)
        assert all(_is_arkouda_series(df_ak[col]) for col in df_ak.columns)
        assert df_ak["x"].tolist() == [10, 20, 30]

    def test_collect_converts_akdf_to_numpy_backed_df(self):
        df = pd_DataFrame({"x": [1, 2, 3]})
        akdf = df.ak.to_ak()

        out = akdf.ak.collect()
        assert isinstance(out, pd_DataFrame)
        assert out["x"].tolist() == [1, 2, 3]
        assert isinstance(out["x"].values, np.ndarray)


class TestAccessorValidationAndMerge:
    def test_assert_all_arkouda_raises_on_non_arkouda_column(self):
        df = pd_DataFrame({"x": [1, 2, 3]})
        with pytest.raises(TypeError, match="must be Arkouda ExtensionArrays"):
            df.ak._assert_all_arkouda(df, "left")

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_ak_merge_matches_basic_expectations(self, how):
        df1 = pd_DataFrame(
            {
                "id": ArkoudaArray(ak_array([1, 2, 3])),
                "name": ArkoudaStringArray(ak_array(["alice", "bob", "carol"])),
            }
        )

        df2 = pd_DataFrame(
            {
                "id": ArkoudaArray(ak_array([2, 3, 4])),
                "score": ArkoudaArray(ak_array([88, 92, 75])),
            }
        )

        result = df1.ak.merge(df2, on="id", how=how)

        assert all(_is_arkouda_series(result[col]) for col in result.columns)

        ids = result["id"].tolist()
        if how == "inner":
            assert ids == [2, 3]
            assert result["score"].tolist() == [88, 92]
        elif how == "left":
            assert ids == [1, 2, 3]
        elif how == "right":
            assert ids == [2, 3, 4]
        elif how == "outer":
            assert sorted(ids) == [1, 2, 3, 4]

    def test_ak_merge_raises_if_either_side_not_arkouda_backed(self):
        df_left = pd_DataFrame({"id": [1, 2, 3]})
        df_right = pd_DataFrame({"id": ArkoudaArray(ak_array([1, 2, 3]))})

        with pytest.raises(TypeError):
            df_left.ak.merge(df_right, on="id")

        df_left2 = pd_DataFrame({"id": ArkoudaArray(ak_array([1, 2, 3]))})
        df_right2 = pd_DataFrame({"id": [1, 2, 3]})

        with pytest.raises(TypeError):
            df_left2.ak.merge(df_right2, on="id")

    def test_dataframe_to_ak_converts_index_to_arkouda_backed(self):
        # plain pandas DF (RangeIndex)
        df = pd.DataFrame({"a": np.arange(5), "b": np.arange(5)})

        # convert to Arkouda-backed pandas DF
        ak_df = df.ak.to_ak()

        # Sanity: columns are Arkouda-backed
        assert isinstance(ak_df["a"].array, ArkoudaExtensionArray)
        assert isinstance(ak_df["b"].array, ArkoudaExtensionArray)

        assert ArkoudaIndexAccessor(ak_df.index).is_arkouda, (
            f"Expected Arkouda-backed index, got {type(ak_df.index).__name__}: {ak_df.index!r}"
        )
        assert isinstance(ak_df.index.array, ArkoudaExtensionArray)

        assert np.array_equal(ak_df["a"].array.to_numpy(), np.arange(5))
        assert np.array_equal(ak_df["b"].array.to_numpy(), np.arange(5))
