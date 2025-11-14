import pytest

import arkouda as ak

from arkouda import Categorical, DataFrame, Index, MultiIndex, Series, cast
from arkouda.testing import (
    assert_almost_equal,
    assert_almost_equivalent,
    assert_arkouda_array_equal,
    assert_arkouda_array_equivalent,
    assert_arkouda_segarray_equal,
    assert_arkouda_strings_equal,
    assert_attr_equal,
    assert_categorical_equal,
    assert_class_equal,
    assert_contains_all,
    assert_copy,
    assert_dict_equal,
    assert_equal,
    assert_equivalent,
    assert_frame_equal,
    assert_frame_equivalent,
    assert_index_equal,
    assert_index_equivalent,
    assert_is_sorted,
    assert_series_equal,
    assert_series_equivalent,
)


class TestAsserters:
    def test_asserters_docstrings(self):
        import doctest

        from arkouda.testing import _asserters

        result = doctest.testmod(_asserters, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @staticmethod
    def build_index(self) -> Index:
        idx = ak.Index(ak.arange(5), name="test1")
        return idx

    @staticmethod
    def build_multi_index(self) -> MultiIndex:
        midx = ak.MultiIndex([ak.arange(5), -1 * ak.arange(5)], names=["test1", "test2"])
        return midx

    @staticmethod
    def build_ak_df(self, index_dtype="int64", index_name=None) -> DataFrame:
        username = ak.array(
            [
                "Alice",
                "Alice",
                "Alice",
                "Bob",
                "Bob",
                "Carol",
            ]
        )
        userid = ak.array([111, 222, 111, 333, 222, 111])
        item = ak.array([0, 0, 1, 1, 2, 0])
        day = ak.array([5, 5, 6, 5, 6, 6])
        amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
        bi = ak.arange(2**200, 2**200 + 6)
        return ak.DataFrame(
            {
                "userName": username,
                "userID": userid,
                "item": item,
                "day": day,
                "amount": amount,
                "bi": bi,
            },
            index=Index(ak.arange(6, dtype=index_dtype), name=index_name),
        )

    @staticmethod
    def perturb(a: ak.pdarray, atol: float, rtol: float, rng=None):
        if rng is None:
            rng = ak.random.default_rng()
        return a + rtol * a + atol * rng.random()

    @staticmethod
    def convert(obj, as_arkouda: bool):
        if not isinstance(
            obj,
            (
                ak.DataFrame,
                ak.Series,
                ak.Index,
                ak.MultiIndex,
                ak.pdarray,
                ak.Categorical,
                ak.Strings,
                ak.SegArray,
            ),
        ):
            raise TypeError("obj must be an arkouda object.")

        if as_arkouda:
            return obj
        elif isinstance(obj, (ak.pdarray, ak.Strings)):
            return obj.to_ndarray()
        elif isinstance(obj, (ak.DataFrame)):
            return obj.to_pandas(retain_index=True)
        elif isinstance(obj, (ak.Series, ak.Index, ak.SegArray, ak.Categorical)):
            return obj.to_pandas()
        return None

    def get_converter(self, as_arkouda: bool):
        def converter(obj):
            return self.convert(obj, as_arkouda)

        return converter

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_almost_equal(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        rng = ak.random.default_rng()
        atol = 0.001
        rtol = 0.001
        a = ak.arange(size, dtype="float64")
        a2 = self.perturb(a, atol=atol, rtol=rtol, rng=rng)
        a3 = a + rtol + atol

        if both_ak:
            assert_almost_equal(a, a2, atol=atol, rtol=rtol)
        assert_almost_equivalent(convert_left(a), convert_right(a2), atol=atol, rtol=rtol)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_almost_equal(a, a3, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_almost_equivalent(convert_left(a), convert_right(a3), atol=atol, rtol=rtol)

        idx = Index(a)
        idx2 = Index(a2)
        idx3 = Index(a3)

        if both_ak:
            assert_almost_equal(idx, idx2, atol=atol, rtol=rtol)
        assert_almost_equivalent(convert_left(idx), convert_right(idx2), atol=atol, rtol=rtol)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_almost_equal(idx, idx3, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_almost_equivalent(convert_left(idx), convert_right(idx3), atol=atol, rtol=rtol)

        s = Series(a)
        s2 = Series(a2)
        s3 = Series(a3)

        if both_ak:
            assert_almost_equal(s, s2, atol=atol, rtol=rtol)
        assert_almost_equivalent(convert_left(s), convert_right(s2), atol=atol, rtol=rtol)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_almost_equal(s, s3, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_almost_equivalent(convert_left(s), convert_right(s3), atol=atol, rtol=rtol)

        df = DataFrame({"col1": a}, index=idx)
        df2 = DataFrame({"col1": a2}, index=idx2)
        df3 = DataFrame({"col1": a3}, index=idx3)

        if both_ak:
            assert_almost_equal(df, df2, atol=atol, rtol=rtol)
        assert_almost_equivalent(convert_left(df), convert_right(df2), atol=atol, rtol=rtol)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_almost_equal(df, df3, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_almost_equivalent(convert_left(df), convert_right(df3), atol=atol, rtol=rtol)

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_almost_equal_multi_dim(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        shape = (2, 2, size)

        rng = ak.random.default_rng()
        atol = 0.001
        rtol = 0.001
        a = ak.arange(size * 4, dtype="float64")
        a2 = self.perturb(a, atol=atol, rtol=rtol, rng=rng)
        a3 = a + rtol + atol

        a = a.reshape(shape)
        a2 = a2.reshape(shape)
        a3 = a3.reshape(shape)

        if both_ak:
            assert_almost_equal(a, a2, atol=atol, rtol=rtol)
        assert_almost_equivalent(convert_left(a), convert_right(a2), atol=atol, rtol=rtol)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_almost_equal(a, a3, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_almost_equivalent(convert_left(a), convert_right(a3), atol=atol, rtol=rtol)

    def test_assert_almost_equal_scalars(self):
        atol = 0.001
        rtol = 0.001

        assert_almost_equal(True, True, atol=atol, rtol=rtol)
        assert_almost_equivalent(True, True, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_almost_equal(True, False, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_almost_equivalent(True, False, atol=atol, rtol=rtol)

        assert_almost_equal(1.0, 1.0, atol=atol, rtol=rtol)
        assert_almost_equivalent(1.0, 1.0, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_almost_equal(1.0, 1.5, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_almost_equivalent(1.0, 1.5, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_index_equal(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        # exact
        i1 = Index(ak.arange(size, dtype="float64"))
        i2 = Index(ak.arange(size, dtype="int64"))
        if both_ak:
            assert_index_equal(i1, i2, exact=False)
        assert_index_equivalent(convert_left(i1), convert_right(i2), exact=False)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_index_equal(i1, i2, exact=True)
        with pytest.raises(AssertionError):
            assert_index_equivalent(convert_left(i1), convert_right(i2), exact=True)

        # check_names
        i3 = Index(ak.arange(size), name="name1")
        i4 = Index(ak.arange(size), name="name1")
        i5 = Index(ak.arange(size), name="name2")

        if both_ak:
            assert_index_equal(i3, i4, check_names=True)
        assert_index_equivalent(convert_left(i3), convert_right(i4), check_names=True)

        if both_ak:
            assert_index_equal(i3, i5, check_names=False)
        assert_index_equivalent(convert_left(i3), convert_right(i5), check_names=False)

        if both_ak:
            with pytest.raises(AssertionError):
                assert_index_equal(i3, i5, check_names=True)
        with pytest.raises(AssertionError):
            assert_index_equivalent(convert_left(i3), convert_right(i5), check_names=True)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_index_equal_categorical(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        # check_categorical
        # check_order
        i1 = Index(Categorical(ak.array(["a", "a", "b"])))
        i3 = Index(Categorical(ak.array(["a", "b", "a"])))
        i4 = Index(Categorical(ak.array(["a", "b", "c"])))
        i5 = Index(Categorical(ak.array(["a", "a", "b"])).sort_values())

        if both_ak:
            assert_index_equal(i1, i1)
        assert_index_equivalent(convert_left(i1), convert_right(i1))

        if both_ak:
            assert_index_equal(i1, i3, check_order=False)
        assert_index_equivalent(convert_left(i1), convert_right(i3), check_order=False)

        if both_ak:
            with pytest.raises(AssertionError):
                assert_index_equal(i1, i3, check_order=True)
        with pytest.raises(AssertionError):
            assert_index_equivalent(convert_left(i1), convert_right(i3), check_order=True)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_index_equal(i1, i3, check_categorical=False)
        with pytest.raises(AssertionError):
            assert_index_equivalent(convert_left(i1), convert_right(i3), check_categorical=False)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_index_equal(i1, i4, check_categorical=False)
        with pytest.raises(AssertionError):
            assert_index_equivalent(convert_left(i1), convert_right(i4), check_categorical=False)
        if both_ak:
            assert_index_equal(i1, i5, check_order=True, check_categorical=True)
        assert_index_equivalent(
            convert_left(i1), convert_right(i5), check_order=True, check_categorical=True
        )

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_index_equal_check_exact(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        # check_exact
        i1 = Index(ak.arange(size, dtype="float64"))
        i2 = Index(ak.arange(size) + 1e-9)
        if both_ak:
            assert_index_equal(i1, i2, check_exact=False)
        assert_index_equivalent(convert_left(i1), convert_right(i2), check_exact=False)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_index_equal(i1, i2, check_exact=True)
        with pytest.raises(AssertionError):
            assert_index_equivalent(convert_left(i1), convert_right(i2), check_exact=True)

        # rtol
        # atol
        i3_float = Index(ak.arange(size, dtype="float64"))

        rng = ak.random.default_rng()
        atol = 0.001
        rtol = 0.001

        i3_atol = Index(ak.arange(size) + atol * rng.random())
        if both_ak:
            assert_index_equal(i3_float, i3_atol, check_exact=False, atol=atol)
        assert_index_equivalent(
            convert_left(i3_float), convert_right(i3_atol), check_exact=False, atol=atol
        )

        i3_atol_rtol = Index(ak.arange(size) + rtol * ak.arange(size) + atol * rng.random())
        if both_ak:
            assert_index_equal(i3_float, i3_atol_rtol, check_exact=False, atol=atol, rtol=rtol)
        assert_index_equivalent(
            convert_left(i3_float), convert_right(i3_atol_rtol), check_exact=False, atol=atol, rtol=rtol
        )

        i3_2rtol = Index(ak.arange(size) + ak.arange(size) * 2 * rtol)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_index_equal(i3_float, i3_2rtol, check_exact=False, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_index_equivalent(
                convert_left(i3_float), convert_right(i3_2rtol), check_exact=False, rtol=rtol
            )

        i3_2atol = Index(ak.arange(size) + 2 * atol)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_index_equal(i3_float, i3_2atol, check_exact=False, atol=atol)
        with pytest.raises(AssertionError):
            assert_index_equivalent(
                convert_left(i3_float), convert_right(i3_2atol), check_exact=False, atol=atol
            )

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_index_equal_multiindex(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        m1 = self.build_multi_index(self)
        m2 = self.build_multi_index(self)

        if both_ak:
            assert_index_equal(m1, m2)
        assert_index_equivalent(convert_left(m1), convert_right(m2))

    def test_assert_attr_equal_index(self):
        idx = self.build_index(self)
        idx2 = self.build_index(self)

        assert_attr_equal("name", idx, idx2, obj="Index")
        assert_attr_equal("names", idx, idx2, obj="Index")
        assert_attr_equal("max_list_size", idx, idx2, obj="Index")

        idx2.name = "test2"
        with pytest.raises(AssertionError):
            assert_attr_equal("name", idx, idx2, obj="Index")
        with pytest.raises(AssertionError):
            assert_attr_equal("names", idx, idx2, obj="Index")

    def test_assert_attr_equal_multiindex(self):
        idx = self.build_index(self)
        midx = self.build_multi_index(self)
        midx2 = self.build_multi_index(self)

        assert_attr_equal("names", midx, midx2, obj="MultiIndex")

        midx3 = ak.MultiIndex([ak.arange(5), -1 * ak.arange(5)], names=["test1", "test3"])
        with pytest.raises(AssertionError):
            assert_attr_equal("names", midx, midx3, obj="Index")
        with pytest.raises(AssertionError):
            assert_attr_equal("names", idx, midx, obj="Index")

        assert_attr_equal("nlevels", midx, midx2, obj="MultiIndex")

    def test_assert_class_equal(self):
        idx = self.build_index(self)
        midx = self.build_multi_index(self)
        midx2 = self.build_multi_index(self)
        df = self.build_ak_df(self)
        s = ak.Series(-1 * ak.arange(5), index=ak.arange(5))

        assert_class_equal(idx, idx)
        assert_class_equal(midx, midx2)
        assert_class_equal(s, s)
        assert_class_equal(df, df)
        with pytest.raises(AssertionError):
            assert_class_equal(midx, idx)
        with pytest.raises(AssertionError):
            assert_class_equal(s, idx)
        with pytest.raises(AssertionError):
            assert_class_equal(df, s)

    def test_assert_arkouda_strings_equal(self):
        a = ak.array(["a", "a", "b", "c"])
        a2 = ak.array(["a", "d", "b", "c"])
        a3 = ak.array(["a", "a", "b", "c", "d"])

        assert_arkouda_strings_equal(a, a)
        assert_arkouda_strings_equal(a, a, index_values=ak.arange(4))
        with pytest.raises(AssertionError):
            assert_arkouda_strings_equal(a, a2)
        with pytest.raises(AssertionError):
            assert_arkouda_strings_equal(a, a3)

        #   check_same
        a_copy = a[:]
        assert_arkouda_strings_equal(a, a_copy)

        assert_arkouda_strings_equal(a, a, check_same="same")
        with pytest.raises(AssertionError):
            assert_arkouda_strings_equal(a, a, check_same="copy")

        assert_arkouda_strings_equal(a, a_copy, check_same="copy")
        with pytest.raises(AssertionError):
            assert_arkouda_strings_equal(a, a_copy, check_same="same")

    def test_assert_dict_equal(self):
        size = 10
        dict1 = {"a": ak.arange(size), "b": -1 * ak.arange(size)}
        dict2 = {"a": ak.arange(size), "b": -1 * ak.arange(size)}
        dict3 = {"a": ak.arange(size), "c": -2 * ak.arange(size)}
        dict4 = {"a": ak.arange(size), "b": -1 * ak.arange(size), "c": -2 * ak.arange(size)}
        dict5 = {"a": ak.arange(size), "b": -2 * ak.arange(size)}

        assert_dict_equal(dict1, dict2)

        for d in [dict3, dict4, dict5]:
            with pytest.raises(AssertionError):
                assert_dict_equal(dict1, d)

    def test_assert_is_sorted(self):
        size = 10
        a = ak.arange(size)
        b = -1 * a
        c = ak.array([1, 2, 5, 4, 3])

        assert_is_sorted(a)
        with pytest.raises(AssertionError):
            assert_is_sorted(b)
        with pytest.raises(AssertionError):
            assert_is_sorted(c)

        idx_a = Index(a)
        idx_b = Index(b)
        idx_c = Index(c)

        assert_is_sorted(idx_a)
        with pytest.raises(AssertionError):
            assert_is_sorted(idx_b)
        with pytest.raises(AssertionError):
            assert_is_sorted(idx_c)

        series_a = Series(a)
        series_b = Series(b)
        series_c = Series(c)

        assert_is_sorted(series_a)
        with pytest.raises(AssertionError):
            assert_is_sorted(series_b)
        with pytest.raises(AssertionError):
            assert_is_sorted(series_c)

    def test_assert_categorical_equal(self):
        c1 = Categorical(
            ak.array(
                [
                    "Alice",
                    "Alice",
                    "Alice",
                    "Bob",
                    "Bob",
                    "Carol",
                ]
            )
        )
        c2 = Categorical(ak.array(["Alice", "Bob", "Alice", "Carol", "Bob", "Alice"])).sort_values()
        assert_categorical_equal(c1, c2, check_category_order=False)
        with pytest.raises(AssertionError):
            assert_categorical_equal(c1, c2, check_category_order=True)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_series_equal_check_names(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        s = Series(ak.array(["a", "b", "c"]), index=Index(ak.arange(3)), name="test")
        if both_ak:
            assert_series_equal(s, s)
        assert_series_equivalent(convert_left(s), convert_right(s))

        # check_names
        s_diff_name = Series(ak.array(["a", "b", "c"]), index=Index(ak.arange(3)), name="different_name")
        if both_ak:
            assert_series_equal(s, s_diff_name, check_names=False)
        assert_series_equivalent(convert_left(s), convert_right(s_diff_name), check_names=False)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_series_equal(s, s_diff_name, check_names=True)
        with pytest.raises(AssertionError):
            assert_series_equivalent(convert_left(s), convert_right(s_diff_name), check_names=True)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_series_equal(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        s = Series(ak.array([1, 0, 2]), index=Index(ak.arange(3)))
        s_float = Series(ak.array([1.0, 0.0, 2.0]), index=Index(ak.arange(3) * 1.0))

        if both_ak:
            assert_series_equal(s, s)
        assert_series_equivalent(convert_left(s), convert_right(s))
        if both_ak:
            assert_series_equal(s_float, s_float)
        assert_series_equivalent(convert_left(s_float), convert_right(s_float))

        #   check_dtype
        if both_ak:
            assert_series_equal(s, s_float, check_dtype=False, check_index_type=False)
        assert_series_equivalent(
            convert_left(s), convert_right(s_float), check_dtype=False, check_index_type=False
        )
        if both_ak:
            with pytest.raises(AssertionError):
                assert_series_equal(s, s_float, check_dtype=False, check_index_type=True)
        with pytest.raises(AssertionError):
            assert_series_equivalent(
                convert_left(s), convert_right(s_float), check_dtype=False, check_index_type=True
            )
        if both_ak:
            with pytest.raises(AssertionError):
                assert_series_equal(s, s_float, check_dtype=True, check_index_type=False)
        with pytest.raises(AssertionError):
            assert_series_equivalent(
                convert_left(s), convert_right(s_float), check_dtype=True, check_index_type=False
            )

        # check_index
        s_diff_index = Series(ak.array([1, 0, 2]), index=Index(ak.arange(3) * 2.0))
        if both_ak:
            assert_series_equal(s, s_diff_index, check_index=False)
        assert_series_equivalent(convert_left(s), convert_right(s_diff_index), check_index=False)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_series_equal(s, s_diff_index, check_index=True)
        with pytest.raises(AssertionError):
            assert_series_equivalent(convert_left(s), convert_right(s_diff_index), check_index=True)

        rng = ak.random.default_rng()
        atol = 0.001
        rtol = 0.001
        s_atol = Series(
            ak.array([1, 0, 2]) + rng.random() * atol, index=Index(ak.arange(3) + rng.random() * atol)
        )

        diff_rtol_atol = rtol * ak.array([1, 0, 2]) + rng.random() * atol
        d2 = rtol * ak.arange(3) + rng.random() * atol

        s_rtol_atol = Series(
            ak.array([1, 0, 2]) + diff_rtol_atol,
            index=Index(ak.arange(3) + d2),
        )

        s_2rtol = Series(
            ak.array([1, 0, 2]) + ak.array([1, 0, 2]) * 2 * rtol,
            index=Index(ak.arange(3) + ak.array([1, 0, 2]) * 2 * rtol),
        )

        s_2atol = Series(
            ak.array([1, 0, 2]) + 2 * atol,
            index=Index(ak.arange(3) + 2 * atol),
        )

        if both_ak:
            assert_series_equal(s_float, s_atol, check_exact=False, atol=atol)
        assert_series_equivalent(
            convert_left(s_float), convert_right(s_atol), check_exact=False, atol=atol
        )
        if both_ak:
            assert_series_equal(s_float, s_rtol_atol, check_exact=False, atol=atol, rtol=rtol)
        assert_series_equivalent(
            convert_left(s_float), convert_right(s_rtol_atol), check_exact=False, atol=atol, rtol=rtol
        )
        if both_ak:
            with pytest.raises(AssertionError):
                assert_series_equal(s_float, s_2rtol, check_exact=False, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_series_equivalent(
                convert_left(s_float), convert_right(s_2rtol), check_exact=False, rtol=rtol
            )
        if both_ak:
            with pytest.raises(AssertionError):
                assert_series_equal(s_float, s_2atol, check_exact=False, atol=atol)
        with pytest.raises(AssertionError):
            assert_series_equivalent(
                convert_left(s_float), convert_right(s_2atol), check_exact=False, atol=atol
            )

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_series_equal_check_like(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        # check_like
        s_unordered_index = Series(ak.array([1, 0, 2]), index=Index(ak.array([0, 2, 1])))
        s_ordered_index = s_unordered_index.sort_index()
        if both_ak:
            assert_series_equal(s_ordered_index, s_unordered_index, check_like=True)
        assert_series_equivalent(
            convert_left(s_ordered_index), convert_right(s_unordered_index), check_like=True
        )
        if both_ak:
            with pytest.raises(AssertionError):
                assert_series_equal(s_ordered_index, s_unordered_index, check_like=False)
        with pytest.raises(AssertionError):
            assert_series_equivalent(
                convert_left(s_ordered_index), convert_right(s_unordered_index), check_like=False
            )

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_series_equal_categorical(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        # check_categorical
        # check_category_order

        s3a = Series(
            Categorical(ak.array(["a", "b", "c"])),
            index=Index(Categorical(ak.array(["a", "a", "b"]))),
            name="test",
        )
        s3b = Series(
            Categorical(ak.array(["a", "b", "c"])).sort_values(),
            index=Index(Categorical(ak.array(["a", "a", "b"]))),
            name="test",
        )
        if both_ak:
            assert_series_equal(s3a, s3a)
        assert_series_equivalent(convert_left(s3a), convert_right(s3a))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_series_equal(s3a, s3b, check_categorical=True, check_category_order=True)
        with pytest.raises(AssertionError):
            assert_series_equivalent(
                convert_left(s3a), convert_right(s3b), check_categorical=True, check_category_order=True
            )
        if both_ak:
            assert_series_equal(s3a, s3b, check_categorical=True, check_category_order=False)
        assert_series_equivalent(
            convert_left(s3a), convert_right(s3b), check_categorical=True, check_category_order=False
        )

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_frame_equal(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        df = self.build_ak_df(self)
        df2 = self.build_ak_df(self)
        if both_ak:
            assert_frame_equal(df, df2)
        assert_frame_equivalent(convert_left(df), convert_right(df2))

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_frame_equal_segarray(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        akdf = ak.DataFrame({"rand": ak.SegArray(ak.array([0, 3, 9]), ak.arange(10))})
        if both_ak:
            assert_frame_equal(akdf, akdf)
        assert_frame_equivalent(convert_left(akdf), convert_right(akdf))

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_frame_equal_check_dtype(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        df = self.build_ak_df(self)

        # check_dtype
        df_cpy = df.copy(deep=True)
        if both_ak:
            assert_frame_equal(df, df_cpy, check_dtype=True)
        assert_frame_equivalent(convert_left(df), convert_right(df_cpy), check_dtype=True)
        df_cpy["day"] = cast(df_cpy["day"], dt="float64")
        if both_ak:
            assert_frame_equal(df_cpy, df_cpy, check_dtype=True)
        assert_frame_equivalent(convert_left(df_cpy), convert_right(df_cpy), check_dtype=True)
        if both_ak:
            assert_frame_equal(df, df_cpy, check_dtype=False)
        assert_frame_equivalent(convert_left(df), convert_right(df_cpy), check_dtype=False)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_frame_equal(df, df_cpy, check_dtype=True)
        with pytest.raises(AssertionError):
            assert_frame_equivalent(convert_left(df), convert_right(df_cpy), check_dtype=True)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_frame_equal_check_index_type(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        df = self.build_ak_df(self)

        # check_index_type
        df_float_index = self.build_ak_df(self, index_dtype="float64")
        if both_ak:
            assert_frame_equal(df, df_float_index, check_index_type=False)
        assert_frame_equivalent(convert_left(df), convert_right(df_float_index), check_index_type=False)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_frame_equal(df, df_float_index, check_index_type=True)
        with pytest.raises(AssertionError):
            assert_frame_equivalent(
                convert_left(df), convert_right(df_float_index), check_index_type=True
            )

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_frame_equal_check_names(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        # check_names
        df_name1 = self.build_ak_df(self, index_name="name1")
        df_name2 = self.build_ak_df(self, index_name="name2")
        if both_ak:
            assert_frame_equal(df_name1, df_name2, check_names=False)
        assert_frame_equivalent(convert_left(df_name1), convert_right(df_name2), check_names=False)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_frame_equal(df_name1, df_name2, check_names=True)
        with pytest.raises(AssertionError):
            assert_frame_equivalent(convert_left(df_name1), convert_right(df_name2), check_names=True)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_frame_equal_check_like(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        df = self.build_ak_df(self)

        # check_like
        df_sorted = df.sort_values("amount")
        if both_ak:
            assert_frame_equal(df, df_sorted, check_like=True)
        assert_frame_equivalent(convert_left(df), convert_right(df_sorted), check_like=True)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_frame_equal(df, df_sorted, check_like=False)
        with pytest.raises(AssertionError):
            assert_frame_equivalent(convert_left(df), convert_right(df_sorted), check_like=False)

        df_new_col_order = df[["bi", "userID", "day", "item", "amount", "userName"]]
        if both_ak:
            assert_frame_equal(df, df_new_col_order, check_like=True)
        assert_frame_equivalent(convert_left(df), convert_right(df_new_col_order), check_like=True)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_frame_equal(df, df_new_col_order, check_column_type=True)
        with pytest.raises(AssertionError):
            assert_frame_equivalent(
                convert_left(df), convert_right(df_new_col_order), check_column_type=True
            )

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_frame_equal_check_categorical(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        # check_categorical
        df = self.build_ak_df(self)
        df["userName"] = Categorical(df["userName"])
        df_ordered = self.build_ak_df(self)
        df_ordered["userName"] = Categorical(df_ordered["userName"]).sort_values()

        if both_ak:
            assert_frame_equal(df, df_ordered, check_categorical=False)
        assert_frame_equivalent(convert_left(df), convert_right(df_ordered), check_categorical=False)
        if both_ak:
            with pytest.raises(AssertionError):
                assert_frame_equal(df, df_ordered, check_categorical=True)
        with pytest.raises(AssertionError):
            assert_frame_equivalent(convert_left(df), convert_right(df_ordered), check_categorical=True)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_frame_equal_check_exact(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        # check_exact
        # rtol
        # atol
        rng = ak.random.default_rng(seed=pytest.seed)
        atol = 0.001
        rtol = 0.001

        df = self.build_ak_df(self)
        df_rtol_atol = self.build_ak_df(self)
        df_rtol_atol["amount"] = (
            df_rtol_atol["amount"] + rtol * df_rtol_atol["amount"] + rng.random() * atol
        )

        if both_ak:
            assert_frame_equal(df, df_rtol_atol, check_exact=False, atol=atol, rtol=rtol)
        assert_frame_equivalent(
            convert_left(df), convert_right(df_rtol_atol), check_exact=False, atol=atol, rtol=rtol
        )

        if both_ak:
            with pytest.raises(AssertionError):
                assert_frame_equal(df, df_rtol_atol, check_exact=True)
        with pytest.raises(AssertionError):
            assert_frame_equivalent(convert_left(df), convert_right(df_rtol_atol), check_exact=True)

        if both_ak:
            with pytest.raises(AssertionError):
                assert_frame_equal(df, df_rtol_atol, check_exact=False, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_frame_equivalent(
                convert_left(df), convert_right(df_rtol_atol), check_exact=False, rtol=rtol
            )

        if both_ak:
            with pytest.raises(AssertionError):
                assert_frame_equal(df, df_rtol_atol, check_exact=False, atol=atol)
        with pytest.raises(AssertionError):
            assert_frame_equivalent(
                convert_left(df), convert_right(df_rtol_atol), check_exact=False, atol=atol
            )

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_equal(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        a = ak.arange(size)
        a2 = a + 1
        idx = Index(a)
        idx2 = Index(a2)
        s = Series(a)
        s2 = Series(a2)
        df = DataFrame({"col": a}, index=idx)
        df2 = DataFrame({"col": a2}, index=idx2)

        if both_ak:
            assert_equal(a, a)
        assert_equivalent(convert_left(a), convert_right(a))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_equal(a, a2)
        with pytest.raises(AssertionError):
            assert_equivalent(convert_left(a), convert_right(a2))

        if both_ak:
            assert_equal(idx, idx)
        assert_equivalent(convert_left(idx), convert_right(idx))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_equal(idx, idx2)
        with pytest.raises(AssertionError):
            assert_equivalent(convert_left(idx), convert_right(idx2))

        if both_ak:
            assert_equal(s, s)
        assert_equivalent(convert_left(s), convert_right(s))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_equal(s, s2)
        with pytest.raises(AssertionError):
            assert_equivalent(convert_left(s), convert_right(s2))

        if both_ak:
            assert_equal(df, df)
        assert_equivalent(convert_left(df), convert_right(df))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_equal(df, df2)
        with pytest.raises(AssertionError):
            assert_equivalent(convert_left(df), convert_right(df2))

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_equal_multi_dim(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        shape = (2, 2, size)
        a = ak.arange(4 * size).reshape(shape)
        a2 = a + 1

        if both_ak:
            assert_equal(a, a)
        assert_equivalent(convert_left(a), convert_right(a))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_equal(a, a2)
        with pytest.raises(AssertionError):
            assert_equivalent(convert_left(a), convert_right(a2))

    def test_assert_equal_scalars(self):
        st = "string1"
        st2 = "string2"

        assert_equal(st, st)
        assert_equivalent(st, st)

        with pytest.raises(AssertionError):
            assert_equal(st, st2)
            assert_equivalent(st, st2)

        n = 1.0
        n2 = 1.5

        assert_equal(n, n)
        assert_equivalent(n, n)

        with pytest.raises(AssertionError):
            assert_equal(n, n2)
        with pytest.raises(AssertionError):
            assert_equivalent(n, n2)

    def test_assert_contains_all(self):
        d = {"a": 1, "b": 2, "c": 3}

        assert_contains_all([], d)
        assert_contains_all(["a", "b"], d)
        with pytest.raises(AssertionError):
            assert_contains_all(["a", "d"], d)

    def test_assert_copy(self):
        arrays = [ak.arange(10), ak.arange(10)]

        with pytest.raises(AssertionError):
            assert_copy(arrays, arrays)

        arrays2 = [arry for arry in arrays]
        with pytest.raises(AssertionError):
            assert_copy(arrays, arrays2)

        arrays3 = [ak.arange(10), ak.arange(10)]
        assert_copy(arrays, arrays3)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_arkouda_array_equal(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        a = ak.arange(size)
        a2 = a + 1
        if both_ak:
            assert_arkouda_array_equal(a, a)
        assert_arkouda_array_equivalent(convert_left(a), convert_right(a))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_arkouda_array_equal(a, a2)
        with pytest.raises(AssertionError):
            assert_arkouda_array_equivalent(convert_left(a), convert_right(a2))

        s = ak.array(["a", "b", "b"])
        s2 = ak.array(["a", "b", "c"])
        if both_ak:
            assert_arkouda_array_equal(s, s)
        assert_arkouda_array_equivalent(convert_left(s), convert_right(s))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_arkouda_array_equal(s, s2)
        with pytest.raises(AssertionError):
            assert_arkouda_array_equivalent(convert_left(s), convert_right(s2))

        c = Categorical(s)
        c2 = Categorical(s2)
        if both_ak:
            assert_arkouda_array_equal(c, c)
        assert_arkouda_array_equivalent(convert_left(c), convert_right(c))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_arkouda_array_equal(c, c2)
        with pytest.raises(AssertionError):
            assert_arkouda_array_equivalent(convert_left(c), convert_right(c2))

        if both_ak:
            with pytest.raises(AssertionError):
                assert_arkouda_array_equal(a, s)
        with pytest.raises(AssertionError):
            assert_arkouda_array_equivalent(convert_left(a), convert_right(s))

        if both_ak:
            with pytest.raises(AssertionError):
                assert_arkouda_array_equal(s, c)
        with pytest.raises(AssertionError):
            assert_arkouda_array_equivalent(convert_left(s), convert_right(c))

    @pytest.mark.skip_if_rank_not_compiled([3])
    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_arkouda_array_equal_multi_dim(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        shape = (2, 2, size)
        a = ak.arange(4 * size).reshape(shape)
        a2 = a + 1
        if both_ak:
            assert_arkouda_array_equal(a, a)
        assert_arkouda_array_equivalent(convert_left(a), convert_right(a))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_arkouda_array_equal(a, a2)
        with pytest.raises(AssertionError):
            assert_arkouda_array_equivalent(convert_left(a), convert_right(a2))

    @pytest.mark.skip_if_rank_not_compiled([2])
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_arkouda_array_equal_shape(self, left_as_arkouda, right_as_arkouda):
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        a = ak.arange(4).reshape((2, 2))
        b = ak.arange(4).reshape((1, 4))
        with pytest.raises(AssertionError):
            assert_arkouda_array_equivalent(convert_left(a), convert_right(b))

    def test_assert_arkouda_segarray_equal(self):
        seg = ak.SegArray(ak.array([0, 3, 9]), ak.arange(10))
        seg_cpy = ak.SegArray(ak.array([0, 3, 9]), ak.arange(10))
        seg_float = ak.SegArray(ak.array([0, 3, 9]), ak.arange(10, dtype="float64"))

        assert_arkouda_segarray_equal(seg, seg)

        assert_arkouda_segarray_equal(seg, seg, check_same="same")
        with pytest.raises(AssertionError):
            assert_arkouda_segarray_equal(seg, seg, check_same="copy")

        assert_arkouda_segarray_equal(seg, seg_cpy)
        assert_arkouda_segarray_equal(seg, seg_cpy, check_same="copy")
        with pytest.raises(AssertionError):
            assert_arkouda_segarray_equal(seg, seg_cpy, check_same="same")

        assert_arkouda_segarray_equal(seg, seg_float, check_dtype=False)
        with pytest.raises(AssertionError):
            assert_arkouda_segarray_equal(seg, seg_float, check_dtype=True)

    @pytest.mark.parametrize("size", pytest.prob_size)
    @pytest.mark.parametrize("left_as_arkouda", [True, False])
    @pytest.mark.parametrize("right_as_arkouda", [True, False])
    def test_assert_arkouda_array_equal_bigint(self, size, left_as_arkouda, right_as_arkouda):
        both_ak = left_as_arkouda and right_as_arkouda
        convert_left = self.get_converter(left_as_arkouda)
        convert_right = self.get_converter(right_as_arkouda)

        a = ak.arange(size, dtype=ak.bigint) + (2**64 - size - 1)
        a2 = a + 1
        if both_ak:
            assert_arkouda_array_equal(a, a)
        assert_arkouda_array_equivalent(convert_left(a), convert_right(a))
        if both_ak:
            with pytest.raises(AssertionError):
                assert_arkouda_array_equal(a, a2)
        with pytest.raises(AssertionError):
            assert_arkouda_array_equivalent(convert_left(a), convert_right(a2))
