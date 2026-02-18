import importlib

import pytest

import arkouda as ak
from arkouda import Series
from arkouda.accessor import (
    CachedAccessor,
    DatetimeAccessor,
    StringAccessor,
    date_operators,
    string_operators,
)
from arkouda.categorical import Categorical
from arkouda.numpy.pdarraycreation import array
from arkouda.numpy.timeclass import Datetime
from arkouda.testing import assert_arkouda_pdarray_equal

_pd = importlib.import_module("pandas")
pd_Timestamp = getattr(_pd, "Timestamp")
pd_Series = getattr(_pd, "Series")
pd_assert_series_equal = getattr(_pd.testing, "assert_series_equal")


class TestAccessor:
    def test_alignment_docstrings(self):
        import doctest

        from arkouda import accessor

        result = doctest.testmod(accessor, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"


class TestCachedAccessor:
    def test_cached_accessor_init(self):
        def mock_accessor(data):
            return data

        accessor = CachedAccessor("test", mock_accessor)
        assert accessor._name == "test"
        assert accessor._accessor == mock_accessor

    def test_cached_accessor_get(self):
        class MockSeries:
            pass

        # Mock a sample accessor
        def mock_accessor(data):
            return data

        series = MockSeries()
        accessor = CachedAccessor("test", mock_accessor)

        # Accessor should cache the result
        result = accessor.__get__(series, MockSeries)
        assert result is series  # As the mock accessor simply returns the object

        # Accessor should now be cached
        assert hasattr(series, "test")
        assert series.test is result

    def test_cached_accessor_get_class_attribute(self):
        class MockSeries:
            pass

        def mock_accessor(data):
            return data

        accessor = CachedAccessor("test", mock_accessor)
        result = accessor.__get__(None, MockSeries)
        assert result == mock_accessor


class TestStringOperators:
    def test_string_operators_decorator(self):
        # Create a mock class
        class MockAccessor:
            @classmethod
            def _make_op(cls, name):
                return name

        decorated_class = string_operators(MockAccessor)
        for name in ["contains", "startswith", "endswith"]:
            assert hasattr(decorated_class, name)
            assert getattr(decorated_class, name) == name

    def test_4524_strings_reproduer(self):
        s = Series(["apple", "banana", "apricot"])
        s.str.startswith("a")
        assert_arkouda_pdarray_equal(s.str.startswith("a").values, ak.array([True, False, True]))


class TestDateOperators:
    def test_date_operators_decorator(self):
        # Create a mock class
        class MockAccessor:
            @classmethod
            def _make_op(cls, name):
                return name

        decorated_class = date_operators(MockAccessor)
        for name in ["floor", "ceil", "round"]:
            assert hasattr(decorated_class, name)
            assert getattr(decorated_class, name) == name


class TestDatetimeAccessor:
    def test_datetime_accessor_init_with_valid_series(self):
        class MockSeries:
            values = Datetime(ak.array([1_000_000_000_000]))

        series = MockSeries()
        accessor = DatetimeAccessor(series)
        assert accessor.series == series

    def test_datetime_accessor_init_with_invalid_series(self):
        class MockSeries:
            values = "not_datetime"

        series = MockSeries()
        with pytest.raises(AttributeError, match="Can only use \.dt accessor with datetimelike values"):
            DatetimeAccessor(series)

    def test_4524_datetime_reproduer(self):
        s = Series(Datetime(ak.array([1_000_000_000_000])))
        pd_assert_series_equal(
            s.dt.floor("s").to_pandas(), pd_Series(pd_Timestamp("1970-01-01 00:16:40"))
        )


class TestStringAccessor:
    def test_string_accessor_init_with_valid_categorical_series(self):
        class MockSeries:
            values = Categorical(array(["a", "b", "a"]))

        series = MockSeries()
        accessor = StringAccessor(series)
        assert accessor.series == series

    def test_string_accessor_init_with_valid_strings_series(self):
        class MockSeries:
            values = array(["a", "b", "a"])

        series = MockSeries()
        accessor = StringAccessor(series)
        assert accessor.series == series

    def test_string_accessor_init_with_invalid_series(self):
        class MockSeries:
            values = "not_categorical_or_strings"

        series = MockSeries()
        with pytest.raises(AttributeError, match="Can only use \.str accessor with string like values"):
            StringAccessor(series)
