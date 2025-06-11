"""
Accessor utilities for Arkouda Series-like objects.

This module defines infrastructure for namespace-based accessors (e.g., `.str`, `.dt`)
on Arkouda Series, mimicking the behavior of pandas-style accessors. It supports
extension methods for string and datetime-like values, enabling operations to be
performed in a clean, grouped syntax.

Exports
-------
__all__ = [
    "CachedAccessor",
    "DatetimeAccessor",
    "Properties",
    "StringAccessor",
    "date_operators",
    "string_operators",
]

Components
----------
CachedAccessor : class
    Descriptor that lazily initializes and caches accessor objects, such as `.str` or `.dt`.

DatetimeAccessor : class
    Implements datetime-like operations (e.g., floor, ceil, round) via the `.dt` accessor.

StringAccessor : class
    Implements string-like operations (e.g., contains, startswith, endswith) via the `.str` accessor.

Properties : base class
    Base class that provides `_make_op` for dynamically attaching operations to accessors.

date_operators : function
    Class decorator that adds datetime operations to `DatetimeAccessor`.

string_operators : function
    Class decorator that adds string operations to `StringAccessor`.

Usage
-----
>>> import arkouda as ak
>>> from arkouda import Series
>>> s = Series(["apple", "banana", "apricot"])
>>> s.str.startswith("a")
0     True
1    False
2     True
dtype: bool

>>> from arkouda import Datetime
>>> t = Series(Datetime(ak.array([1_000_000_000_000])))
>>> t.dt.floor("D")
0   1970-01-01
dtype: datetime64[ns]

Notes
-----
These accessors are automatically attached to compatible Series objects.
Users should not instantiate accessors directly — use `.str` and `.dt` instead.

"""

from arkouda.categorical import Categorical
from arkouda.numpy.strings import Strings
from arkouda.numpy.timeclass import Datetime

__all__ = [
    "CachedAccessor",
    "DatetimeAccessor",
    "Properties",
    "StringAccessor",
    "date_operators",
    "string_operators",
]


class CachedAccessor:
    """
    Custom property-like object.

    A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``df.foo``.
    accessor : cls
        Class with the extension methods.

    Notes
    -----
    For accessor, The class's __init__ method assumes that one of
    ``Series``, ``DataFrame`` or ``Index`` as the
    single argument ``data``.

    """

    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        accessor_obj = self._accessor(obj)
        # Replace the property with the accessor object. Inspired by:
        # https://www.pydanny.com/cached-property.html
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        # NDFrame
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def string_operators(cls):
    for name in ["contains", "startswith", "endswith"]:
        setattr(cls, name, cls._make_op(name))
    return cls


def date_operators(cls):
    for name in ["floor", "ceil", "round"]:
        setattr(cls, name, cls._make_op(name))
    return cls


class Properties:
    @classmethod
    def _make_op(cls, name):
        def accessop(self, *args, **kwargs):
            from .pandas import Series

            results = getattr(self.series.values, name)(*args, **kwargs)
            return Series(data=results, index=self.series.index)

        return accessop


@date_operators
class DatetimeAccessor(Properties):
    def __init__(self, series):
        data = series.values
        if not isinstance(data, Datetime):
            raise AttributeError("Can only use .dt accessor with datetimelike values")

        self.series = series


@string_operators
class StringAccessor(Properties):
    def __init__(self, series):
        data = series.values
        if not (isinstance(data, Categorical) or isinstance(data, Strings)):
            raise AttributeError("Can only use .str accessor with string like values")

        self.series = series
