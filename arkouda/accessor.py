"""
Accessor utilities for Arkouda Series-like objects.

This module defines infrastructure for namespace-based accessors (e.g., `.str`, `.dt`)
on Arkouda Series, mimicking the behavior of pandas-style accessors. It supports
extension methods for string and datetime-like values, enabling operations to be
performed in a clean, grouped syntax.

Components
----------
:class:`.CachedAccessor`
    Descriptor that lazily initializes and caches accessor objects, such as `.str` or `.dt`.

:class:`.DatetimeAccessor`
    Implements datetime-like operations (e.g., floor, ceil, round) via the `.dt` accessor.

:class:`.StringAccessor`
    Implements string-like operations (e.g., contains, startswith, endswith) via the `.str` accessor.

:class:`.Properties`
    Base class that provides `_make_op` for dynamically attaching operations to accessors.

:func:`.date_operators`
    Class decorator that adds datetime operations to `DatetimeAccessor`.

:func:`.string_operators`
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

from typing import TYPE_CHECKING, TypeVar

from arkouda.numpy.strings import Strings
from arkouda.numpy.timeclass import Datetime


if TYPE_CHECKING:
    from arkouda.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")

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
    Descriptor for caching namespace-based accessors.

    This custom property-like object enables lazy initialization of accessors
    (e.g., `.str`, `.dt`) on Series-like objects, similar to pandas-style extension
    accessors.

    Parameters
    ----------
    name : str
        The name of the namespace to be accessed (e.g., ``df.foo``).
    accessor : type
        A class implementing the accessor logic.

    Notes
    -----
    The `accessor` class's ``__init__`` method must accept a single positional
    argument, which should be one of ``Series``, ``DataFrame``, or ``Index``.
    """

    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        """
        Retrieve and cache the accessor instance for the calling object.

        Parameters
        ----------
        obj : object
            The instance that the accessor is being called on.
        cls : type
            The class of the object.

        Returns
        -------
        object
            The accessor instance attached to the object.

        """
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
    """
    Add common string operation methods to a StringAccessor class.

    This class decorator dynamically attaches string operations (`contains`,
    `startswith`, `endswith`) to the given class using the `_make_op` helper.

    Parameters
    ----------
    cls : type
        The accessor class to decorate.

    Returns
    -------
    type
        The accessor class with string methods added.

    Notes
    -----
    Used internally to implement the `.str` accessor API.

    """
    for name in ["contains", "startswith", "endswith"]:
        setattr(cls, name, cls._make_op(name))
    return cls


def date_operators(cls):
    """
    Add common datetime operation methods to a DatetimeAccessor class.

    This class decorator dynamically attaches datetime operations (`floor`,
    `ceil`, `round`) to the given class using the `_make_op` helper.

    Parameters
    ----------
    cls : type
        The accessor class to decorate.

    Returns
    -------
    type
        The accessor class with datetime methods added.

    Notes
    -----
    Used internally to implement the `.dt` accessor API.

    """
    for name in ["floor", "ceil", "round"]:
        setattr(cls, name, cls._make_op(name))
    return cls


class Properties:
    """
    Base class for accessor implementations in Arkouda.

    Provides the `_make_op` class method to dynamically generate accessor methods
    that wrap underlying `Strings` or `Datetime` operations and return new Series.

    Notes
    -----
    This class is subclassed by `StringAccessor` and `DatetimeAccessor`, and is not
    intended to be used directly.

    Examples
    --------
    Subclasses should define `_make_op("operation_name")`, which will generate
    a method that applies `series.values.operation_name(...)` and returns a new Series.

    """

    @classmethod
    def _make_op(cls, name):
        def accessop(self, *args, **kwargs):
            from .pandas import Series

            results = getattr(self.series.values, name)(*args, **kwargs)
            return Series(data=results, index=self.series.index)

        return accessop


@date_operators
class DatetimeAccessor(Properties):
    r"""
    Accessor for datetime-like operations on Arkouda Series.

    Provides datetime methods such as `.floor()`, `.ceil()`, and `.round()`,
    mirroring the `.dt` accessor in pandas.

    This accessor is automatically attached to Series objects that wrap
    `arkouda.Datetime` values. It should not be instantiated directly.

    Parameters
    ----------
    series : arkouda.pandas.Series
        The Series object containing `Datetime` values.

    Raises
    ------
    AttributeError
        If the underlying Series values are not of type `arkouda.Datetime`.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda import Datetime, Series
    >>> s = Series(Datetime(ak.array([1_000_000_000_000])))
    >>> s.dt.floor("D")
    0   1970-01-01
    dtype: datetime64[ns]

    """

    def __init__(self, series):
        data = series.values
        if not isinstance(data, Datetime):
            raise AttributeError("Can only use .dt accessor with datetimelike values")

        self.series = series


@string_operators
class StringAccessor(Properties):
    """
    Accessor for string operations on Arkouda Series.

    Provides string-like methods such as `.contains()`, `.startswith()`, and
    `.endswith()` via the `.str` accessor, similar to pandas.

    This accessor is automatically attached to Series objects that wrap
    `arkouda.Strings` or `arkouda.Categorical` values. It should not be instantiated directly.

    Parameters
    ----------
    series : arkouda.pandas.Series
        The Series object containing `Strings` or `Categorical` values.

    Raises
    ------
    AttributeError
        If the underlying Series values are not `Strings` or `Categorical`.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda import Series
    >>> s = Series(["apple", "banana", "apricot"])
    >>> s.str.startswith("a")
    0     True
    1    False
    2     True
    dtype: bool

    """

    def __init__(self, series):
        data = series.values
        if not (isinstance(data, Categorical) or isinstance(data, Strings)):
            raise AttributeError("Can only use .str accessor with string like values")

        self.series = series
