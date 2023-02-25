from arkouda.categorical import Categorical
from arkouda.strings import Strings
from arkouda.timeclass import Datetime


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
            from . import Series

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
