# flake8: noqa
import warnings

warnings.warn(
    "arkouda.accessor is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.accessor instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.accessor import (
    CachedAccessor,
    DatetimeAccessor,
    Properties,
    StringAccessor,
    date_operators,
    string_operators,
)
