# flake8: noqa

import warnings

warnings.warn(
    "arkouda.series is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.series instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.series import Series
