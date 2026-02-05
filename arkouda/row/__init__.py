# flake8: noqa

import warnings

warnings.warn(
    "arkouda.row is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.row instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.row import Row
