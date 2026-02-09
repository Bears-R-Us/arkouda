# flake8: noqa

import warnings

warnings.warn(
    "arkouda.matcher is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.matcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.matcher import *
