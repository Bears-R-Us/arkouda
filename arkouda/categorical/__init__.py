# flake8: noqa

import warnings

warnings.warn(
    "arkouda.categorical is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.categorical instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.categorical import *
