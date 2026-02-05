# flake8: noqa

import warnings

warnings.warn(
    "arkouda.io is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.io instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.io import *
