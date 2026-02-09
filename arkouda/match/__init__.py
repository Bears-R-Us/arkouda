# flake8: noqa

import warnings

warnings.warn(
    "arkouda.match is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.match instead.",
    DeprecationWarning,
    stacklevel=2,
)
from arkouda.pandas.match import *
