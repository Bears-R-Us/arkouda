# flake8: noqa

import warnings

warnings.warn(
    "arkouda.index is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.index instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.index import *
