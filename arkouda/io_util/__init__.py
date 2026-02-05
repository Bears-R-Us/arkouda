# flake8: noqa

import warnings

warnings.warn(
    "arkouda.io_util is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.io_util instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.io_util import *
