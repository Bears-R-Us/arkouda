# flake8: noqa

import warnings

warnings.warn(
    "arkouda.dataframe is deprecated and will be removed in a future release. "
    "Please use arkouda.pandas.dataframe instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.pandas.dataframe import *
