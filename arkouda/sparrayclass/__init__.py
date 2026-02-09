# flake8: noqa

import warnings

warnings.warn(
    "arkouda.sparrayclass is deprecated and will be removed in a future release. "
    "Please use arkouda.scipy.sparrayclass instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.scipy.sparrayclass import create_sparray, sparray
