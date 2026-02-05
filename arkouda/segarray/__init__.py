# flake8: noqa

import warnings

warnings.warn(
    "arkouda.segarray is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.segarray instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.segarray import SegArray
