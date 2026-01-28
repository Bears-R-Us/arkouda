# flake8: noqa

import warnings

warnings.warn(
    "arkouda.pdarraysetops is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.pdarraysetops instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.pdarraysetops import (
    concatenate,
    in1d,
    indexof1d,
    intersect1d,
    setdiff1d,
    setxor1d,
    union1d,
)
