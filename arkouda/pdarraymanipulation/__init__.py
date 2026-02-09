# flake8: noqa

import warnings

warnings.warn(
    "arkouda.pdarraymanipulation is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.pdarraymanipulation instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.pdarraymanipulation import delete, hstack, vstack
