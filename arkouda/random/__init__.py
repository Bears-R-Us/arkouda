# flake8: noqa

import warnings

warnings.warn(
    "arkouda.random is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.random instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.random import Generator, randint, standard_normal, uniform
