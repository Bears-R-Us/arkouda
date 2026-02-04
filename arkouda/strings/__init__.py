# flake8: noqa

import warnings

warnings.warn(
    "arkouda.strings is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.strings instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.strings import Strings
