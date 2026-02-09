# flake8: noqa

import warnings

warnings.warn(
    "arkouda.timeclass is deprecated and will be removed in a future release. "
    "Please use arkouda.numpy.timeclass instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.numpy.timeclass import Datetime, Timedelta, date_range, timedelta_range
