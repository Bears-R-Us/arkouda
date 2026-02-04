# flake8: noqa

import warnings

warnings.warn(
    "arkouda.logger is deprecated and will be removed in a future release. "
    "Please use arkouda.core.logger instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.core.logger import LogLevel,enable_verbose,disable_verbose,write_log,enableVerbose,disableVerbose
