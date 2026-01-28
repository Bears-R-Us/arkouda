# flake8: noqa

import warnings

warnings.warn(
    "arkouda.message is deprecated and will be removed in a future release. "
    "Please use arkouda.core.message instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.core.message import MessageFormat, MessageType, ParameterObject, ReplyMessage, RequestMessage
