# flake8: noqa

import warnings

warnings.warn(
    "arkouda.client is deprecated and will be removed in a future release. "
    "Please use arkouda.core.client instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.core.client import connect, disconnect, shutdown, get_config, get_registration_config, get_max_array_rank, get_mem_used, get_mem_avail, get_mem_status, get_server_commands, print_server_commands, generate_history, ruok
