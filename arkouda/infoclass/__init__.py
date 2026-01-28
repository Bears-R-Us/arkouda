# flake8: noqa

import warnings

warnings.warn(
    "arkouda.infoclass is deprecated and will be removed in a future release. "
    "Please use arkouda.core.infoclass instead.",
    DeprecationWarning,
    stacklevel=2,
)

from arkouda.core.infoclass import AllSymbols, RegisteredSymbols, information, list_registry, list_symbol_table, pretty_print_information
