# Prologue: shared variables, flags, dependency helpers, and checks.
# This file is intentionally small; logic is split across make/prologue/*.mk and make/deps/*.mk.

include make/prologue/core.mk
include make/prologue/chapel.mk
include make/prologue/paths.mk
include make/prologue/config.mk
include make/prologue/arrow_shims.mk

# Dependency meta-targets and per-dependency rules
include make/deps/common.mk
include make/deps/zmq.mk
include make/deps/hdf5.mk
include make/deps/arrow.mk
include make/deps/iconv.mk
include make/deps/idn2.mk
include make/deps/blosc.mk
include make/deps/pytables.mk

include make/prologue/system_env.mk
include make/prologue/python_interop.mk
include make/prologue/checks.mk
include make/prologue/targets_help.mk
