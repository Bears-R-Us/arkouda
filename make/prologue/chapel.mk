# Chapel compiler configuration and base flags
CHPL := chpl
ARKOUDA_CHPL_HOME=$(shell $(CHPL) --print-chpl-home 2>/dev/null)
CHPL_MAJOR := $(shell $(CHPL) --version | sed -n "s/chpl version \([0-9]\)\.[0-9]*.*/\1/p")
CHPL_MINOR := $(shell $(CHPL) --version | sed -n "s/chpl version [0-9]\.\([0-9]*\).*/\1/p")

# We need to make the HDF5 API use the 1.10.x version for compatibility between 1.10 and 1.12
CHPL_FLAGS += --ccflags="-DH5_USE_110_API"

# silence warnings about '@arkouda' annotations being unrecognized
CHPL_FLAGS += --using-attribute-toolname arkouda

CHPL_DEBUG_FLAGS += --print-passes
ifeq ($(shell expr $(CHPL_MINOR) \>= 7),1)
CHPL_DEBUG_FLAGS += --print-passes-memory
endif

ifdef ARKOUDA_DEVELOPER
ARKOUDA_QUICK_COMPILE = true
ARKOUDA_RUNTIME_CHECKS = true
# ARKOUDA_DEBUG = true
endif

ifdef ARKOUDA_QUICK_COMPILE
CHPL_FLAGS += --no-checks --no-loop-invariant-code-motion --no-fast-followers --ccflags="-O0"
else
CHPL_FLAGS += --fast
endif

ifdef ARKOUDA_RUNTIME_CHECKS
CHPL_FLAGS += --checks
endif

# FIXME: enabling this as of Chapel 2.7.0 may cause compiler crashes. bug fixes in the Chapel compiler
# are needed before this can be reenabled by default.
# ifdef ARKOUDA_DEBUG
# # In the future, we can just use --debug which implies -g and --debug-safe-optimizations-only
# CHPL_FLAGS += -g
# ifeq ($(shell expr $(CHPL_MINOR) \>= 7),1)
# CHPL_FLAGS += --debug-safe-optimizations-only
# endif
# endif

CHPL_FLAGS += -smemTrack=true -smemThreshold=1048576

# We have seen segfaults with cache remote at some node counts
CHPL_FLAGS += --no-cache-remote

# For configs that use a fixed heap, but still have first-touch semantics
# (gasnet-ibv-large) interleave large allocations to reduce the performance hit
# from getting progressively worse NUMA affinity due to memory reuse.
CHPL_HELP := $(shell $(CHPL) --devel --help)
ifneq (,$(findstring interleave-memory,$(CHPL_HELP)))
CHPL_FLAGS += --interleave-memory
endif

# Allows the user to set a different instantiation limit
ifneq ($(instantiate_max),)
CHPL_USER_FLAGS += --instantiate-max=$(instantiate_max)
endif

CHPL_FLAGS += -lhdf5 -lhdf5_hl -lzmq -liconv -lidn2 -lparquet -larrow

