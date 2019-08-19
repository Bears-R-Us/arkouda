# Makefile for Arkouda
ARKOUDA_MAIN_MODULE := arkouda_server

.PHONY: default
default: $(ARKOUDA_MAIN_MODULE)

CHPL := chpl
CHPL_FLAGS += --print-passes
CHPL_FLAGS += --ccflags="-Wno-incompatible-pointer-types" --cache-remote --instantiate-max 1024 --fast
CHPL_FLAGS += -lhdf5 -lhdf5_hl -lzmq

# add-path: Append custom paths for non-system software.
define add-path
CHPL_FLAGS += -I$(1)/include -L$(1)/lib --ldflags="-Wl,-rpath=$(1)/lib"
endef
# Usage: $(eval $(call add-path,/home/user/anaconda3/envs/arkouda))
#                               ^ no space after comma
-include Makefile.paths # Add entries to this file.

# System Environment
ifdef LD_RUN_PATH
CHPL_FLAGS += --ldflags="-Wl,-rpath=$(LD_RUN_PATH)"
endif

.PHONY: all
all: $(ARKOUDA_MAIN_MODULE)

Makefile.paths:
	touch $@

$(ARKOUDA_MAIN_MODULE): $(shell find src/ -type f -name '*.chpl') Makefile Makefile.paths
	$(CHPL) $(CHPL_FLAGS) src/$(ARKOUDA_MAIN_MODULE).chpl

.PHONY: clean
clean:
	$(RM) $(ARKOUDA_MAIN_MODULE) $(ARKOUDA_MAIN_MODULE)_real
