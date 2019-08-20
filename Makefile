# Makefile for Arkouda
ARKOUDA_SOURCE_DIR := src
ARKOUDA_MAIN_MODULE := arkouda_server
ARKOUDA_MAKEFILES := Makefile Makefile.paths

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

####################
#### Arkouda.mk ####
####################

$(ARKOUDA_MAIN_MODULE): $(shell find $(ARKOUDA_SOURCE_DIR)/ -type f -name '*.chpl') $(ARKOUDA_MAKEFILES)
	$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_SOURCE_DIR)/$(ARKOUDA_MAIN_MODULE).chpl -o $@

CLEAN_TARGETS += arkouda-clean
.PHONY: arkouda-clean
arkouda-clean:
	$(RM) $(ARKOUDA_MAIN_MODULE) $(ARKOUDA_MAIN_MODULE)_real

#################
#### Test.mk ####
#################

TEST_SOURCE_DIR := test
TEST_SOURCE_FILES := $(wildcard $(TEST_SOURCE_DIR)/*.chpl)
TEST_MODULES := $(basename $(notdir $(TEST_SOURCE_FILES)))

TEST_BINARY_SIGIL := t-
TEST_TARGETS := $(addprefix $(TEST_BINARY_SIGIL),$(TEST_MODULES))
TEST_CHPL_FLAGS ?= $(CHPL_FLAGS)

.PHONY: test
test: $(TEST_TARGETS)

.PHONY: $(TEST_TARGETS) # Force tests to always rebuild.
$(TEST_TARGETS): $(TEST_BINARY_SIGIL)%: $(TEST_SOURCE_DIR)/%.chpl
	$(CHPL) $(TEST_CHPL_FLAGS) -M $(ARKOUDA_SOURCE_DIR) $< -o $@

CLEAN_TARGETS += test-clean
.PHONY: test-clean
test-clean:
	$(RM) $(TEST_TARGETS) $(addsuffix _real,$(TEST_TARGETS))

#####################
#### Epilogue.mk ####
#####################

.PHONY: clean
clean: $(CLEAN_TARGETS)
