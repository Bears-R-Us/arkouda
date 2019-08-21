# Makefile for Arkouda
PROJECT_NAME := arkouda
ARKOUDA_SOURCE_DIR := src
ARKOUDA_MAIN_MODULE := arkouda_server
ARKOUDA_MAKEFILES := Makefile Makefile.paths

DEFAULT_TARGET := $(ARKOUDA_MAIN_MODULE)
.PHONY: default
default: $(DEFAULT_TARGET)

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

ALL_TARGETS := $(ARKOUDA_MAIN_MODULE)
.PHONY: all
all: $(ALL_TARGETS)

Makefile.paths:
	touch $@

####################
#### Arkouda.mk ####
####################

define ARKOUDA_HELP_TEXT
# default		$(DEFAULT_TARGET)
  help
  all			$(ALL_TARGETS)

# $(ARKOUDA_MAIN_MODULE)	Can override CHPL_FLAGS
  arkouda-help
  arkouda-clean

endef
export ARKOUDA_HELP_TEXT
HELP_TARGETS += arkouda-help
.PHONY: arkouda-help
arkouda-help:
	@echo "$$ARKOUDA_HELP_TEXT"

ARKOUDA_SOURCES := $(shell find $(ARKOUDA_SOURCE_DIR)/ -type f -name '*.chpl')
ARKOUDA_MAIN_SOURCE := $(ARKOUDA_SOURCE_DIR)/$(ARKOUDA_MAIN_MODULE).chpl

$(ARKOUDA_MAIN_MODULE): $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES)
	$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_MAIN_SOURCE) -o $@

CLEAN_TARGETS += arkouda-clean
.PHONY: arkouda-clean
arkouda-clean:
	$(RM) $(ARKOUDA_MAIN_MODULE) $(ARKOUDA_MAIN_MODULE)_real

####################
#### Archive.mk ####
####################

define ARCHIVE_HELP_TEXT
# archive		COMMIT=$(COMMIT)
  archive-help
  archive-clean

endef
export ARCHIVE_HELP_TEXT
HELP_TARGETS += archive-help
.PHONY: archive-help
archive-help:
	@echo "$$ARCHIVE_HELP_TEXT"

COMMIT ?= master
ARCHIVE_EXTENSION := tar.gz
ARCHIVE_FILENAME := $(PROJECT_NAME)-$(subst /,_,$(COMMIT)).$(ARCHIVE_EXTENSION)

.PHONY: archive
archive: $(ARCHIVE_FILENAME)

.PHONY: $(ARCHIVE_FILENAME)
$(ARCHIVE_FILENAME):
	git archive --format=$(ARCHIVE_EXTENSION) --prefix=$(subst .$(ARCHIVE_EXTENSION),,$(ARCHIVE_FILENAME))/ $(COMMIT) > $@

CLEAN_TARGETS += archive-clean
.PHONY: archive-clean
archive-clean:
	$(RM) $(PROJECT_NAME)-*.$(ARCHIVE_EXTENSION)

################
#### Doc.mk ####
################

define DOC_HELP_TEXT
# doc			Generate $(DOC_DIR)/ with chpldoc
  doc-help
  doc-clean

endef
export DOC_HELP_TEXT
HELP_TARGETS += doc-help
.PHONY: doc-help
doc-help:
	@echo "$$DOC_HELP_TEXT"

DOC_DIR := doc
CHPLDOC := chpldoc
CHPLDOC_FLAGS := --process-used-modules

.PHONY: doc
doc: $(DOC_DIR)/index.html

$(DOC_DIR)/index.html: $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES)
	$(CHPLDOC) $(CHPLDOC_FLAGS) $(ARKOUDA_MAIN_SOURCE) -o $(DOC_DIR)

CLEAN_TARGETS += doc-clean
.PHONY: doc-clean
doc-clean:
	$(RM) -r $(DOC_DIR)

#################
#### Test.mk ####
#################

TEST_SOURCE_DIR := test
TEST_SOURCES := $(wildcard $(TEST_SOURCE_DIR)/*.chpl)
TEST_MODULES := $(basename $(notdir $(TEST_SOURCES)))

TEST_BINARY_DIR := test-bin
TEST_BINARY_SIGIL := #t-
TEST_TARGETS := $(addprefix $(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL),$(TEST_MODULES))
TEST_CHPL_FLAGS ?= $(CHPL_FLAGS)

define TEST_HELP_TEXT
# test			Build all tests ($(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL)*); Can override TEST_CHPL_FLAGS
  test-help
  test-clean
 $(foreach t,$(sort $(TEST_TARGETS)), $(t)\n)
endef
export TEST_HELP_TEXT
HELP_TARGETS += test-help
.PHONY: test-help
test-help:
	@echo "$$TEST_HELP_TEXT"

.PHONY: test
test: $(TEST_TARGETS)

$(TEST_BINARY_DIR):
	mkdir -p $(TEST_BINARY_DIR)

.PHONY: $(TEST_TARGETS) # Force tests to always rebuild.
$(TEST_TARGETS): $(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL)%: $(TEST_SOURCE_DIR)/%.chpl | $(TEST_BINARY_DIR)
	$(CHPL) $(TEST_CHPL_FLAGS) -M $(ARKOUDA_SOURCE_DIR) $< -o $@

CLEAN_TARGETS += test-clean
.PHONY: test-clean
test-clean:
	$(RM) $(TEST_TARGETS) $(addsuffix _real,$(TEST_TARGETS))

#####################
#### Epilogue.mk ####
#####################

define CLEAN_HELP_TEXT
# clean
  clean-help
 $(foreach t,$(CLEAN_TARGETS), $(t)\n)
endef
export CLEAN_HELP_TEXT
HELP_TARGETS += clean-help
.PHONY: clean-help
clean-help:
	@echo "$$CLEAN_HELP_TEXT"

.PHONY: clean
clean: $(CLEAN_TARGETS)

.PHONY: help
help: $(HELP_TARGETS)
