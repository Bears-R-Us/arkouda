# Makefile for Arkouda
ARKOUDA_PROJECT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

PROJECT_NAME := arkouda
ARKOUDA_SOURCE_DIR := src
ARKOUDA_MAIN_MODULE := arkouda_server
ARKOUDA_MAKEFILES := Makefile Makefile.paths

DEFAULT_TARGET := $(ARKOUDA_MAIN_MODULE)
.PHONY: default
default: $(DEFAULT_TARGET)

VERBOSE ?= 0

CHPL := chpl
CHPL_DEBUG_FLAGS += --print-passes
CHPL_FLAGS += --fast --legacy-classes
CHPL_FLAGS += -lhdf5 -lhdf5_hl -lzmq

# --cache-remote does not work with ugni in Chapel 1.20
COMM = $(shell $(CHPL_HOME)/util/chplenv/chpl_comm.py 2>/dev/null)
ifneq ($(COMM),ugni)
CHPL_FLAGS += --cache-remote
endif

# add-path: Append custom paths for non-system software.
# Note: Darwin `ld` only supports `-rpath <path>`, not `-rpath=<paths>`.
define add-path
CHPL_FLAGS += -I$(1)/include -L$(1)/lib --ldflags="-Wl,-rpath,$(1)/lib"
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

# args: RuleTarget DefinedHelpText
define create_help_target
export $(2)
HELP_TARGETS += $(1)
.PHONY: $(1)
$(1):
	@echo "$$$$$(2)"
endef

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
$(eval $(call create_help_target,arkouda-help,ARKOUDA_HELP_TEXT))

ARKOUDA_SOURCES = $(shell find $(ARKOUDA_SOURCE_DIR)/ -type f -name '*.chpl')
ARKOUDA_MAIN_SOURCE := $(ARKOUDA_SOURCE_DIR)/$(ARKOUDA_MAIN_MODULE).chpl

$(ARKOUDA_MAIN_MODULE): $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES)
	$(CHPL) $(CHPL_DEBUG_FLAGS) $(CHPL_FLAGS) $(ARKOUDA_MAIN_SOURCE) -o $@

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
$(eval $(call create_help_target,archive-help,ARCHIVE_HELP_TEXT))

COMMIT ?= master
ARCHIVE_EXTENSION := tar.gz
ARCHIVE_FILENAME := $(PROJECT_NAME)-$(subst /,_,$(COMMIT)).$(ARCHIVE_EXTENSION)

.PHONY: archive
archive: $(ARCHIVE_FILENAME)

.PHONY: $(ARCHIVE_FILENAME)
$(ARCHIVE_FILENAME):
	git archive --format=$(ARCHIVE_EXTENSION) --prefix=$(subst .$(ARCHIVE_EXTENSION),,$(ARCHIVE_FILENAME))/ $(COMMIT) > $@

CLEANALL_TARGETS += archive-clean
.PHONY: archive-clean
archive-clean:
	$(RM) $(PROJECT_NAME)-*.$(ARCHIVE_EXTENSION)

################
#### Doc.mk ####
################

define DOC_HELP_TEXT
# doc			Generate $(DOC_DIR)/ with doc-* for server, etc.
  doc-help
  doc-clean
  doc-server
  doc-python

endef
$(eval $(call create_help_target,doc-help,DOC_HELP_TEXT))

DOC_DIR := doc
DOC_SERVER_OUTPUT_DIR := $(ARKOUDA_PROJECT_DIR)/$(DOC_DIR)/server
DOC_PYTHON_OUTPUT_DIR := $(ARKOUDA_PROJECT_DIR)/$(DOC_DIR)/python

DOC_COMPONENTS := \
	$(DOC_SERVER_OUTPUT_DIR) \
	$(DOC_PYTHON_OUTPUT_DIR)
$(DOC_COMPONENTS):
	mkdir -p $@

.PHONY: doc
doc: doc-server doc-python

CHPLDOC := chpldoc
CHPLDOC_FLAGS := --process-used-modules
.PHONY: doc-server
doc-server: $(DOC_SERVER_OUTPUT_DIR)/index.html
$(DOC_SERVER_OUTPUT_DIR)/index.html: $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES) | $(DOC_SERVER_OUTPUT_DIR)
	@echo "Building documentation for: Server"
	$(CHPLDOC) $(CHPLDOC_FLAGS) $(ARKOUDA_MAIN_SOURCE) -o $(DOC_SERVER_OUTPUT_DIR)

DOC_PYTHON_SOURCE_DIR := pydoc
DOC_PYTHON_SOURCES = $(shell find $(DOC_PYTHON_SOURCE_DIR)/ -type f)
.PHONY: doc-python
doc-python: $(DOC_PYTHON_OUTPUT_DIR)/index.html
$(DOC_PYTHON_OUTPUT_DIR)/index.html: $(DOC_PYTHON_SOURCES) $(ARKOUDA_MAKEFILES)
	@echo "Building documentation for: Python"
	$(eval $@_TMP := $(shell mktemp -d))
	@# Build the documentation to a temporary output directory.
	cd $(DOC_PYTHON_SOURCE_DIR) && $(MAKE) BUILDDIR=$($@_TMP) html
	@# Delete old output directory and move `html` directory to its place.
	$(RM) -r $(DOC_PYTHON_OUTPUT_DIR)
	mv $($@_TMP)/html $(DOC_PYTHON_OUTPUT_DIR)
	$(RM) -r $($@_TMP)

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

ifeq ($(VERBOSE),1)
TEST_CHPL_FLAGS ?= $(CHPL_DEBUG_FLAGS) $(CHPL_FLAGS)
else
TEST_CHPL_FLAGS ?= $(CHPL_FLAGS)
endif

define TEST_HELP_TEXT
# test			Build all tests ($(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL)*); Can override TEST_CHPL_FLAGS
  test-help
  test-clean
 $(foreach t,$(sort $(TEST_TARGETS)), $(t)\n)
endef
$(eval $(call create_help_target,test-help,TEST_HELP_TEXT))

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
$(eval $(call create_help_target,clean-help,CLEAN_HELP_TEXT))

.PHONY: clean cleanall
clean: $(CLEAN_TARGETS)
cleanall: clean $(CLEANALL_TARGETS)

.PHONY: help
help: $(HELP_TARGETS)
