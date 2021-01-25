# Makefile for Arkouda
ARKOUDA_PROJECT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

PROJECT_NAME := arkouda
ARKOUDA_SOURCE_DIR := $(ARKOUDA_PROJECT_DIR)/src
ARKOUDA_MAIN_MODULE := arkouda_server
ARKOUDA_MAKEFILES := Makefile Makefile.paths

DEFAULT_TARGET := $(ARKOUDA_MAIN_MODULE)
.PHONY: default
default: $(DEFAULT_TARGET)

VERBOSE ?= 0

define ARKOUDA_QUICK_COMPILE
CHPL_FLAGS += --no-checks --no-loop-invariant-code-motion --no-fast-followers --ccflags="-O0"
endef
CHPL := chpl
CHPL_DEBUG_FLAGS += --print-passes
ifdef ARKOUDA_DEVELOPER
CHPL_FLAGS += --ccflags="-O1"
else ifdef ARKOUDA_QUICK_COMPILE
CHPL_FLAGS += --no-checks --no-loop-invariant-code-motion --no-fast-followers --ccflags="-O0"
else
#CHPL_FLAGS += --fast
endif
CHPL_FLAGS += -smemTrack=true
CHPL_FLAGS += -lhdf5 -lhdf5_hl -lzmq

# add-path: Append custom paths for non-system software.
# Note: Darwin `ld` only supports `-rpath <path>`, not `-rpath=<paths>`.
define add-path
CHPL_FLAGS += -I$(1)/include -L$(1)/lib --ldflags="-Wl,-rpath,$(1)/lib"
endef
# Usage: $(eval $(call add-path,/home/user/anaconda3/envs/arkouda))
#                               ^ no space after comma
-include Makefile.paths # Add entries to this file.

ifdef ARKOUDA_ZMQ_PATH
$(eval $(call add-path,$(ARKOUDA_ZMQ_PATH)))
endif
ifdef ARKOUDA_HDF5_PATH
$(eval $(call add-path,$(ARKOUDA_HDF5_PATH)))
endif

.PHONY: install-deps
install-deps: install-zmq install-hdf5

DEP_DIR := dep
DEP_INSTALL_DIR := $(ARKOUDA_PROJECT_DIR)/$(DEP_DIR)
DEP_BUILD_DIR := $(ARKOUDA_PROJECT_DIR)/$(DEP_DIR)/build

ZMQ_VER := 4.3.2
ZMQ_NAME_VER := zeromq-$(ZMQ_VER)
ZMQ_BUILD_DIR := $(DEP_BUILD_DIR)/$(ZMQ_NAME_VER)
ZMQ_INSTALL_DIR := $(DEP_INSTALL_DIR)/zeromq-install
ZMQ_LINK := https://github.com/zeromq/libzmq/releases/download/v$(ZMQ_VER)/$(ZMQ_NAME_VER).tar.gz
install-zmq:
	@echo "Installing ZeroMQ"
	rm -rf $(ZMQ_BUILD_DIR) $(ZMQ_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)
	cd $(DEP_BUILD_DIR) && curl -sL $(ZMQ_LINK) | tar xz
	cd $(ZMQ_BUILD_DIR) && ./configure --prefix=$(ZMQ_INSTALL_DIR) CFLAGS=-O3 CXXFLAGS=-O3 && make && make install
	rm -r $(ZMQ_BUILD_DIR)
	echo '$$(eval $$(call add-path,$(ZMQ_INSTALL_DIR)))' >> Makefile.paths

HDF5_MAJ_MIN_VER := 1.10
HDF5_VER := 1.10.5
HDF5_NAME_VER := hdf5-$(HDF5_VER)
HDF5_BUILD_DIR := $(DEP_BUILD_DIR)/$(HDF5_NAME_VER)
HDF5_INSTALL_DIR := $(DEP_INSTALL_DIR)/hdf5-install
HDF5_LINK :=  https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$(HDF5_MAJ_MIN_VER)/$(HDF5_NAME_VER)/src/$(HDF5_NAME_VER).tar.gz
install-hdf5:
	@echo "Installing HDF5"
	rm -rf $(HDF5_BUILD_DIR) $(HDF5_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)
	cd $(DEP_BUILD_DIR) && curl -sL $(HDF5_LINK) | tar xz
	cd $(HDF5_BUILD_DIR) && ./configure --prefix=$(HDF5_INSTALL_DIR) --enable-optimization=high --enable-hl && make && make install
	rm -rf $(HDF5_BUILD_DIR)
	echo '$$(eval $$(call add-path,$(HDF5_INSTALL_DIR)))' >> Makefile.paths


# System Environment
ifdef LD_RUN_PATH
#CHPL_FLAGS += --ldflags="-Wl,-rpath=$(LD_RUN_PATH)"
# This pattern handles multiple paths separated by :
TEMP_FLAGS = $(patsubst %,--ldflags="-Wl+-rpath+%",$(strip $(subst :, ,$(LD_RUN_PATH))))
# The comma hack is necessary because commas can't appear in patsubst args
comma:= ,
CHPL_FLAGS += $(subst +,$(comma),$(TEMP_FLAGS))
endif

ifdef LD_LIBRARY_PATH
CHPL_FLAGS += $(patsubst %,-L%,$(strip $(subst :, ,$(LD_LIBRARY_PATH))))
endif

.PHONY: check-deps
ifndef ARKOUDA_SKIP_CHECK_DEPS
CHECK_DEPS = check-chpl check-zmq check-hdf5
endif
check-deps: $(CHECK_DEPS)

CHPL_MINOR := $(shell $(CHPL) --version | sed -n "s/chpl version 1\.\([0-9]*\).*/\1/p")
CHPL_VERSION_OK := $(shell test $(CHPL_MINOR) -ge 22 && echo yes)
CHPL_VERSION_WARN := $(shell test $(CHPL_MINOR) -le 22 && echo yes)
.PHONY: check-chpl
check-chpl:
ifneq ($(CHPL_VERSION_OK),yes)
	$(error Chapel 1.22.0 or newer is required)
endif
ifeq ($(CHPL_VERSION_WARN),yes)
	$(warning Chapel 1.23.0 or newer is recommended)
endif

CHPL_VERSION_122 := $(shell test $(CHPL_MINOR) -eq 22 && echo yes)
ifeq ($(CHPL_VERSION_122),yes)
CHPL_FLAGS += --instantiate-max 512
endif

ZMQ_CHECK = $(DEP_INSTALL_DIR)/checkZMQ.chpl
check-zmq: $(ZMQ_CHECK)
	@echo "Checking for ZMQ"
	$(CHPL) $(CHPL_FLAGS) $< -o $(DEP_INSTALL_DIR)/$@
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

HDF5_CHECK = $(DEP_INSTALL_DIR)/checkHDF5.chpl
check-hdf5: $(HDF5_CHECK)
	@echo "Checking for HDF5"
	$(CHPL) $(CHPL_FLAGS) $< -o $(DEP_INSTALL_DIR)/$@
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

ALL_TARGETS := $(ARKOUDA_MAIN_MODULE)
.PHONY: all
all: $(ALL_TARGETS)

Makefile.paths:
	@touch $@

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

  tags			Create developer TAGS file
  tags-clean

endef
$(eval $(call create_help_target,arkouda-help,ARKOUDA_HELP_TEXT))

# Set the arkouda server version from the VERSION file
VERSIONFILE=./VERSION
# Test for existence of VERSION file
ifneq ("$(wildcard $(VERSIONFILE))","")
	VERSION=$(shell cat ${VERSIONFILE})
else
	VERSION=$(shell date +'%Y.%m.%d')
endif

# Version needs to be escape-quoted for chpl to interpret as string
CHPL_FLAGS_WITH_VERSION = $(CHPL_FLAGS)
CHPL_FLAGS_WITH_VERSION += -sarkoudaVersion="\"$(VERSION)\""

ifdef ARKOUDA_PRINT_PASSES_FILE
	PRINT_PASSES_FLAGS := --print-passes-file $(ARKOUDA_PRINT_PASSES_FILE)
endif

ARKOUDA_SOURCES = $(shell find $(ARKOUDA_SOURCE_DIR)/ -type f -name '*.chpl')
ARKOUDA_MAIN_SOURCE := $(ARKOUDA_SOURCE_DIR)/$(ARKOUDA_MAIN_MODULE).chpl

$(ARKOUDA_MAIN_MODULE): check-deps $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES)
	$(CHPL) $(CHPL_DEBUG_FLAGS) $(PRINT_PASSES_FLAGS) $(CHPL_FLAGS_WITH_VERSION) $(ARKOUDA_MAIN_SOURCE) -o $@

CLEAN_TARGETS += arkouda-clean
.PHONY: arkouda-clean
arkouda-clean:
	$(RM) $(ARKOUDA_MAIN_MODULE) $(ARKOUDA_MAIN_MODULE)_real

.PHONY: tags
tags:
	-@(cd $(ARKOUDA_SOURCE_DIR) && $(CHPL_HOME)/util/chpltags -r . > /dev/null \
		&& echo "Updated $(ARKOUDA_SOURCE_DIR)/TAGS" \
		|| echo "Tags utility not available.  Skipping tags generation.")

CLEANALL_TARGETS += tags-clean
.PHONY: tags-clean
tags-clean:
	$(RM) $(ARKOUDA_SOURCE_DIR)/TAGS

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

DOC_DIR := docs
DOC_SERVER_OUTPUT_DIR := $(ARKOUDA_PROJECT_DIR)/$(DOC_DIR)/server
DOC_PYTHON_OUTPUT_DIR := $(ARKOUDA_PROJECT_DIR)/$(DOC_DIR)

DOC_COMPONENTS := \
	$(DOC_SERVER_OUTPUT_DIR) \
	$(DOC_PYTHON_OUTPUT_DIR)
$(DOC_COMPONENTS):
	mkdir -p $@

$(DOC_DIR):
	mkdir -p $@

.PHONY: doc
doc: doc-python doc-server 

CHPLDOC := chpldoc
CHPLDOC_FLAGS := --process-used-modules
.PHONY: doc-server
doc-server: ${DOC_DIR} $(DOC_SERVER_OUTPUT_DIR)/index.html
$(DOC_SERVER_OUTPUT_DIR)/index.html: $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES) | $(DOC_SERVER_OUTPUT_DIR)
	@echo "Building documentation for: Server"
	@# Build the documentation to the Chapel output directory
	$(CHPLDOC) $(CHPLDOC_FLAGS) $(ARKOUDA_MAIN_SOURCE) -o $(DOC_SERVER_OUTPUT_DIR)
	@# Create the .nojekyll file needed for github pages in the  Chapel output directory
	touch $(DOC_SERVER_OUTPUT_DIR)/.nojekyll
	@echo "Completed building documentation for: Server"

DOC_PYTHON_SOURCE_DIR := pydoc
DOC_PYTHON_SOURCES = $(shell find $(DOC_PYTHON_SOURCE_DIR)/ -type f)
.PHONY: doc-python
doc-python: ${DOC_DIR} $(DOC_PYTHON_OUTPUT_DIR)/index.html
$(DOC_PYTHON_OUTPUT_DIR)/index.html: $(DOC_PYTHON_SOURCES) $(ARKOUDA_MAKEFILES)
	@echo "Building documentation for: Python"
	$(eval $@_TMP := $(shell mktemp -d))
	@# Build the documentation to a temporary output directory.
	cd $(DOC_PYTHON_SOURCE_DIR) && $(MAKE) BUILDDIR=$($@_TMP) html
	@# Delete old python docs but retain Chapel docs in $(DOC_SERVER_OUTPUT_DIR).
	$(RM) -r docs/*html docs/*js docs/_static docs/_sources docs/autoapi docs/setup/ docs/usage docs/*inv
	@# Move newly-generated python docs including .nojekyll file needed for github pages.
	mv $($@_TMP)/html/* $($@_TMP)/html/.nojekyll $(DOC_PYTHON_OUTPUT_DIR)
	@# Remove temporary directory.
	$(RM) -r $($@_TMP)
	@# Remove server/index.html placeholder file to prepare for doc-server content
	$(RM) docs/server/index.html
	@echo "Completed building documentation for: Python"

CLEAN_TARGETS += doc-clean
.PHONY: doc-clean
doc-clean:
	$(RM) -r $(DOC_DIR)

check:
	@$(ARKOUDA_PROJECT_DIR)/util/test/checkInstall

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
test: test-python

.PHONY: test-chapel
test-chapel: $(TEST_TARGETS)

.PHONY: test-all
test-all: test-python test-chapel

mypy:
	python3 -m mypy arkouda

$(TEST_BINARY_DIR):
	mkdir -p $(TEST_BINARY_DIR)

.PHONY: $(TEST_TARGETS) # Force tests to always rebuild.
$(TEST_TARGETS): $(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL)%: $(TEST_SOURCE_DIR)/%.chpl | $(TEST_BINARY_DIR)
	$(CHPL) $(TEST_CHPL_FLAGS) -M $(ARKOUDA_SOURCE_DIR) $< -o $@

test/%: test/%.chpl
	$(CHPL) $@.chpl $(TEST_CHPL_FLAGS) -M $(ARKOUDA_SOURCE_DIR)

print-%:
	@echo "$($*)"

test-python: 
	python3 -m pytest -c pytest.ini

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
