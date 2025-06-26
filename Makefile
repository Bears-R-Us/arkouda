# Makefile for Arkouda
ARKOUDA_PROJECT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ARKOUDA_PROJECT_DIR := $(patsubst %/,%,$(ARKOUDA_PROJECT_DIR))

PROJECT_NAME := arkouda
ARKOUDA_SOURCE_DIR := $(ARKOUDA_PROJECT_DIR)/src
ARKOUDA_REGISTRY_DIR=$(ARKOUDA_SOURCE_DIR)/registry
ARKOUDA_MAIN_MODULE := arkouda_server
ARKOUDA_MAKEFILES := Makefile Makefile.paths

DEFAULT_TARGET := $(ARKOUDA_MAIN_MODULE)
.PHONY: default
default: $(DEFAULT_TARGET)

VERBOSE ?= 0

CHPL := chpl

# We need to make the HDF5 API use the 1.10.x version for compatibility between 1.10 and 1.12
CHPL_FLAGS += --ccflags="-DH5_USE_110_API"

# silence warnings about '@arkouda' annotations being unrecognized
CHPL_FLAGS += --using-attribute-toolname arkouda

CHPL_DEBUG_FLAGS += --print-passes

ifdef ARKOUDA_DEVELOPER
ARKOUDA_QUICK_COMPILE = true
ARKOUDA_RUNTIME_CHECKS = true
endif

ifdef ARKOUDA_QUICK_COMPILE
CHPL_FLAGS += --no-checks --no-loop-invariant-code-motion --no-fast-followers --ccflags="-O0"
else
CHPL_FLAGS += --fast
endif

ifdef ARKOUDA_RUNTIME_CHECKS
CHPL_FLAGS += --checks
endif

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

# add-path: Append custom paths for non-system software.
# Note: Darwin `ld` only supports `-rpath <path>`, not `-rpath=<paths>`.
define add-path
ifneq ("$(wildcard $(1)/lib64)","")
  INCLUDE_FLAGS += -I$(1)/include -L$(1)/lib64
  CHPL_FLAGS    += -I$(1)/include -L$(1)/lib64 --ldflags="-Wl,-rpath,$(1)/lib64"
endif
INCLUDE_FLAGS += -I$(1)/include -L$(1)/lib
CHPL_FLAGS    += -I$(1)/include -L$(1)/lib --ldflags="-Wl,-rpath,$(1)/lib"
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
ifdef ARKOUDA_ARROW_PATH
$(eval $(call add-path,$(ARKOUDA_ARROW_PATH)))
endif
ifdef ARKOUDA_ICONV_PATH
$(eval $(call add-path,$(ARKOUDA_ICONV_PATH)))
endif
ifdef ARKOUDA_IDN2_PATH
$(eval $(call add-path,$(ARKOUDA_IDN2_PATH)))
endif

ifndef ARKOUDA_CONFIG_FILE
ARKOUDA_CONFIG_FILE := $(ARKOUDA_PROJECT_DIR)/ServerModules.cfg
endif

ifndef ARKOUDA_REGISTRATION_CONFIG
ARKOUDA_REGISTRATION_CONFIG := $(ARKOUDA_PROJECT_DIR)/registration-config.json
endif

CHPL_FLAGS += -lhdf5 -lhdf5_hl -lzmq -liconv -lidn2 -lparquet -larrow

ARROW_READ_FILE_NAME += $(ARKOUDA_SOURCE_DIR)/parquet/ReadParquet
ARROW_READ_CPP += $(ARROW_READ_FILE_NAME).cpp
ARROW_READ_H += $(ARROW_READ_FILE_NAME).h
ARROW_READ_O += $(ARKOUDA_SOURCE_DIR)/ReadParquet.o

ARROW_WRITE_FILE_NAME += $(ARKOUDA_SOURCE_DIR)/parquet/WriteParquet
ARROW_WRITE_CPP += $(ARROW_WRITE_FILE_NAME).cpp
ARROW_WRITE_H += $(ARROW_WRITE_FILE_NAME).h
ARROW_WRITE_O += $(ARKOUDA_SOURCE_DIR)/WriteParquet.o

ARROW_UTIL_FILE_NAME += $(ARKOUDA_SOURCE_DIR)/parquet/UtilParquet
ARROW_UTIL_CPP += $(ARROW_UTIL_FILE_NAME).cpp
ARROW_UTIL_H += $(ARROW_UTIL_FILE_NAME).h
ARROW_UTIL_O += $(ARKOUDA_SOURCE_DIR)/UtilParquet.o

.PHONY: install-deps
install-deps: install-zmq install-hdf5 install-arrow install-iconv install-idn2

.PHONY: deps-download-source
deps-download-source: zmq-download-source hdf5-download-source arrow-download-source iconv-download-source idn2-download-source

DEP_DIR := dep
DEP_INSTALL_DIR := $(ARKOUDA_PROJECT_DIR)/$(DEP_DIR)
DEP_BUILD_DIR := $(ARKOUDA_PROJECT_DIR)/$(DEP_DIR)/build

ZMQ_VER := 4.3.5
ZMQ_NAME_VER := zeromq-$(ZMQ_VER)
ZMQ_BUILD_DIR := $(DEP_BUILD_DIR)/$(ZMQ_NAME_VER)
ZMQ_INSTALL_DIR := $(DEP_INSTALL_DIR)/zeromq-install
ZMQ_LINK := https://github.com/zeromq/libzmq/releases/download/v$(ZMQ_VER)/$(ZMQ_NAME_VER).tar.gz

zmq-download-source:
	mkdir -p $(DEP_BUILD_DIR)
    #If the build directory does not exist,  create it
    ifeq (,$(wildcard ${ZMQ_BUILD_DIR}*/.*))
        #   If the tar.gz not found, download it
        ifeq (,$(wildcard ${DEP_BUILD_DIR}/${ZMQ_NAME_VER}*.tar.gz))
			cd $(DEP_BUILD_DIR) && curl -sL $(ZMQ_LINK) | tar xz
        #   Otherwise just unzip it
        else
			cd $(DEP_BUILD_DIR) && tar -xzf $(ZMQ_NAME_VER)*.tar.gz
        endif
    endif

install-zmq: zmq-download-source
	@echo "Installing ZeroMQ"
	rm -rf $(ZMQ_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)

	cd $(ZMQ_BUILD_DIR) && ./configure --prefix=$(ZMQ_INSTALL_DIR) CFLAGS=-O3 CXXFLAGS=-O3 && make && make install
	echo '$$(eval $$(call add-path,$(ZMQ_INSTALL_DIR)))' >> Makefile.paths

zmq-clean:
	rm -r $(ZMQ_BUILD_DIR)

HDF5_MAJ_MIN_VER := 1.14
HDF5_VER := 1.14.4
HDF5_NAME_VER := hdf5-$(HDF5_VER)

# new hdf5 path requires underscored delimited and "v" prepended
UNDERSCORED_LINK_HDF5_MAJ_MIN_VER := v1_14
UNDERSCORED_LINK_HDF5_VER := v1_14_4

HDF5_BUILD_DIR := $(DEP_BUILD_DIR)/$(HDF5_NAME_VER)
HDF5_INSTALL_DIR := $(DEP_INSTALL_DIR)/hdf5-install

# I think this seems good, but I don't love the hardcoded "-3" I'd like some input on that
HDF5_LINK := https://support.hdfgroup.org/releases/hdf5/$(UNDERSCORED_LINK_HDF5_MAJ_MIN_VER)/$(UNDERSCORED_LINK_HDF5_VER)/downloads/$(HDF5_NAME_VER)-3.tar.gz

hdf5-download-source:
	mkdir -p $(DEP_BUILD_DIR)
    #If the build directory does not exist,  create it
    ifeq (,$(wildcard ${HDF5_BUILD_DIR}*/.*))
        #   If the tar.gz not found, download it
        ifeq (,$(wildcard ${DEP_BUILD_DIR}/$(HDF5_NAME_VER)*tar.gz))
			cd $(DEP_BUILD_DIR) && curl -sL $(HDF5_LINK) | tar xz
        #   Otherwise just unzip it
        else
			cd $(DEP_BUILD_DIR) && tar -xzf $(HDF5_NAME_VER)*.tar.gz
        endif
    endif

install-hdf5: hdf5-download-source
	@echo "Installing HDF5"
	rm -rf $(HDF5_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)

	cd $(HDF5_BUILD_DIR)* && ./configure --prefix=$(HDF5_INSTALL_DIR) --enable-optimization=high --enable-hl && make && make install
	echo '$$(eval $$(call add-path,$(HDF5_INSTALL_DIR)))' >> Makefile.paths


hdf5-clean:
	rm -rf $(HDF5_BUILD_DIR)


install-arrow-quick:
	./scripts/install_arrow_quick.sh $(DEP_BUILD_DIR)


ARROW_VER := 19.0.1
ARROW_NAME_VER := apache-arrow-$(ARROW_VER)
ARROW_FULL_NAME_VER := arrow-apache-arrow-$(ARROW_VER)
ARROW_BUILD_DIR := $(DEP_BUILD_DIR)/$(ARROW_FULL_NAME_VER)
ARROW_DEP_DIR :=  $(DEP_BUILD_DIR)/arrow_dependencies
ARROW_INSTALL_DIR := $(DEP_INSTALL_DIR)/arrow-install
ARROW_SOURCE_LINK := https://github.com/apache/arrow/archive/refs/tags/$(ARROW_NAME_VER).tar.gz

NUM_CORES := $(shell nproc --all)

ARROW_DEPENDENCY_SOURCE := BUNDLED

arrow-download-source:
	mkdir -p $(DEP_BUILD_DIR)

    #   If the tar.gz file does not exist, fetch it
    ifeq (,$(wildcard ${DEP_BUILD_DIR}/$(ARROW_NAME_VER).tar.gz))
		cd $(DEP_BUILD_DIR) && wget $(ARROW_SOURCE_LINK)
    endif

	cd $(DEP_BUILD_DIR) && tar -xvf $(ARROW_NAME_VER).tar.gz

    # if the arrow dependency directory is empty of tar.gz, download the dependencies
    ifeq (,$(wildcard $(ARROW_DEP_DIR)/*.tar.gz))
		rm -fr $(DEP_BUILD_DIR)/arrow_exports.sh
		mkdir -p $(ARROW_DEP_DIR)
		cd $(ARROW_BUILD_DIR)/cpp/thirdparty/ && ./download_dependencies.sh $(ARROW_DEP_DIR) > $(DEP_BUILD_DIR)/arrow_exports.sh
    endif

	rm -fr $(ARROW_BUILD_DIR)

install-arrow: arrow-download-source
	@echo "Installing Apache Arrow/Parquet"
	@echo "from build directory: ${DEP_BUILD_DIR}"
	rm -rf $(ARROW_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)

	cd $(DEP_BUILD_DIR) && tar -xvf $(ARROW_NAME_VER).tar.gz
	mkdir -p $(ARROW_BUILD_DIR)/cpp/build-release

	cd $(DEP_BUILD_DIR) && . ./arrow_exports.sh && cd $(ARROW_BUILD_DIR)/cpp/build-release && cmake -S $(ARROW_BUILD_DIR)/cpp .. -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_INSTALL_PREFIX=$(ARROW_INSTALL_DIR) -DCMAKE_BUILD_TYPE=Release -DARROW_PARQUET=ON -DARROW_WITH_SNAPPY=ON -DARROW_WITH_BROTLI=ON -DARROW_WITH_BZ2=ON -DARROW_WITH_LZ4=ON -DARROW_WITH_ZLIB=ON -DARROW_WITH_ZSTD=ON -DARROW_DEPENDENCY_SOURCE=$(ARROW_DEPENDENCY_SOURCE) $(ARROW_OPTIONS) && make -j$(NUM_CORES)

	cd $(ARROW_BUILD_DIR)/cpp/build-release && make install

	echo '$$(eval $$(call add-path,$(ARROW_INSTALL_DIR)))' >> Makefile.paths

arrow-clean:
	rm -rf $(DEP_BUILD_DIR)/apache-arrow*
	rm -rf $(DEP_BUILD_DIR)/arrow-apache-arrow*
	rm -rf $(ARROW_DEP_DIR)
	rm -fr $(DEP_BUILD_DIR)/arrow_exports.sh

pytables-download-source:
	mkdir -p $(DEP_BUILD_DIR)

    #   If the PyTables directory does not exist, fetch it
    ifeq (,$(wildcard ${DEP_BUILD_DIR}/PyTables))
		cd $(DEP_BUILD_DIR) && git clone https://github.com/PyTables/PyTables.git
		cd $(DEP_BUILD_DIR)/PyTables && git submodule update --init --recursive
    endif

install-pytables: pytables-download-source
	@echo "Installing PyTables"
	cd $(DEP_BUILD_DIR) && python3 -m pip install PyTables/

pytables-clean:
	rm -rf $(DEP_BUILD_DIR)/PyTables


ICONV_VER := 1.17
ICONV_NAME_VER := libiconv-$(ICONV_VER)
ICONV_BUILD_DIR := $(DEP_BUILD_DIR)/$(ICONV_NAME_VER)
ICONV_INSTALL_DIR := $(DEP_INSTALL_DIR)/libiconv-install
ICONV_LINK := https://ftp.gnu.org/pub/gnu/libiconv/libiconv-$(ICONV_VER).tar.gz

iconv-download-source:
	mkdir -p $(DEP_BUILD_DIR)

    #If the build directory does not exist,  create it
    ifeq (,$(wildcard ${ICONV_BUILD_DIR}*/.*))
        #   If the tar.gz not found, download it
        ifeq (,$(wildcard ${DEP_BUILD_DIR}/libiconv-${ICONV_VER}.tar.gz))
			cd $(DEP_BUILD_DIR) && curl -sL $(ICONV_LINK) | tar xz
        #   Otherwise just unzip it
        else
			cd $(DEP_BUILD_DIR) && tar -xzf libiconv-$(ICONV_VER).tar.gz
        endif
    endif

install-iconv: iconv-download-source
	@echo "Installing iconv"
	rm -rf $(ICONV_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)

	cd $(ICONV_BUILD_DIR) && ./configure --prefix=$(ICONV_INSTALL_DIR) && make && make install
	echo '$$(eval $$(call add-path,$(ICONV_INSTALL_DIR)))' >> Makefile.paths

iconv-clean:
	rm -rf $(ICONV_BUILD_DIR)

LIBIDN_VER := 2.3.4
LIBIDN_NAME_VER := libidn2-$(LIBIDN_VER)
LIBIDN_BUILD_DIR := $(DEP_BUILD_DIR)/$(LIBIDN_NAME_VER)
LIBIDN_INSTALL_DIR := $(DEP_INSTALL_DIR)/libidn2-install
LIBIDN_LINK := https://ftp.gnu.org/gnu/libidn/libidn2-$(LIBIDN_VER).tar.gz

idn2-download-source:
	mkdir -p $(DEP_BUILD_DIR)

    #If the build directory does not exist,  create it
    ifeq (,$(wildcard $(LIBIDN_BUILD_DIR)*/.*))
        # If the tar.gz is not found, download it
        ifeq (,$(wildcard ${DEP_BUILD_DIR}/libidn2-$(LIBIDN_VER)*.tar.gz))
			cd $(DEP_BUILD_DIR) && curl -sL $(LIBIDN_LINK) | tar xz
        # Otherwise just unzip it
        else
			cd $(DEP_BUILD_DIR) && tar -xzf libidn2-$(LIBIDN_VER)*.tar.gz
        endif
    endif

install-idn2: idn2-download-source
	@echo "Installing libidn2"
	rm -rf $(LIBIDN_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)

	cd $(LIBIDN_BUILD_DIR) && ./configure --prefix=$(LIBIDN_INSTALL_DIR) && make && make install
	echo '$$(eval $$(call add-path,$(LIBIDN_INSTALL_DIR)))' >> Makefile.paths

idn2-clean:
	rm -rf $(LIBIDN_BUILD_DIR)

BLOSC_BUILD_DIR := $(DEP_BUILD_DIR)/c-blosc2
BLOSC_INSTALL_DIR := $(DEP_INSTALL_DIR)/c-blosc-install

blosc-download-source:
	mkdir -p $(DEP_BUILD_DIR)

    #If the build directory does not exist,  create it
    ifeq (,$(wildcard $(BLOSC_BUILD_DIR)/.*))
		cd $(DEP_BUILD_DIR) && git clone https://github.com/Blosc/c-blosc2.git
    endif

install-blosc: blosc-download-source
	@echo "Installing blosc"
	rm -rf $(BLOSC_INSTALL_DIR)
	mkdir -p $(BLOSC_INSTALL_DIR)

	cd $(BLOSC_BUILD_DIR) && cmake -DCMAKE_INSTALL_PREFIX=$(BLOSC_INSTALL_DIR) && make && make install
	echo '$$(eval $$(call add-path,$(BLOSC_INSTALL_DIR)))' >> Makefile.paths

blosc-clean:
	rm -rf $(BLOSC_BUILD_DIR)

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

#
# Flags for Python interop: this is required for ApplyMsg to call into the Python interpreter
#
# Adds the include path for the Python.h header file
CHPL_FLAGS += --ccflags -isystem$(shell python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
# Adds the library path for the Python shared library
CHPL_FLAGS += -L$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
CHPL_FLAGS += --ldflags -Wl,-rpath,$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
# Adds the Python shared library to the list of libraries to link against
CHPL_FLAGS += -lpython$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
# Ignore warnings from the Python headers. This is irrelevant for newer Python versions
CHPL_FLAGS += --ccflags -Wno-macro-redefined

PYTHON_VERSION := $(shell python3 -c "import sys; print(*sys.version_info[:2], sep='.')")


.PHONY: check-deps
ifndef ARKOUDA_SKIP_CHECK_DEPS
CHECK_DEPS = check-chpl check-zmq check-hdf5 check-re2 check-arrow check-iconv check-idn2
endif
check-deps: $(CHECK_DEPS)

ARKOUDA_CHPL_HOME=$(shell $(CHPL) --print-chpl-home 2>/dev/null)

SANITIZER = $(shell $(ARKOUDA_CHPL_HOME)/util/chplenv/chpl_sanitizers.py --exe 2>/dev/null)
ifneq ($(SANITIZER),none)
ARROW_SANITIZE=-fsanitize=$(SANITIZER)
endif

CHPL_CXX = $(shell $(ARKOUDA_CHPL_HOME)/util/config/compileline --compile-c++ 2>/dev/null)
ifeq ($(CHPL_CXX),)
CHPL_CXX=$(CXX)
endif

.PHONY: compile-arrow-cpp
compile-arrow-cpp:
	make compile-arrow-write
	make compile-arrow-read
	make compile-arrow-util

.PHONY: compile-arrow-write
compile-arrow-write:
	$(CHPL_CXX) -O3 -std=c++17 -c $(ARROW_WRITE_CPP) -o $(ARROW_WRITE_O) $(INCLUDE_FLAGS) $(ARROW_SANITIZE)

.PHONY: compile-arrow-read
compile-arrow-read:
	$(CHPL_CXX) -O3 -std=c++17 -c $(ARROW_READ_CPP) -o $(ARROW_READ_O) $(INCLUDE_FLAGS) $(ARROW_SANITIZE)

.PHONY: compile-arrow-util
compile-arrow-util:
	$(CHPL_CXX) -O3 -std=c++17 -c $(ARROW_UTIL_CPP) -o $(ARROW_UTIL_O) $(INCLUDE_FLAGS) $(ARROW_SANITIZE)

$(ARROW_UTIL_O): $(ARROW_UTIL_CPP) $(ARROW_UTIL_H)
	make compile-arrow-util

$(ARROW_READ_O): $(ARROW_READ_CPP) $(ARROW_READ_H)
	make compile-arrow-read

$(ARROW_WRITE_O): $(ARROW_WRITE_CPP) $(ARROW_WRITE_H)
	make compile-arrow-write

CHPL_MAJOR := $(shell $(CHPL) --version | sed -n "s/chpl version \([0-9]\)\.[0-9]*.*/\1/p")
CHPL_MINOR := $(shell $(CHPL) --version | sed -n "s/chpl version [0-9]\.\([0-9]*\).*/\1/p")
CHPL_VERSION_OK := $(shell test $(CHPL_MAJOR) -ge 2 -o $(CHPL_MINOR) -ge 0  && echo yes)
# CHPL_VERSION_WARN := $(shell test $(CHPL_MAJOR) -eq 1 -a $(CHPL_MINOR) -le 33 && echo yes)
.PHONY: check-chpl
check-chpl:
ifneq ($(CHPL_VERSION_OK),yes)
	$(error Chapel 2.0 or newer is required, found $(CHPL_MAJOR).$(CHPL_MINOR))
endif
# ifeq ($(CHPL_VERSION_WARN),yes)
# 	$(warning Chapel 1.33.0 or newer is recommended, found $(CHPL_MAJOR).$(CHPL_MINOR))
# endif

ZMQ_CHECK = $(DEP_INSTALL_DIR)/checkZMQ.chpl
check-zmq: $(ZMQ_CHECK)
	@echo "Checking for ZMQ"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

HDF5_CHECK = $(DEP_INSTALL_DIR)/checkHDF5.chpl
check-hdf5: $(HDF5_CHECK)
	@echo "Checking for HDF5"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

RE2_CHECK = $(DEP_INSTALL_DIR)/checkRE2.chpl
check-re2: $(RE2_CHECK)
	@echo "Checking for RE2"
	@$(CHPL) $(CHPL_FLAGS) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

ARROW_CHECK = $(DEP_INSTALL_DIR)/checkArrow.chpl
check-arrow: $(ARROW_CHECK) $(ARROW_UTIL_O) $(ARROW_READ_O) $(ARROW_WRITE_O)
	@echo "Checking for Arrow"
	make compile-arrow-cpp
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< $(ARROW_M) -M $(ARKOUDA_SOURCE_DIR) -I $(ARKOUDA_SOURCE_DIR)/parquet -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

ICONV_CHECK = $(DEP_INSTALL_DIR)/checkIconv.chpl
check-iconv: $(ICONV_CHECK)
	@echo "Checking for iconv"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) -M $(ARKOUDA_SOURCE_DIR) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

IDN2_CHECK = $(DEP_INSTALL_DIR)/checkIdn2.chpl
check-idn2: $(IDN2_CHECK)
	@echo "Checking for idn2"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) -M $(ARKOUDA_SOURCE_DIR) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/master/pydoc/setup/BUILD.md)\n"
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
VERSION=$(shell python3 -c "import versioneer; print(versioneer.get_versions()[\"version\"])")
# Test for existence of VERSION file
# ifneq ("$(wildcard $(VERSIONFILE))","")
# 	VERSION=$(shell cat ${VERSIONFILE})
# else
# 	VERSION=$(shell date +'%Y.%m.%d')
# endif

# Version needs to be escape-quoted for chpl to interpret as string
CHPL_FLAGS_WITH_VERSION = $(CHPL_FLAGS)
CHPL_FLAGS_WITH_VERSION += -sarkoudaVersion="\"$(VERSION)\"" -spythonVersion="\""$(PYTHON_VERSION)"\""

ifdef ARKOUDA_PRINT_PASSES_FILE
	PRINT_PASSES_FLAGS := --print-passes-file $(ARKOUDA_PRINT_PASSES_FILE)
endif

ifdef REGEX_MAX_CAPTURES
	REGEX_MAX_CAPTURES_FLAG = -sregexMaxCaptures=$(REGEX_MAX_CAPTURES)
endif

ARKOUDA_SOURCES = $(shell find $(ARKOUDA_SOURCE_DIR)/ -type f -name '*.chpl')
ARKOUDA_MAIN_SOURCE := $(ARKOUDA_SOURCE_DIR)/$(ARKOUDA_MAIN_MODULE).chpl

ifeq ($(shell expr $(CHPL_MINOR) \= 0),1)
	ARKOUDA_COMPAT_MODULES += -M $(ARKOUDA_SOURCE_DIR)/compat/eq-20
endif

ifeq ($(shell expr $(CHPL_MINOR) \= 1),1)
	ARKOUDA_COMPAT_MODULES += -M $(ARKOUDA_SOURCE_DIR)/compat/eq-21
endif

ifeq ($(shell expr $(CHPL_MINOR) \= 2),1)
	ARKOUDA_COMPAT_MODULES += -M $(ARKOUDA_SOURCE_DIR)/compat/eq-22
endif

ifeq ($(shell expr $(CHPL_MINOR) \= 3),1)
	ARKOUDA_COMPAT_MODULES += -M $(ARKOUDA_SOURCE_DIR)/compat/eq-23
endif

ifeq ($(shell expr $(CHPL_MINOR) \>= 4),1)
	ARKOUDA_COMPAT_MODULES += -M $(ARKOUDA_SOURCE_DIR)/compat/ge-24
endif

ifeq ($(shell expr $(CHPL_MINOR) \>= 2),1)
	ARKOUDA_KEYPART_FLAG := -suseKeyPartStatus=true
endif

ifeq ($(shell expr $(CHPL_MINOR) \<= 1),1)
	ARKOUDA_RW_DEFAULT_FLAG := -sOpenReaderLockingDefault=false -sOpenWriterLockingDefault=false
endif

SERVER_CONFIG_SCRIPT=$(ARKOUDA_SOURCE_DIR)/parseServerConfig.py
# This is the main compilation statement section
$(ARKOUDA_MAIN_MODULE): check-deps register-commands $(ARROW_UTIL_O) $(ARROW_READ_O) $(ARROW_WRITE_O) $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES)
	$(eval MOD_GEN_OUT=$(shell python3 $(SERVER_CONFIG_SCRIPT) $(ARKOUDA_CONFIG_FILE) $(ARKOUDA_SOURCE_DIR)))

	$(CHPL) $(CHPL_DEBUG_FLAGS) $(PRINT_PASSES_FLAGS) $(REGEX_MAX_CAPTURES_FLAG) $(OPTIONAL_SERVER_FLAGS) $(CHPL_FLAGS_WITH_VERSION) $(CHPL_COMPAT_FLAGS) $(ARKOUDA_MAIN_SOURCE) $(ARKOUDA_COMPAT_MODULES) $(ARKOUDA_SERVER_USER_MODULES) $(MOD_GEN_OUT) $(ARKOUDA_RW_DEFAULT_FLAG) $(ARKOUDA_KEYPART_FLAG) $(ARKOUDA_REGISTRY_DIR)/Commands.chpl -I$(ARKOUDA_SOURCE_DIR)/parquet -o $@

CLEAN_TARGETS += arkouda-clean
.PHONY: arkouda-clean
arkouda-clean:
	$(RM) $(ARKOUDA_MAIN_MODULE) $(ARKOUDA_MAIN_MODULE)_real $(ARROW_UTIL_O) $(ARROW_READ_O) $(ARROW_WRITE_O)

.PHONY: tags
tags:
	-@(cd $(ARKOUDA_SOURCE_DIR) && $(ARKOUDA_CHPL_HOME)/util/chpltags -r . > /dev/null \
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
doc: stub-gen doc-python stub-clean doc-server

CHPLDOC := chpldoc
CHPLDOC_FLAGS := --process-used-modules
.PHONY: doc-server
doc-server: ${DOC_DIR} $(DOC_SERVER_OUTPUT_DIR)/index.html
$(DOC_SERVER_OUTPUT_DIR)/index.html: $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES) | $(DOC_SERVER_OUTPUT_DIR)
	@echo "Building documentation for: Server"
	@# Build the documentation to the Chapel output directory
	$(CHPLDOC) $(CHPLDOC_FLAGS) $(ARKOUDA_REGISTRY_DIR)/doc-support.chpl $(ARKOUDA_MAIN_SOURCE) $(ARKOUDA_SOURCE_DIR)/compat/eq-22/* -o $(DOC_SERVER_OUTPUT_DIR)
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
doc-clean: stub-clean
	$(RM) -r $(DOC_DIR)

check:
	@$(ARKOUDA_PROJECT_DIR)/server_util/test/checkInstall

ruff-format:
	ruff format .
	#  Verify if it will pass the CI check:
	ruff format --check --diff

isort:
	isort --gitignore --float-to-top .
	#  Verify if it will pass the CI check:
	isort --check-only --diff .


.PHONY: check-doc-examples
check-doc-examples:
	@# skip binaries, only look in .py files, use extended regex
	@if grep -R -I -nE '^[[:space:]]*>>>[[:space:]]*#' \
	     --include="*.py" $(ARKOUDA_PROJECT_DIR)/arkouda; then \
	  echo "💥 Found comment-only doctest examples (>>> #…); please remove them"; \
	  exit 1; \
	fi
	
	
format: isort ruff-format check-doc-examples
	#   Run docstring linter
	pydocstyle
	#   Run flake8
	flake8 --config=$(ARKOUDA_PROJECT_DIR)/setup.cfg $(ARKOUDA_PROJECT_DIR)/arkouda

darglint: 
	#   Check darglint linter for doc strings:
	darglint -v 2 arkouda
	
docstr-coverage:
	#   Check coverage for doc strings:
	docstr-coverage arkouda --config .docstr.yaml


chplcheck:
	#   Check chapel linter, ignoring files in .chplcheckignore:
	find src -type f -name '*.chpl'   | grep -v -f .chplcheckignore   | xargs chplcheck --setting LineLength.Max=105 --add-rules src/scripts/chplcheck_ak_prefix.py --disable-rule CamelCaseFunctions  

#################
#### Test.mk ####
#################

TEST_SOURCE_DIR := tests/server
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
test-chapel:
	start_test $(TEST_SOURCE_DIR)

.PHONY: test-all
test-all: test-python test-chapel

mypy:
	python3 -m mypy arkouda

$(TEST_BINARY_DIR):
	mkdir -p $(TEST_BINARY_DIR)

.PHONY: $(TEST_TARGETS) # Force tests to always rebuild.
$(TEST_TARGETS): $(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL)%: $(TEST_SOURCE_DIR)/%.chpl | $(TEST_BINARY_DIR)
	$(CHPL) $(TEST_CHPL_FLAGS) -M $(ARKOUDA_SOURCE_DIR) $(ARKOUDA_COMPAT_MODULES) $< -o $@

print-%:
	$(info $($*)) @trues

size=100
skip_doctest="False"
test-python:
	python3 -m pytest -c pytest.ini --size=$(size) $(ARKOUDA_PYTEST_OPTIONS) --skip_doctest=$(skip_doctest) --html=.pytest/report.html --self-contained-html
	python3 -m pytest -c pytest.opts.ini --size=$(size) $(ARKOUDA_PYTEST_OPTIONS)

CLEAN_TARGETS += test-clean
.PHONY: test-clean
test-clean:
	$(RM) $(TEST_TARGETS) $(addsuffix _real,$(TEST_TARGETS))

size_bm = 10**8
DATE := $(shell date '+%Y_%m_%d_%H_%M_%S')
out=benchmark_v2/data/benchmark_stats_$(DATE).json
.PHONY: benchmark
benchmark:
	mkdir -p benchmark_v2/data
	python3 -m pytest -c benchmark.ini --benchmark-autosave --benchmark-storage=file://benchmark_v2/.benchmarks --size=$(size_bm) --benchmark-json=$(out)
	python3 benchmark_v2/reformat_benchmark_results.py --benchmark-data $(out)

version:
	@echo $(VERSION);


#####################
#### register.mk ####
#####################

register-commands:
	$(ARKOUDA_REGISTRY_DIR)/register_commands.bash $(ARKOUDA_REGISTRY_DIR) $(ARKOUDA_REGISTRATION_CONFIG) $(ARKOUDA_CONFIG_FILE) $(ARKOUDA_SOURCE_DIR)

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

stub-gen:
	python3 pydoc/preprocess/generate_import_stubs.py

stub-clean:
	find . -name "*.pyi" -type f -delete
