# Makefile for Arkouda
ARKOUDA_PROJECT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ARKOUDA_PROJECT_DIR := $(patsubst %/,%,$(ARKOUDA_PROJECT_DIR))

PROJECT_NAME := arkouda
ARKOUDA_SOURCE_DIR := $(ARKOUDA_PROJECT_DIR)/src
ARKOUDA_REGISTRY_DIR=$(ARKOUDA_SOURCE_DIR)/registry
ARKOUDA_MAIN_MODULE := arkouda_server
ARKOUDA_MAKEFILES := Makefile Makefile.paths $(wildcard make/*.mk)

DEFAULT_TARGET := $(ARKOUDA_MAIN_MODULE)
.PHONY: default
default: $(DEFAULT_TARGET)

VERBOSE ?= 0

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

# add-path: Append custom paths for non-system software.
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

# Allows the user to set a different instantiation limit
ifneq ($(instantiate_max),)
CHPL_USER_FLAGS += --instantiate-max=$(instantiate_max)
endif


PYTHON_VERSION := $(shell python3 -c "import sys; print(*sys.version_info[:2], sep='.')")


.PHONY: check-deps
ifndef ARKOUDA_SKIP_CHECK_DEPS
CHECK_DEPS = check-chpl check-zmq check-hdf5 check-re2 check-arrow check-iconv check-idn2
endif
check-deps: $(CHECK_DEPS)

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
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

HDF5_CHECK = $(DEP_INSTALL_DIR)/checkHDF5.chpl
check-hdf5: $(HDF5_CHECK)
	@echo "Checking for HDF5"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

RE2_CHECK = $(DEP_INSTALL_DIR)/checkRE2.chpl
check-re2: $(RE2_CHECK)
	@echo "Checking for RE2"
	@$(CHPL) $(CHPL_FLAGS) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

ARROW_CHECK = $(DEP_INSTALL_DIR)/checkArrow.chpl
check-arrow: $(ARROW_CHECK) $(ARROW_UTIL_O) $(ARROW_READ_O) $(ARROW_WRITE_O)
	@echo "Checking for Arrow"
	make compile-arrow-cpp
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< $(ARROW_M) -M $(ARKOUDA_SOURCE_DIR) -I $(ARKOUDA_SOURCE_DIR)/parquet -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

ICONV_CHECK = $(DEP_INSTALL_DIR)/checkIconv.chpl
check-iconv: $(ICONV_CHECK)
	@echo "Checking for iconv"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) -M $(ARKOUDA_SOURCE_DIR) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

IDN2_CHECK = $(DEP_INSTALL_DIR)/checkIdn2.chpl
check-idn2: $(IDN2_CHECK)
	@echo "Checking for idn2"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) -M $(ARKOUDA_SOURCE_DIR) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
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

