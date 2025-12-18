# add-path: Append custom paths for non-system software.
define add-path
# Always add headers for both C compilation and Chapel-generated C compilation
INCLUDE_FLAGS += -I$(1)/include
CHPL_FLAGS    += -I$(1)/include

# Add lib64 if present (independent of lib)
ifneq ("$(wildcard $(1)/lib64)","")
CHPL_FLAGS += -L$(1)/lib64 --ldflags="-Wl,-rpath,$(1)/lib64"
endif

# Add lib if present
ifneq ("$(wildcard $(1)/lib)","")
CHPL_FLAGS += -L$(1)/lib --ldflags="-Wl,-rpath,$(1)/lib"
endif
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

