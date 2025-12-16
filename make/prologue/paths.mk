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

