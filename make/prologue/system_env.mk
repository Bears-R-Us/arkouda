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
