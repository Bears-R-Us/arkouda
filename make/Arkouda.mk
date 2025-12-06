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

ifeq ($(shell expr $(CHPL_MINOR) \>= 4),1)
	ARKOUDA_COMPAT_MODULES_DIR = $(ARKOUDA_SOURCE_DIR)/compat/ge-24
	ARKOUDA_COMPAT_MODULES += -M $(ARKOUDA_COMPAT_MODULES_DIR)
endif

ifeq ($(shell expr $(CHPL_MINOR) \< 6),1)
	ARKOUDA_KEYPART_FLAG := -suseKeyPartStatus=true
endif

SERVER_CONFIG_SCRIPT=$(ARKOUDA_SOURCE_DIR)/parseServerConfig.py
# This is the main compilation statement section
$(ARKOUDA_MAIN_MODULE): check-deps register-commands $(ARROW_UTIL_O) $(ARROW_READ_O) $(ARROW_WRITE_O) $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES)
	$(eval MOD_GEN_OUT=$(shell python3 $(SERVER_CONFIG_SCRIPT) $(ARKOUDA_CONFIG_FILE) $(ARKOUDA_SOURCE_DIR)))

	$(CHPL) $(CHPL_DEBUG_FLAGS) $(CHPL_USER_FLAGS) $(PRINT_PASSES_FLAGS) $(REGEX_MAX_CAPTURES_FLAG) $(OPTIONAL_SERVER_FLAGS) $(CHPL_FLAGS_WITH_VERSION) $(CHPL_COMPAT_FLAGS) $(ARKOUDA_MAIN_SOURCE) $(ARKOUDA_COMPAT_MODULES) $(ARKOUDA_SERVER_USER_MODULES) $(MOD_GEN_OUT) $(ARKOUDA_RW_DEFAULT_FLAG) $(ARKOUDA_KEYPART_FLAG) $(ARKOUDA_REGISTRY_DIR)/Commands.chpl -I$(ARKOUDA_SOURCE_DIR)/parquet -o $@

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

