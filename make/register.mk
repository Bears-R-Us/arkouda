#####################
#### register.mk ####
#####################

# All Chapel sources under ARKOUDA_SOURCE_DIR; changes here re-trigger registration.
ARKOUDA_CHPL_SOURCES := $(shell find $(ARKOUDA_SOURCE_DIR) -name '*.chpl')

.PHONY: register-commands

register-commands: $(ARKOUDA_CONFIG_FILE) \
                   $(ARKOUDA_REGISTRATION_CONFIG) \
                   $(ARKOUDA_REGISTRY_DIR)/register_commands.bash \
                   $(ARKOUDA_CHPL_SOURCES)
	$(ARKOUDA_REGISTRY_DIR)/register_commands.bash \
	    $(ARKOUDA_REGISTRY_DIR) \
	    $(ARKOUDA_REGISTRATION_CONFIG) \
	    $(ARKOUDA_CONFIG_FILE) \
	    $(ARKOUDA_SOURCE_DIR)


