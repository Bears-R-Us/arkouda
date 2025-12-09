#####################
#### register.mk ####
#####################

register-commands:
	$(ARKOUDA_REGISTRY_DIR)/register_commands.bash $(ARKOUDA_REGISTRY_DIR) $(ARKOUDA_REGISTRATION_CONFIG) $(ARKOUDA_CONFIG_FILE) $(ARKOUDA_SOURCE_DIR)

