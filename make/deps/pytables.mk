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


