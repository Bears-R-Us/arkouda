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

