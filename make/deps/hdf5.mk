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


