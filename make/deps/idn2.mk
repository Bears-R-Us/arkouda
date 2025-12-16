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

