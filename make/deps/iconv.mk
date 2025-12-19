ICONV_VER := 1.17
ICONV_NAME_VER := libiconv-$(ICONV_VER)
ICONV_BUILD_DIR := $(DEP_BUILD_DIR)/$(ICONV_NAME_VER)
ICONV_INSTALL_DIR := $(DEP_INSTALL_DIR)/libiconv-install
ICONV_LINK := https://ftp.gnu.org/pub/gnu/libiconv/libiconv-$(ICONV_VER).tar.gz

.PHONY: \
	iconv-download-source \
	install-iconv \
	iconv-clean


iconv-download-source:
	mkdir -p $(DEP_BUILD_DIR)

    #If the build directory does not exist,  create it
    ifeq (,$(wildcard ${ICONV_BUILD_DIR}*/.*))
        #   If the tar.gz not found, download it
        ifeq (,$(wildcard ${DEP_BUILD_DIR}/libiconv-${ICONV_VER}.tar.gz))
			cd $(DEP_BUILD_DIR) && curl -sL $(ICONV_LINK) | tar xz
        #   Otherwise just unzip it
        else
			cd $(DEP_BUILD_DIR) && tar -xzf libiconv-$(ICONV_VER).tar.gz
        endif
    endif

install-iconv: iconv-download-source
	@echo "Installing iconv"
	rm -rf $(ICONV_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)

	cd $(ICONV_BUILD_DIR) && ./configure --prefix=$(ICONV_INSTALL_DIR) && make && make install
	echo '$$(eval $$(call add-path,$(ICONV_INSTALL_DIR)))' >> Makefile.paths

iconv-clean:
	rm -rf $(ICONV_BUILD_DIR)

