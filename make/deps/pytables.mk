.PHONY: \
	pytables-download-source \
	install-pytables \
	pytables-clean

# Latest release as of 2026-03-04: 3.11.1
PYTABLES_VERSION ?= 3.11.1
PYTABLES_TAG     ?= v$(PYTABLES_VERSION)
PYTABLES_DIR     ?= $(DEP_BUILD_DIR)/PyTables

pytables-download-source:
	mkdir -p $(DEP_BUILD_DIR)
	@if [ ! -d "$(PYTABLES_DIR)/.git" ]; then \
		cd $(DEP_BUILD_DIR) && git clone https://github.com/PyTables/PyTables.git; \
	fi
	@cd $(PYTABLES_DIR) && \
		git fetch --tags --force && \
		git checkout -f $(PYTABLES_TAG) && \
		git submodule update --init --recursive

install-pytables: pytables-download-source
	@echo "Installing PyTables $(PYTABLES_VERSION) from tag $(PYTABLES_TAG)"
	cd $(DEP_BUILD_DIR) && python3 -m pip install ./PyTables

pytables-clean:
	rm -rf $(PYTABLES_DIR)
