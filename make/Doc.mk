################
#### Doc.mk ####
################

define DOC_HELP_TEXT
# doc			Generate $(DOC_DIR)/ with doc-* for server, etc.
  doc-help
  doc-clean
  doc-server
  doc-python
  stub-gen
  stub-clean

endef
$(eval $(call create_help_target,doc-help,DOC_HELP_TEXT))

DOC_DIR := docs
DOC_SERVER_OUTPUT_DIR := $(ARKOUDA_PROJECT_DIR)/$(DOC_DIR)/server
DOC_PYTHON_OUTPUT_DIR := $(ARKOUDA_PROJECT_DIR)/$(DOC_DIR)

DOC_COMPONENTS := \
	$(DOC_SERVER_OUTPUT_DIR) \
	$(DOC_PYTHON_OUTPUT_DIR)
$(DOC_COMPONENTS):
	mkdir -p $@

$(DOC_DIR):
	mkdir -p $@

.PHONY: doc
doc: stub-gen doc-python stub-clean doc-server

CHPLDOC := chpldoc
CHPLDOC_FLAGS := --process-used-modules
.PHONY: doc-server
doc-server: ${DOC_DIR} $(DOC_SERVER_OUTPUT_DIR)/index.html
$(DOC_SERVER_OUTPUT_DIR)/index.html: $(ARKOUDA_SOURCES) $(ARKOUDA_MAKEFILES) | $(DOC_SERVER_OUTPUT_DIR)
	@echo "Building documentation for: Server"
	@# Build the documentation to the Chapel output directory
	$(CHPLDOC) $(CHPLDOC_FLAGS) $(ARKOUDA_REGISTRY_DIR)/doc-support.chpl $(ARKOUDA_MAIN_SOURCE) $(ARKOUDA_SOURCE_DIR)/compat/eq-22/* -o $(DOC_SERVER_OUTPUT_DIR)
	@# Create the .nojekyll file needed for github pages in the  Chapel output directory
	touch $(DOC_SERVER_OUTPUT_DIR)/.nojekyll
	@echo "Completed building documentation for: Server"

DOC_PYTHON_SOURCE_DIR := pydoc
DOC_PYTHON_SOURCES = $(shell find $(DOC_PYTHON_SOURCE_DIR)/ -type f)
.PHONY: doc-python
doc-python: ${DOC_DIR} $(DOC_PYTHON_OUTPUT_DIR)/index.html
$(DOC_PYTHON_OUTPUT_DIR)/index.html: $(DOC_PYTHON_SOURCES) $(ARKOUDA_MAKEFILES)
	@echo "Building documentation for: Python"
	$(eval $@_TMP := $(shell mktemp -d))
	@# Build the documentation to a temporary output directory.
	cd $(DOC_PYTHON_SOURCE_DIR) && $(MAKE) BUILDDIR=$($@_TMP) html
	@# Delete old python docs but retain Chapel docs in $(DOC_SERVER_OUTPUT_DIR).
	$(RM) -r docs/*html docs/*js docs/_static docs/_sources docs/autoapi docs/setup/ docs/usage docs/*inv
	@# Move newly-generated python docs including .nojekyll file needed for github pages.
	mv $($@_TMP)/html/* $($@_TMP)/html/.nojekyll $(DOC_PYTHON_OUTPUT_DIR)
	@# Remove temporary directory.
	$(RM) -r $($@_TMP)
	@# Remove server/index.html placeholder file to prepare for doc-server content
	$(RM) docs/server/index.html
	@echo "Completed building documentation for: Python"

CLEAN_TARGETS += doc-clean
.PHONY: doc-clean
doc-clean: stub-clean
	$(RM) -r $(DOC_DIR)

check:
	@$(ARKOUDA_PROJECT_DIR)/server_util/test/checkInstall

.PHONY: check-doc-examples
check-doc-examples:
	@# skip binaries, only look in .py files, use extended regex
	@if grep -R -I -nE '^[[:space:]]*>>>[[:space:]]*#' \
	     --include="*.py" $(ARKOUDA_PROJECT_DIR)/arkouda; then \
	  echo "💥 Found comment-only doctest examples (>>> #…); please remove them"; \
	  exit 1; \
	fi
	
.PHONY: stub-gen stub-clean
	
stub-gen:
	python3 pydoc/preprocess/generate_import_stubs.py

stub-clean:
	find . -name "*.pyi" -type f -delete

