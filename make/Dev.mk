
ruff-format:
	ruff format .
	ruff check $(ARKOUDA_PROJECT_DIR)/arkouda --fix
	#  Verify if it will pass the CI check:
	ruff format --check --diff

isort:
	ruff check --select I --fix .

format: ruff-format isort check-doc-examples
	#   Run flake8
	flake8 $(ARKOUDA_PROJECT_DIR)/arkouda
	flake8 $(ARKOUDA_PROJECT_DIR)/tests
	
docstr-coverage:
	#   Check coverage for doc strings:
	docstr-coverage arkouda --config .docstr.yaml


chplcheck:
	#   Check chapel linter, ignoring files in .chplcheckignore:
	find src -type f -name '*.chpl'   | grep -v -f .chplcheckignore   | xargs chplcheck --setting LineLength.Max=105 --add-rules src/scripts/chplcheck_ak_prefix.py --disable-rule CamelCaseFunctions  
	
	
COV_MIN ?= 100
.PHONY: coverage
coverage:
	python3 -m pytest -c pytest.ini  --cov=$(ARKOUDA_PROJECT_DIR)/arkouda --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=$(COV_MIN) --size=$(size) $(ARKOUDA_PYTEST_OPTIONS) --skip_doctest="True"
