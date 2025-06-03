class TestDataTypeFunctions:
    def test_dtypes_docstrings(self):
        import doctest

        from arkouda.array_api import _dtypes

        result = doctest.testmod(_dtypes, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_data_type_functions_docstrings(self):
        import doctest

        from arkouda.array_api import data_type_functions

        result = doctest.testmod(
            data_type_functions, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
