class TestArkoudaStringsExtension:
    def test_strings_extension_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _arkouda_string_array

        result = doctest.testmod(
            _arkouda_string_array, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
