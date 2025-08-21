class TestArkoudaBaseExtension:
    def test_base_extension_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _arkouda_base_array

        result = doctest.testmod(
            _arkouda_base_array, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
