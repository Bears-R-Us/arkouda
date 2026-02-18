class TestConstants:
    def test_constants_docstrings(self):
        import doctest

        from arkouda.array_api import _constants

        result = doctest.testmod(_constants, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
