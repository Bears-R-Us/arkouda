class TestInfoClass:
    def test_infoclass_docstrings(self):
        import doctest

        from arkouda import infoclass

        result = doctest.testmod(infoclass, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
