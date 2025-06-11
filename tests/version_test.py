class TestVersion:
    def test_version_docstrings(self):
        import doctest

        from arkouda import _version

        result = doctest.testmod(_version, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
