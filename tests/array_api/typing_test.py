class TestTyping:
    def test_typing_docstrings(self):
        import doctest

        from arkouda.array_api import _typing as typing_module

        result = doctest.testmod(
            typing_module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
