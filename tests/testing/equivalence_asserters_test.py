class TestEquivalenceAsserters:
    def test_equivalence_asserters_docstrings(self):
        import doctest

        from arkouda.testing import _equivalence_asserters

        result = doctest.testmod(
            _equivalence_asserters, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"
