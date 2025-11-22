import platform

from os.path import expanduser

from arkouda import security


class TestSecurity:
    def test_security_docstrings(self):
        import doctest

        result = doctest.testmod(security, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    def test_generate_token(self):
        assert 32 == len(security.generate_token(32))
        assert 16 == len(security.generate_token(16))

    def test_get_home(self):
        assert expanduser("~") == security.get_home_directory()

    def test_get_username(self):
        assert security.get_username() in security.username_tokenizer[platform.system()](
            security.get_home_directory()
        )

    def test_get_arkouda_directory(self):
        ak_directory = security.get_arkouda_client_directory()
        assert f"{security.get_home_directory()}/.arkouda" == str(ak_directory)
