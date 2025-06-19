import os
import tempfile

import pytest

from arkouda.pandas import io_util


class TestIOUtil:
    def test_io_util_docstrings(self):
        import doctest

        result = doctest.testmod(io_util, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @classmethod
    def setup_class(cls):
        cls.io_test_dir_base = f"{pytest.temp_directory}/io_test_dir"
        io_util.get_directory(cls.io_test_dir_base)

    def test_write_line_to_file(self):
        with tempfile.TemporaryDirectory(dir=self.io_test_dir_base) as tmp_dirname:
            test_file = f"{tmp_dirname}/testfile.txt"
            test_line = "localhost:5555,9ty4h6olr4"
            io_util.write_line_to_file(path=test_file, line=test_line)
            assert os.path.exists(test_file)
            assert open(test_file).read() == f"{test_line}\n"
            io_util.write_line_to_file(test_file, "line2")
            assert open(test_file).read() == f"{test_line}\nline2\n"

    def test_delimited_file_to_dict(self):
        with tempfile.TemporaryDirectory(dir=self.io_test_dir_base) as tmp_dirname:
            file_name = f"{tmp_dirname}/testfile.txt"
            values = {"localhost:5555": "9ty4h6olr4", "127.0.0.1:5556": "6ky3i91l17"}
            io_util.dict_to_delimited_file(file_name, values, ",")
            values_read = io_util.delimited_file_to_dict(path=file_name, delimiter=",")
            assert values_read == values

    def test_delete_directory(self):
        path = f"{pytest.temp_directory}/test_dir"
        io_util.get_directory(path)

        from os.path import isdir

        assert isdir(path) == True

        io_util.delete_directory(path)
        assert isdir(path) == False

        # Check no error when run on non-existant directory:
        io_util.delete_directory(path)

    def test_directory_exists(self):
        with tempfile.TemporaryDirectory(dir=self.io_test_dir_base) as tmp_dirname:
            assert io_util.directory_exists(tmp_dirname)
            assert not io_util.directory_exists(f"{tmp_dirname}/xyz")
