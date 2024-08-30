import os
import tempfile

from arkouda import io_util


class TestIOUtil:
    @classmethod
    def setup_class(cls):
        cls.io_test_dir_base = f"{os.getcwd()}/io_test_dir"
        io_util.get_directory(cls.io_test_dir_base)

    def test_write_line_to_file(self):
        with tempfile.TemporaryDirectory(dir=self.io_test_dir_base) as tmp_dirname:
            io_util.write_line_to_file(
                path=f"{tmp_dirname}/testfile.txt", line="localhost:5555,9ty4h6olr4"
            )
            assert os.path.exists(f"{tmp_dirname}/testfile.txt")

    def test_delimited_file_to_dict(self):
        with tempfile.TemporaryDirectory(dir=self.io_test_dir_base) as tmp_dirname:
            file_name = f"{tmp_dirname}/testfile.txt"
            io_util.write_line_to_file(path=file_name, line="localhost:5555,9ty4h6olr4")
            io_util.write_line_to_file(path=file_name, line="127.0.0.1:5556,6ky3i91l17")
            values = io_util.delimited_file_to_dict(path=file_name, delimiter=",")
            assert values
            assert "9ty4h6olr4" == values["localhost:5555"]
            assert "6ky3i91l17" == values["127.0.0.1:5556"]

    def test_delete_directory(self):
        path = "{}/test_dir".format(os.getcwd())
        io_util.get_directory(path)

        from os.path import isdir

        assert isdir(path) == True

        io_util.delete_directory(path)
        assert isdir(path) == False

        # Check no error when run on non-existant directory:
        io_util.delete_directory(path)
