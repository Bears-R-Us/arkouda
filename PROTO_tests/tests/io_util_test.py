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
            io_util.write_line_to_file(
                path="{}/testfile.txt".format(tmp_dirname), line="localhost:5555,9ty4h6olr4"
            )
            io_util.write_line_to_file(
                path="{}/testfile.txt".format(tmp_dirname), line="127.0.0.1:5556,6ky3i91l17"
            )
            values = io_util.delimited_file_to_dict(
                path="{}/testfile.txt".format(tmp_dirname), delimiter=","
            )
            assert values
            assert "9ty4h6olr4" == values["localhost:5555"]
            assert "6ky3i91l17" == values["127.0.0.1:5556"]
