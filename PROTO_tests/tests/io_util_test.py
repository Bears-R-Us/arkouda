import os
import shutil
from pathlib import Path

import pytest

from arkouda import io_util


class TestIOUtil:
    @classmethod
    def setup_class(cls):
        cls.io_test_dir = "{}/io_util_test/".format(os.getcwd())
        io_util.get_directory(cls.io_test_dir)

    def test_get_directory(self):
        assert dir
        Path.rmdir(Path(self.io_test_dir))
        assert not os.path.exists(self.io_test_dir)
        io_util.get_directory(self.io_test_dir)
        assert os.path.exists(self.io_test_dir)

    def test_write_line_to_file(self):
        io_util.write_line_to_file(
            path="{}/testfile.txt".format(self.io_test_dir), line="localhost:5555,9ty4h6olr4"
        )
        assert os.path.exists("{}/testfile.txt".format(self.io_test_dir))
        Path.unlink(Path("{}/testfile.txt".format(self.io_test_dir)))

    def test_delimited_file_to_dict(self):
        io_util.write_line_to_file(
            path="{}/testfile.txt".format(self.io_test_dir), line="localhost:5555,9ty4h6olr4"
        )
        io_util.write_line_to_file(
            path="{}/testfile.txt".format(self.io_test_dir), line="127.0.0.1:5556,6ky3i91l17"
        )
        values = io_util.delimited_file_to_dict(
            path="{}/testfile.txt".format(self.io_test_dir), delimiter=","
        )
        assert values
        assert "9ty4h6olr4" == values["localhost:5555"]
        assert "6ky3i91l17" == values["127.0.0.1:5556"]
        Path.unlink(Path("{}/testfile.txt".format(self.io_test_dir)))

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(Path(cls.io_test_dir))
