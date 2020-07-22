import unittest, os, sys, shutil
from pathlib import Path
from context import arkouda as ak
from arkouda import io_util
from base_test import ArkoudaTest

class IOUtilTest(ArkoudaTest):

    @classmethod
    def setUpClass(cls):
        super(IOUtilTest, cls).setUpClass()
        IOUtilTest.io_test_dir = '{}/io_util_test/'.format(os.getcwd())
        io_util.get_directory(IOUtilTest.io_test_dir)

    def testGetDirectory(self):
        dir = io_util.get_directory(IOUtilTest.io_test_dir)
        self.assertTrue(dir)
        Path.rmdir(Path(IOUtilTest.io_test_dir))
        self.assertFalse(os.path.exists(IOUtilTest.io_test_dir))
        io_util.get_directory(IOUtilTest.io_test_dir)
        self.assertTrue(os.path.exists(IOUtilTest.io_test_dir))

    def testWriteLineToFile(self):
        io_util.write_line_to_file(path='{}/testfile.txt'.format(IOUtilTest.io_test_dir),
                                       line='localhost:5555,9ty4h6olr4')
        self.assertTrue(os.path.exists('{}/testfile.txt'.format(IOUtilTest.io_test_dir)))
        Path.unlink(Path('{}/testfile.txt'.format(IOUtilTest.io_test_dir)))

    def testDelimitedFileToDict(self):
        io_util.write_line_to_file(path='{}/testfile.txt'.format(IOUtilTest.io_test_dir),
                                       line='localhost:5555,9ty4h6olr4')
        io_util.write_line_to_file(path='{}/testfile.txt'.format(IOUtilTest.io_test_dir),
                                       line='127.0.0.1:5556,6ky3i91l17')
        values = io_util.delimited_file_to_dict(path='{}/testfile.txt'.format(IOUtilTest.io_test_dir),
                                delimiter=',')
        self.assertTrue(values)
        self.assertEqual('9ty4h6olr4', values['localhost:5555'])
        self.assertEqual('6ky3i91l17', values['127.0.0.1:5556'])
        Path.unlink(Path('{}/testfile.txt'.format(IOUtilTest.io_test_dir)))

    @classmethod
    def tearDownClass(cls):
        super(IOUtilTest, cls).tearDownClass()
        shutil.rmtree(Path(IOUtilTest.io_test_dir))
