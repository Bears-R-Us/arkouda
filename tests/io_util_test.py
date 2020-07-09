import unittest, os, sys, shutil
from pathlib import Path
from context import arkouda as ak
from arkouda import io_util
from base_test import ArkoudaTest

class IOUtilTest(ArkoudaTest):

    @classmethod
    def setUpClass(cls):
        super(IOUtilTest, cls).setUpClass()
        io_util.get_directory('/tmp/test')

    def testGetDirectory(self):
        dir = io_util.get_directory('/tmp/test')
        self.assertTrue(dir)
        Path.rmdir(Path('/tmp/test'))
        self.assertFalse(os.path.exists('/tmp/test'))
        io_util.get_directory('/tmp/test')
        self.assertTrue(os.path.exists('/tmp/test'))

    def testWriteLineToFile(self):
        io_util.write_line_to_file(path='/tmp/test/testfile.txt',
                                       line='localhost:5555,9ty4h6olr4')
        self.assertTrue(os.path.exists('/tmp/test/testfile.txt'))
        Path.unlink(Path('/tmp/test/testfile.txt'))

    def testDelimitedFileToDict(self):
        io_util.write_line_to_file(path='/tmp/test/testfile.txt',
                                       line='localhost:5555,9ty4h6olr4')
        io_util.write_line_to_file(path='/tmp/test/testfile.txt',
                                       line='127.0.0.1:5556,6ky3i91l17')
        values = io_util.delimited_file_to_dict(path='/tmp/test/testfile.txt',
                                delimiter=',')
        self.assertTrue(values)
        self.assertEqual('9ty4h6olr4', values['localhost:5555'])
        self.assertEqual('6ky3i91l17', values['127.0.0.1:5556'])
        Path.unlink(Path('/tmp/test/testfile.txt'))

    @classmethod
    def tearDownClass(cls):
        super(IOUtilTest, cls).tearDownClass()
        #os.rmdir('/tmp/test')
        shutil.rmtree(Path('/tmp/test'))
