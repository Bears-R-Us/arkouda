import os, platform, shutil
from os.path import expanduser
from base_test import ArkoudaTest
from context import arkouda
from arkouda import security, io_util
import unittest 

'''
Tests Arkouda client-side security functionality
'''
class SecurityTest(ArkoudaTest):

    def testGenerateToken(self):
        self.assertEqual(32, len(security.generate_token(32)))
        self.assertEqual(16, len(security.generate_token(16)))

    def testGetHome(self):        
        self.assertEqual(expanduser('~'), security.get_home_directory())

    def testGetUsername(self):
        self.assertTrue(security.get_username() in security.username_tokenizer \
                   [platform.system()](security.get_home_directory()))

    # :TODO: need to figure out why shutil fails to delete the .arkouda directory
    @unittest.skip
    def testGetArkoudaDirectory(self):
        security_test_dir = '{}/arkouda_test'.format(os.getcwd())
        os.environ['ARKOUDA_CLIENT_DIRECTORY'] = str(security_test_dir) 
        ak_directory = security.get_arkouda_client_directory()
        self.assertEquals('{}/.arkouda'.format(security_test_dir), str(ak_directory))
        shutil.rmtree(security_test_dir)
