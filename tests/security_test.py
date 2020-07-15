import os, platform, shutil
from os.path import expanduser
from base_test import ArkoudaTest
from context import arkouda
from arkouda import security, io_util

'''
Tests basic Arkouda client functionality
'''
class SecurityTest(ArkoudaTest):

    def testGenerateToken(self):
        self.assertEqual(32, len(security.generate_token(32)))
        self.assertEqual(16, len(security.generate_token(16)))

    def testGetHome(self):
        self.assertEqual(expanduser('~'), security.get_home_directory())

    def testGetUsername(self):
        self.assertTrue(security.get_username() in security.username_tokenizer\
                   [platform.system()](security.get_home_directory()))
    def testGetArkoudaDirectory(self):
        io_util.get_directory('/tmp/arkouda_test')
        os.environ['ARKOUDA_CLIENT_DIRECTORY'] = '/tmp/arkouda_test' 
        ak_directory = security.get_arkouda_client_directory()
        self.assertTrue('/tmp/arkouda_test/.arkouda' in str(ak_directory))
        shutil.rmtree('/tmp/arkouda_test')
