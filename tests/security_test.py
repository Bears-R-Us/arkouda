import platform
from os.path import expanduser
from base_test import ArkoudaTest
from context import arkouda
from arkouda import security

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

    def testGetArkoudaDirectory(self):
        ak_directory = security.get_arkouda_client_directory()
        self.assertEqual('{}/.arkouda'.format(security.get_home_directory()), 
                         str(ak_directory))
