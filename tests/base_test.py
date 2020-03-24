import unittest
import os, subprocess
import arkouda as ak

class ArkoudaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try: 
            arkouda_path = os.environ['ARKOUDA_SERVER_PATH']
        except KeyError:
            raise EnvironmentError(('The ARKOUDA_SERVER_PATH env variable output from the ' + 
                                   '"which arkouda_server" command must be set'))

        ArkoudaTest.ak_server = subprocess.Popen([arkouda_path, '--ServerPort=5566', '--quiet'])

    def setUp(self):
        ak.client.connect(port=5566)
        
    def tearDown(self):
        ak.client.disconnect()
        
    @classmethod
    def tearDownClass(cls):
        ArkoudaTest.ak_server.terminate()
        