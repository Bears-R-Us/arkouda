import unittest
import subprocess, os
from logging import Logger, Formatter, StreamHandler
from context import arkouda as ak
from util.test.util import get_arkouda_server

test_logger = Logger(name='arkouda_logger')
handler = StreamHandler()
handler.setFormatter(Formatter('[%(name)s] Line %(lineno)d %(levelname)s: %(message)s'))    
test_logger.addHandler(handler)

'''
ArkoudaTest defines the base Arkouda test logic for starting up the arkouda_server at the 
launch of a unittest TestCase and shutting down the arkouda_server at the completion of
the unittest TestCase.

Note: each Arkouda TestCase class extends ArkoudaTest and encompasses 1..n test methods, the 
names of which match the pattern 'test*' (e.g., ArkoudaTest.test_arkouda_server).
'''
class ArkoudaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
      
        '''
        Configures and starts the arkouda_server process via the util.get_arkouda_server (host) 
        and ARKOUDA_SERVER_PORT env variable, defaulting to 5566
        
        :return: None
        :raise: RuntimeError if there is an error in configuring or starting arkouda_server
        '''    
        ArkoudaTest.port = os.getenv('ARKOUDA_SERVER_PORT', 5566)
        ArkoudaTest.server = os.getenv('ARKOUDA_SERVER_HOST', 'localhost')
        ArkoudaTest.is_full_stack = os.getenv('FULL_STACK_TEST', False)

        if ArkoudaTest.is_full_stack:
            try: 
                arkouda_path = get_arkouda_server() 
                ArkoudaTest.ak_server = subprocess.Popen([arkouda_path, 
                                              '--ServerPort={}'.format(ArkoudaTest.port), '--quiet'])
                print('Started arkouda_server in full stack test mode')
            except Exception as e:
                raise RuntimeError('in configuring or starting the arkouda_server: {}, check ' +
                         'environment and/or arkouda_server installation', e)
        else:
            print('in client stack test mode')

    def setUp(self):
        
        '''
        Connects an Arkouda client for each test case
        
        :return: None
        :raise: ConnectionError if exception is raised in connecting to arkouda_server
        '''
        try:
            ak.client.connect(server=ArkoudaTest.server, port=ArkoudaTest.port)
        except Exception as e:
            print(str(e))
            raise ConnectionError(' in connecting to arkouda_server', e)
    
    def test_arkouda_server(self):
      
        '''
        Simply confirsm the arkouda_server process started up correctly and is running
        
        :raise: AssertionError if the ArkoudaTest.ak_server object is None
        '''
        self.assertTrue(ArkoudaTest.ak_server)
    
    def tearDown(self):
      
        '''
        Disconnects the client connection for each test case
        :return: None
        :raise: ConnectionError if exception is raised in connecting to arkouda_server
        '''
        ak.client.disconnect()
        
    @classmethod
    def tearDownClass(cls):
      
        '''
        Shuts down the arkouda_server started in the setUpClass method
        :return: None
        :raise: RuntimeError if there is an error in shutting down arkouda_server
        '''
        if ArkoudaTest.is_full_stack:
            try:
                ArkoudaTest.ak_server.terminate()
            except Exception as e:
                raise RuntimeError('in terminating arkouda_server process', e)
