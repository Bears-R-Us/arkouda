import unittest
import subprocess, os, signal
from context import arkouda as ak
from util.test.util import start_arkouda_server, stop_arkouda_server, get_arkouda_numlocales

'''
ArkoudaTest defines the base Arkouda test logic for starting up the arkouda_server at the 
launch of a unittest TestCase and shutting down the arkouda_server at the completion of
the unittest TestCase.

Note: each Arkouda TestCase class extends ArkoudaTest and encompasses 1..n test methods, the 
names of which match the pattern 'test*' (e.g., ArkoudaTest.test_arkouda_server).
'''
class ArkoudaTest(unittest.TestCase):

    verbose = bool(os.getenv('ARKOUDA_VERBOSE', 'False'))
    port = os.getenv('ARKOUDA_SERVER_PORT')
    server = os.getenv('ARKOUDA_SERVER_HOST')

    port_server_set = port != None and server != None
    full_stack_mode = bool(os.getenv('ARKOUDA_FULL_STACK_TEST', not port_server_set))
    
    @classmethod
    def setUpClass(cls):
        '''
        If in full stack mode, Configures and starts the arkouda_server process
        
        :return: None
        :raise: RuntimeError if there is an error in configuring or starting arkouda_server
        '''
        if ArkoudaTest.full_stack_mode:
            print('starting in full stack mode')
            try: 
                nl = get_arkouda_numlocales()
                (ArkoudaTest.server, ArkoudaTest.port, ArkoudaTest.ak_server) = start_arkouda_server(nl)
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
            raise ConnectionError(e)
    
    def tearDown(self):
        '''
        Disconnects the client connection for each test case
        :return: None
        :raise: ConnectionError if exception is raised in connecting to arkouda_server
        '''
        try:
            ak.client.disconnect()
        except Exception as e:
            raise ConnectionError(e)

    @classmethod
    def tearDownClass(cls):
        '''
        Shuts down the arkouda_server started in the setUpClass method if test is run
        in full stack mode, noop if in client stack mode.

        :return: None
        '''
        if ArkoudaTest.full_stack_mode:
            print('Stopping arkouda_server in full stack test mode')
            stop_arkouda_server()
