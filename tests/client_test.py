from base_test import ArkoudaTest
from context import arkouda as ak

'''
Tests basic Arkouda client functionality
'''
from util.test.util import start_arkouda_server
class ClientTest(ArkoudaTest):
    
    def test_client_connected(self):
        '''
        Tests the following methods:
        ak.client.connected()
        ak.client.disconnect()
        ak.client.connect()
        
        :return: None
        :raise: AssertionError if an assert* method returns incorrect value or
                if there is a error in connecting or disconnecting from  the
                Arkouda server
        '''
        self.assertTrue(ak.client.connected)
        try:
            ak.disconnect()
        except Exception as e:
            raise AssertionError(e)
   
        self.assertFalse(ak.client.connected)
        try:
            ak.connect(server=ArkoudaTest.server, port=ArkoudaTest.port)
        except Exception as e:
            raise AssertionError(e)
        self.assertTrue(ak.client.connected)
        
    def test_disconnect_on_disconnected_client(self):
        '''
        Tests the ak.disconnect() method invoked on a client that is already
        disconnect to ensure there is no error
        '''
        ak.disconnect()
        self.assertFalse(ak.client.connected)
        ak.disconnect()
        ak.connect(server=ArkoudaTest.server, port=ArkoudaTest.port)
        
    def test_shutdown(self):
        '''
        Tests the ak.shutdown() method
        '''
        ak.shutdown()
        start_arkouda_server(numlocales=1)
        
    def test_client_get_config(self):
        '''
        Tests the ak.client.get_config() method
        
        :return: None
        :raise: AssertionError if one or more Config values are not as expected 
                or the call to ak.client.get_config() fails 
        '''
        try:
            config = ak.client.get_config()
        except Exception as e:
            raise AssertionError(e)
        self.assertEqual(ArkoudaTest.port, config['ServerPort'])
        self.assertTrue('arkoudaVersion' in config)
        
    def test_client_context(self):   
        '''
        Tests the ak.client.context method
        
        :return: None
        :raise: AssertionError if one or more context values are not as expected 
                or the call to ak.client.context fails 
        '''      
        try: 
            context = ak.client.context
        except Exception as e:
            raise AssertionError(e)
        self.assertTrue(context)
        self.assertFalse(context.closed)
        
    def test_get_mem_used(self):
        '''
        Tests the ak.client.get_mem_used method
        
        :return: None
        :raise: AssertionError if one or more ak.get_mem_used values are not as 
                expected or the call to ak.client.get_mem_used() fails 
        '''  
        try:
            mem_used = ak.client.get_mem_used()
        except Exception as e:
            raise AssertionError(e)
        self.assertTrue(mem_used > 0)
        
        
    def test_no_op(self):
        '''
        Tests the ak.client._no_op method
        
        :return: None
        :raise: AssertionError if return message is not 'noop'
        '''   
        noop = ak.client._no_op()
        self.assertEqual('noop', noop)
        
    def test_client_configuration(self):
        '''
        Tests the ak.client.set_defaults() method as well as set/get
        parrayIterThresh, maxTransferBytes, and verbose config params.
        '''
        ak.client.pdarrayIterThresh = 50
        ak.client.maxTransferBytes = 1048576000
        ak.client.verbose = True
        self.assertEqual(50, ak.client.pdarrayIterThresh)
        self.assertEqual(1048576000, ak.client.maxTransferBytes)
        self.assertTrue(ak.client.verbose)
        ak.client.set_defaults()
        self.assertEqual(100, ak.client.pdarrayIterThresh)
        self.assertEqual(1073741824, ak.client.maxTransferBytes)
        self.assertFalse(ak.client.verbose)
