from base_test import ArkoudaTest
from context import arkouda as ak

class ClientTest(ArkoudaTest):
    
    def test_client_connected(self):
        self.assertTrue(ak.client.connected)
        ak.client.disconnect()
        self.assertFalse(ak.client.connected)
        ak.client.connect(server=ArkoudaTest.server, port=ArkoudaTest.port)
        self.assertTrue(ak.client.connected)
        
    def test_client_get_config(self):
        config = ak.client.get_config()
        self.assertEqual(ArkoudaTest.port, config['ServerPort'])
        self.assertTrue('arkoudaVersion' in config.keys())
        
    def test_client_context(self):
        context = ak.client.context
        self.assertTrue(context)
        self.assertFalse(context.closed)
