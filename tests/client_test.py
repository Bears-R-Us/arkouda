from base_test import ArkoudaTest, test_logger
from context import arkouda as ak

class ClientTest(ArkoudaTest):
    
    def test_client_connected(self):
        self.assertTrue(ak.client.connected)
        ak.client.disconnect()
        self.assertFalse(ak.client.connected)
        arkouda_test
        ak.client.connect(port=5566)
        self.assertTrue(ak.client.connected)
        
    def test_client_get_config(self):
        config = ak.client.get_config()
        self.assertEqual(5566, config['ServerPort'])
        self.assertTrue('arkoudaVersion' in config.keys())
        
    def test_client_context(self):
        context = ak.client.context
        self.assertTrue(context)
        self.assertFalse(context.closed)
