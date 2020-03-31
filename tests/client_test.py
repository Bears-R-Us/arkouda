import base_test
from context import arkouda as ak

class ClientTest(base_test.ArkoudaTest):
    
    def testClientConnected(self):
        self.assertTrue(ak.client.connected)
        ak.client.disconnect()
        self.assertFalse(ak.client.connected)
        ak.client.connect(port=5566)
        self.assertTrue(ak.client.connected)
        
    def testClientGetConfig(self):
        config = ak.client.get_config()
        self.assertEqual(5566, config['ServerPort'])
        self.assertTrue('arkoudaVersion' in config.keys())
        
    def testClientContext(self):
        context = ak.client.context
        self.assertTrue(context)
        self.assertFalse(context.closed)
