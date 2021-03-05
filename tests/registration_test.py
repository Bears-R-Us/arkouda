from context import arkouda as ak
from base_test import ArkoudaTest

class RegistrationTest(ArkoudaTest):
    
    def setUp(self):
        ArkoudaTest.setUp(self)
        self.a_array = ak.ones(10,dtype=ak.int64)
        self.b_array = ak.ones(10,dtype=ak.int64)

    def test_register(self):
        '''
        Tests the following:
        
        1. register invocation
        2. pdarray.name matches register name
        3. original and registered pdarray are equal
        4. method invocation on a cleared, registered array succeeds
        '''
        ar_array = self.a_array.register('test_int64_a')

        self.assertEqual('test_int64_a', ar_array.name)
        self.assertNotEqual(self.a_array.name, ar_array.name)  

        self.assertTrue((self.a_array.to_ndarray() == 
                                      ar_array.to_ndarray()).all())
        ak.clear()
        str(ar_array)
        
        with self.assertRaises(RuntimeError):
            str(self.a_array)
        
    def test_unregister(self):
        '''
        Tests the following:
        
        1. unregister invocation
        2. method invocation on a cleared, unregistered array raises RuntimeError
        '''
        ar_array = self.a_array.register('test_int64_a') 
        
        self.assertEqual('[1 1 1 1 1 1 1 1 1 1]', str(ar_array))
        ar_array.unregister()
        self.assertEqual('[1 1 1 1 1 1 1 1 1 1]', str(ar_array))
        
        ak.clear()
        
        with self.assertRaises(RuntimeError):
            str(ar_array)
            
        with self.assertRaises(RuntimeError):            
            repr(ar_array)
    
    def test_attach(self):
        '''
        Tests the following:
        
        1. Attaching to a registered pdarray
        2. The registered and attached pdarrays are equal
        3. The attached pdarray is deleted server-side following
           unregister of registered pdarray and invocation of 
           ak.clear()
        4. method invocation on cleared attached array raises RuntimeError
        '''
        ar_array = self.a_array.register('test_int64_a')
        aar_array = ak.attach_pdarray('test_int64_a')
        
        self.assertEqual(ar_array.name, aar_array.name)
        self.assertTrue((ar_array.to_ndarray() == 
                                      aar_array.to_ndarray()).all())
        
        ak.disconnect()
        ak.connect(server=ArkoudaTest.server, port=ArkoudaTest.port)
        aar_array = ak.attach_pdarray('test_int64_a')
        
        self.assertEqual(ar_array.name, aar_array.name)
        self.assertTrue((ar_array.to_ndarray() == 
                                    aar_array.to_ndarray()).all())
        
        ar_array.unregister()
        ak.clear()
        
        with self.assertRaises(RuntimeError):            
            str(aar_array)
            
        with self.assertRaises(RuntimeError):            
            repr(aar_array)
    
    def test_clear(self): 
        '''
        Tests the following:
        
        1. clear() removes server-side pdarrays that are unregistered
        2. Registered pdarrays remain after ak.clear()
        3. All cleared pdarrays throw RuntimeError upon method invocation
        4. Method invocation on registered arrays succeeds after ak.clear()
        '''
        ar_array = self.a_array.register('test_int64_a')
        br_array = self.a_array.register('test_int64_b')
        twos_array = ak.ones(10,dtype=ak.int64).register('twos_array')
        twos_array.fill(2)
        
        g_twos_array = self.a_array + self.b_array
        self.assertTrue((twos_array.to_ndarray() == 
                                     g_twos_array.to_ndarray()).all())

        ak.clear()

        with self.assertRaises(RuntimeError):
            str(self.a_array)

        with self.assertRaises(RuntimeError):
            self.a_array + self.b_array

        g_twos_array = ar_array + br_array
        self.assertTrue((twos_array.to_ndarray() == 
                                       g_twos_array.to_ndarray()).all())
