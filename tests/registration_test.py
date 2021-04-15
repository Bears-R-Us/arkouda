from context import arkouda as ak
from base_test import ArkoudaTest
from arkouda.pdarrayclass import RegistrationError


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

        self.assertEqual('test_int64_a', self.a_array.name, "Expect name to change inplace")
        self.assertTrue(self.a_array is ar_array, "These should be the same object")
        self.assertTrue((self.a_array.to_ndarray() == ar_array.to_ndarray()).all())
        ak.clear()
        # Both ar_array and self.a_array point to the same object, so both should still be usable.
        str(ar_array)
        str(self.a_array)

        try:
            self.a_array.unregister()
        except (RuntimeError, RegistrationError):
            pass  # Will be tested in `test_unregister`

    def test_double_register(self):
        """
        Tests the case when two objects get registered using the same user_defined_name
        """
        a = ak.ones(3, dtype=ak.int64)
        b = ak.ones(3, dtype=ak.int64)
        b.fill(2)
        a.register("foo")

        with self.assertRaises(RegistrationError, msg="Should raise an Error"):
            b.register("foo")

        # Clean up the registry
        a.unregister()

    def test_registration_type_check(self):
        """
        Tests type checking of user_defined_name for register and attach
        """

        a = ak.ones(3, dtype=ak.int64)

        with self.assertRaises(TypeError, msg="register() should raise TypeError when user_defined_name is not a str"):
            a.register(7)
        with self.assertRaises(TypeError, msg="attach() should raise TypeError when user_defined_name is not a str"):
            a.attach(7)

        ak.clear()

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
        aar_array = self.a_array.register('test_int64_aa')

        self.assertTrue(ar_array is aar_array,
                        msg="With inplace modification, these should be the same")
        self.assertEqual(ar_array.name, "test_int64_aa",
                         msg="ar_array.name should be updated with inplace modification")

        twos_array = ak.ones(10,dtype=ak.int64).register('twos_array')
        twos_array.fill(2)
        
        g_twos_array = self.a_array + self.b_array
        self.assertTrue((twos_array.to_ndarray() == 
                                     g_twos_array.to_ndarray()).all())

        ak.clear() # This should remove self.b_array and g_twos_array

        with self.assertRaises(RuntimeError, msg="g_twos_array should have been cleared because it wasn't registered"):
            str(g_twos_array)

        with self.assertRaises(RuntimeError, msg="self.b_array should have been cleared because it wasn't registered"):
            str(self.b_array)

        # Assert these exist by invoking them and not receiving an exception
        str(self.a_array)
        str(ar_array)

        with self.assertRaises(RuntimeError, msg="Should raise error because self.b_array was cleared"):
            self.a_array + self.b_array

        g_twos_array = ar_array + aar_array
        self.assertTrue((twos_array.to_ndarray() == 
                                       g_twos_array.to_ndarray()).all())

    def test_register_info(self):
        '''
        Tests the following:

        1. info() with an empty symbol table returns '__EMPTY_SYMBOLTABLE__' regardless of arguments
        2. info(ak.RegisteredSymbols) when no objects are registered returns '__EMPTY_REGISTRY__'
        3. The registered field is set to false for objects that have not been registered
        4. The registered field is set to true for objects that have been registered
        5. info(ak.AllSymbols) contains both registered and non-registered objects
        6. info(ak.RegisteredSymbols) only contains registered objects
        '''
        # Cleanup symbol table from previous tests
        cleanup()

        self.assertEqual(ak.info(ak.AllSymbols), '__EMPTY_SYMBOLTABLE__',
                         msg='info(AllSymbols) empty symbol table message failed')
        self.assertEqual(ak.info(ak.RegisteredSymbols), '__EMPTY_SYMBOLTABLE__',
                         msg='info(RegisteredSymbols) empty symbol table message failed')

        my_array = ak.ones(10, dtype=ak.int64)
        self.assertTrue('registered:false' in ak.info(ak.AllSymbols).split(),
                        msg='info(AllSymbols) should contain non-registered objects')

        self.assertEqual(ak.info(ak.RegisteredSymbols), '__EMPTY_REGISTRY__',
                         msg='info(RegisteredSymbols) empty registry message failed')

        # After register(), the registered field should be set to true for all info calls
        my_array.register('keep_me')
        self.assertTrue('registered:true' in ak.info('keep_me').split(),
                        msg='keep_me array not found or not registered')
        self.assertTrue('registered:true' in ak.info(ak.AllSymbols).split(),
                        msg='No registered objects were found in symbol table')
        self.assertTrue('registered:true' in ak.info(ak.RegisteredSymbols).split(),
                        msg='No registered objects were found in registry')

        not_registered_array = ak.ones(10, dtype=ak.int64)
        self.assertTrue(len(ak.info(ak.AllSymbols).split('\n')) > len(ak.info(ak.RegisteredSymbols).split('\n')),
                        msg='info(AllSymbols) should have more objects than info(RegisteredSymbols) before clear()')
        ak.clear()
        self.assertTrue(len(ak.info(ak.AllSymbols).split('\n')) == len(ak.info(ak.RegisteredSymbols).split('\n')),
                        msg='info(AllSymbols) and info(RegisteredSymbols) should have same num of objects after clear()')

        # After unregister(), the registered field should be set to false for AllSymbol and object name info calls
        # RegisteredSymbols info calls should return '__EMPTY_REGISTRY__'
        my_array.unregister()
        self.assertTrue('registered:false' in ak.info("keep_me").split(),
                        msg='info(keep_me) registered field should be false after unregister()')
        self.assertTrue('registered:false' in ak.info(ak.AllSymbols).split(),
                        msg='info(AllSymbols) should contain unregistered objects')
        self.assertEqual(ak.info(ak.RegisteredSymbols), '__EMPTY_REGISTRY__',
                         msg='info(RegisteredSymbols) empty registry message failed after unregister()')

        # After clear(), every info call should return '__EMPTY_SYMBOLTABLE__'
        ak.clear()
        self.assertEqual(ak.info("keep_me"), '__EMPTY_SYMBOLTABLE__',
                         msg='info(keep_me) empty symbol message failed')
        self.assertEqual(ak.info(ak.AllSymbols), '__EMPTY_SYMBOLTABLE__',
                         msg='info(AllSymbols) empty symbol table message failed')
        self.assertEqual(ak.info(ak.RegisteredSymbols), '__EMPTY_SYMBOLTABLE__',
                         msg='info(RegisteredSymbols) empty symbol table message failed')

def cleanup():
    ak.clear()
    if ak.info(ak.AllSymbols) != '__EMPTY_SYMBOLTABLE__':
        for registered_object in filter(None, ak.info(ak.AllSymbols).split('\n')):
            name = registered_object.split()[0].split(':')[1].replace('"', '')
            ak.unregister_pdarray(name)
        ak.clear()
