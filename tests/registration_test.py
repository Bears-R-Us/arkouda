import pytest
from context import arkouda as ak
from base_test import ArkoudaTest
from arkouda.pdarrayclass import RegistrationError, unregister_pdarray_by_name
N = 100
UNIQUE = N // 4


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

        1. info() with an empty symbol table returns ak.EmptySymbolTable regardless of arguments
        2. info(ak.RegisteredSymbols) when no objects are registered returns ak.EmptyRegistry
        3. The registered field is set to false for objects that have not been registered
        4. The registered field is set to true for objects that have been registered
        5. info(ak.AllSymbols) contains both registered and non-registered objects
        6. info(ak.RegisteredSymbols) only contains registered objects
        '''
        # Cleanup symbol table from previous tests
        cleanup()

        self.assertEqual(ak.info(ak.AllSymbols), ak.EmptySymbolTable,
                         msg='info(AllSymbols) empty symbol table message failed')
        self.assertEqual(ak.info(ak.RegisteredSymbols), ak.EmptySymbolTable,
                         msg='info(RegisteredSymbols) empty symbol table message failed')

        my_array = ak.ones(10, dtype=ak.int64)
        self.assertTrue('registered:false' in ak.info(ak.AllSymbols).split(),
                        msg='info(AllSymbols) should contain non-registered objects')

        self.assertEqual(ak.info(ak.RegisteredSymbols), ak.EmptyRegistry,
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
        # RegisteredSymbols info calls should return ak.EmptyRegistry
        my_array.unregister()
        self.assertTrue('registered:false' in ak.info("keep_me").split(),
                        msg='info(keep_me) registered field should be false after unregister()')
        self.assertTrue('registered:false' in ak.info(ak.AllSymbols).split(),
                        msg='info(AllSymbols) should contain unregistered objects')
        self.assertEqual(ak.info(ak.RegisteredSymbols), ak.EmptyRegistry,
                         msg='info(RegisteredSymbols) empty registry message failed after unregister()')

        # After clear(), every info call should return ak.EmptySymbolTable
        ak.clear()
        self.assertEqual(ak.info("keep_me"), ak.EmptySymbolTable,
                         msg='info(keep_me) empty symbol message failed')
        self.assertEqual(ak.info(ak.AllSymbols), ak.EmptySymbolTable,
                         msg='info(AllSymbols) empty symbol table message failed')
        self.assertEqual(ak.info(ak.RegisteredSymbols), ak.EmptySymbolTable,
                         msg='info(RegisteredSymbols) empty symbol table message failed')

    def test_is_registered(self):
        """
        Tests the pdarray.is_registered() function
        """
        cleanup()
        a = ak.ones(10, dtype=ak.int64)
        self.assertFalse(a.is_registered())

        a.register('keep')
        self.assertTrue(a.is_registered())

        a.unregister()
        self.assertFalse(a.is_registered())
        ak.clear()

    def test_list_registry(self):
        """
        Tests the generic ak.list_registry() function
        """
        cleanup()
        # Test list_registry when the symbol table is empty
        self.assertFalse(ak.list_registry(), "registry should be empty")

        a = ak.ones(10, dtype=ak.int64)
        # list_registry() should return an empty list which is implicitly False
        self.assertFalse(ak.list_registry())

        a.register('keep')
        self.assertTrue('keep' in ak.list_registry())
        cleanup()

    def test_string_registration_suite(self):
        cleanup()
        # Initial registration should set name
        keep = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        self.assertTrue(keep.register("keep_me").name == "keep_me")
        self.assertTrue(keep.offsets.name == "keep_me.offsets")
        self.assertTrue(keep.bytes.name == "keep_me.bytes")

        self.assertTrue(keep.is_registered(), "Expected Strings object to be registered")

        # Register a second time to confirm name change
        self.assertTrue(keep.register("kept").name == "kept")
        self.assertTrue(keep.offsets.name == "kept.offsets")
        self.assertTrue(keep.bytes.name == "kept.bytes")
        self.assertTrue(keep.is_registered(), "Object should be registered with updated name")

        # Add an item to discard, confirm our registered item remains and discarded item is gone
        discard = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        ak.clear()
        self.assertTrue(keep.name == "kept")
        self.assertTrue(keep.offsets.name == "kept.offsets")
        self.assertTrue(keep.bytes.name == "kept.bytes")
        with self.assertRaises(RuntimeError, msg="discard was not registered and should be discarded"):
            str(discard)

        # Unregister, should remain usable until we clear
        keep.unregister()
        str(keep) # Should not cause error
        self.assertFalse(keep.is_registered(), "This item should no longer be registered")
        ak.clear()
        with self.assertRaises(RuntimeError, msg="keep was unregistered and should be cleared"):
            str(keep) # should cause RuntimeError

        # Test attach functionality
        s1 = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        self.assertTrue(s1.register("uut").is_registered(), "uut should be registered")
        s1 = None
        self.assertTrue(s1 is None, "Reference should be cleared")
        s1 = ak.Strings.attach("uut")
        self.assertTrue(s1.is_registered(), "Should have re-attached to registered object")
        str(s1)  # This will throw an exception if the object doesn't exist server-side

        # Test the Strings unregister by name using previously registered object
        ak.Strings.unregister_strings_by_name("uut")
        self.assertFalse(s1.is_registered(), "Expected object to be unregistered")
        cleanup()

    def test_string_is_registered(self):
        """
        Tests the Strings.is_registered() function
        """
        keep = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        self.assertFalse(keep.is_registered())

        keep.register('keep_me')
        self.assertTrue(keep.is_registered())

        keep.unregister()
        self.assertFalse(keep.is_registered())

        # Now mess with one of the internal pieces to test is_registered() logic
        self.assertTrue(keep.register("uut").is_registered(), "Re-register keep as uut")
        ak.unregister_pdarray_by_name("uut.bytes")
        with self.assertRaises(RegistrationError, msg="Expected RegistrationError on mis-matched pieces"):
            keep.is_registered()

        ak.clear()

    def test_delete_registered(self):
        """
        Tests the following:

        1. delete cmd doesn't delete registered objects and returns appropriate message
        2. delete cmd does delete non-registered objects and returns appropriate message
        3. delete cmd raises RuntimeError for unknown symbols
        """
        cleanup()
        a = ak.ones(3, dtype=ak.int64)
        b = ak.ones(3, dtype=ak.int64)

        # registered objects are not deleted from symbol table
        a.register('keep')
        self.assertEqual(ak.client.generic_msg(cmd='delete', args=a.name),
                         f'registered symbol, {a.name}, not deleted')
        self.assertTrue(a.name in ak.list_symbol_table())

        # non-registered objects are deleted from symbol table
        self.assertEqual(ak.client.generic_msg(cmd='delete', args=b.name),
                         'deleted ' + b.name)
        self.assertTrue(b.name not in ak.list_symbol_table())

        # RuntimeError when calling delete on an object not in the symbol table
        with self.assertRaises(RuntimeError):
            ak.client.generic_msg(cmd='delete', args='not_in_table')

    def test_categorical_registration_suite(self):
        """
        Test register, is_registered, attach, unregister, unregister_categorical_by_name
        """
        cleanup()  # Make sure we start with a clean registry
        c = ak.Categorical(ak.array([f"my_cat {i}" for i in range(1, 11)]))
        self.assertFalse(c.is_registered(), "test_me should be unregistered")
        self.assertTrue(c.register("test_me").is_registered(), "test_me categorical should be registered")
        c = None  # Should trigger destructor, but survive server deletion because it is registered
        self.assertTrue(c is None, "The reference to `c` should be None")
        c = ak.Categorical.attach("test_me")
        self.assertTrue(c.is_registered(), "test_me categorical should be registered after attach")
        c.unregister()
        self.assertFalse(c.is_registered(), "test_me should be unregistered")
        self.assertTrue(c.register("another_name").name == "another_name" and c.is_registered())

        # Test static unregister_by_name
        ak.Categorical.unregister_categorical_by_name("another_name")
        self.assertFalse(c.is_registered(), "another_name should be unregistered")

        # now mess with the subcomponents directly to test is_registered mis-match logic
        c.register("another_name")
        unregister_pdarray_by_name("another_name.codes")
        with pytest.raises(RegistrationError):
            c.is_registered()

    def test_attach_weak_binding(self):
        """
        Ultimately pdarrayclass issues delete calls to the server when a bound object goes out of scope, if you bind
        to a server object more than once and one of those goes out of scope it affects all other references to it.
        """
        cleanup()
        a = ak.ones(3, dtype=ak.int64).register("a_reg")
        self.assertTrue(str(a), "Expected to pass")
        b = ak.attach_pdarray("a_reg")
        b.unregister()
        b = None  # Force out of scope
        with self.assertRaises(RuntimeError):
            str(a)

def cleanup():
    ak.clear()
    if ak.info(ak.AllSymbols) != ak.EmptySymbolTable:
        for registered_object in filter(None, ak.info(ak.AllSymbols).split('\n')):
            name = registered_object.split()[0].split(':')[1].replace('"', '')
            ak.unregister_pdarray_by_name(name)
        ak.clear()
