from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.numpy.pdarrayclass import RegistrationError


def cleanup():
    ak.clear()
    for registered_name in ak.list_registry()["Objects"]:
        ak.unregister(registered_name)
    ak.clear()


class RegistrationTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)

    def test_pdarray_registration(self):
        reg_name = "MyPDArray"
        pda = ak.arange(10)
        pda.register(reg_name)

        # validate registered_name set
        self.assertEqual(pda.registered_name, reg_name)
        self.assertTrue(pda.is_registered())

        reg = ak.list_registry()

        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(pda.name in reg["Components"])

        # validate attach
        attach_pda = ak.attach(reg_name)
        self.assertTrue(attach_pda.is_registered())
        self.assertEqual(pda.registered_name, attach_pda.registered_name)
        self.assertListEqual(pda.to_list(), attach_pda.to_list())
        self.assertIsInstance(attach_pda, ak.pdarray)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            pda.register("AlternateName")

        # validate error handling for attempt to reuse name
        pda2 = ak.ones(10)
        with self.assertRaises(RuntimeError):
            pda2.register(reg_name)

        pda.unregister()
        self.assertFalse(pda.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_Strings_registration(self):
        reg_name = "MyPDArray"
        pda = ak.array(["abc", "123", "a", "1", "Test"])
        pda.register(reg_name)

        # validate registered_name set
        self.assertEqual(pda.registered_name, reg_name)
        self.assertTrue(pda.is_registered())

        reg = ak.list_registry()

        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(pda.name in reg["Components"])

        # validate attach
        attach_pda = ak.attach(reg_name)
        self.assertTrue(attach_pda.is_registered())
        self.assertEqual(pda.registered_name, attach_pda.registered_name)
        self.assertListEqual(pda.to_list(), attach_pda.to_list())
        self.assertIsInstance(attach_pda, ak.Strings)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            pda.register("AlternateName")

        # validate error handling for attempt to reuse name
        pda2 = ak.ones(10)
        with self.assertRaises(RuntimeError):
            pda2.register(reg_name)

        pda.unregister()
        self.assertFalse(pda.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_ipv4_registration(self):
        reg_name = "MyIPv4"
        ip = ak.IPv4(ak.arange(10))
        ip.register(reg_name)

        # validate registered_name set
        self.assertEqual(ip.registered_name, reg_name)
        self.assertTrue(ip.is_registered())

        reg = ak.list_registry()
        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(ip.name in reg["Components"])

        # validate attach
        attach_ip = ak.attach(reg_name)
        self.assertTrue(attach_ip.is_registered())
        self.assertEqual(ip.registered_name, attach_ip.registered_name)
        self.assertListEqual(ip.to_list(), attach_ip.to_list())
        self.assertIsInstance(attach_ip, ak.IPv4)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            ip.register("AlternateName")

        # validate error handling for attempt to reuse name
        ip2 = ak.IPv4(ak.arange(1000, 1010))
        with self.assertRaises(RuntimeError):
            ip2.register(reg_name)

        ip.unregister()
        self.assertFalse(ip.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_datetime_registration(self):
        reg_name = "MyDateTime"
        dt = ak.Datetime(ak.arange(0, 100))
        dt.register(reg_name)

        # validate registered_name set
        self.assertEqual(dt.registered_name, reg_name)
        self.assertTrue(dt.is_registered())

        reg = ak.list_registry()
        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(dt.name in reg["Components"])

        # validate attach
        attach_dt = ak.attach(reg_name)
        self.assertTrue(attach_dt.is_registered())
        self.assertEqual(dt.registered_name, attach_dt.registered_name)
        self.assertListEqual(dt.to_list(), attach_dt.to_list())
        self.assertIsInstance(attach_dt, ak.Datetime)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            dt.register("AlternateName")

        # validate error handling for attempt to reuse name
        dt2 = ak.Datetime(ak.arange(1000, 1100))
        with self.assertRaises(RuntimeError):
            dt2.register(reg_name)

        dt.unregister()
        self.assertFalse(dt.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_timedelta_registration(self):
        reg_name = "MyTimeDelta"
        td = ak.Timedelta(ak.arange(0, 100))
        td.register(reg_name)

        # validate registered_name set
        self.assertEqual(td.registered_name, reg_name)
        self.assertTrue(td.is_registered())

        reg = ak.list_registry()
        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(td.name in reg["Components"])

        # validate attach
        attach_td = ak.attach(reg_name)
        self.assertTrue(attach_td.is_registered())
        self.assertEqual(td.registered_name, attach_td.registered_name)
        self.assertListEqual(td.to_list(), attach_td.to_list())
        self.assertIsInstance(attach_td, ak.Timedelta)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            td.register("AlternateName")

        # validate error handling for attempt to reuse name
        td2 = ak.Timedelta(ak.arange(1000, 1100))
        with self.assertRaises(RuntimeError):
            td2.register(reg_name)

        td.unregister()
        self.assertFalse(td.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_segarray_registration(self):
        reg_name = "MySegArray"
        sa = ak.SegArray(ak.arange(0, 20, 2), ak.arange(40))
        sa.register(reg_name)

        # validate registered_name set
        self.assertEqual(sa.registered_name, reg_name)
        self.assertTrue(sa.is_registered())

        reg = ak.list_registry()
        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(sa.segments.name in reg["Components"])
        self.assertTrue(sa.values.name in reg["Components"])

        # validate attach
        attach_sa = ak.attach(reg_name)
        self.assertTrue(attach_sa.is_registered())
        self.assertEqual(sa.registered_name, attach_sa.registered_name)
        self.assertListEqual(sa.to_list(), attach_sa.to_list())
        self.assertIsInstance(attach_sa, ak.SegArray)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            sa.register("AlternateName")

        # validate error handling for attempt to reuse name
        sa2 = ak.SegArray(ak.arange(0, 20, 2), ak.arange(40))
        with self.assertRaises(RuntimeError):
            sa2.register(reg_name)

        sa.unregister()
        self.assertFalse(sa.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_dataframe_registration(self):
        from pandas.testing import assert_frame_equal

        reg_name = "MyDataFrame"
        df = ak.DataFrame(
            {
                "pda": ak.arange(10),
                "str": ak.random_strings_uniform(0, 3, 10),
                "ip": ak.IPv4(ak.arange(10)),
                "dt": ak.Datetime(ak.arange(10)),
                "td": ak.Timedelta(ak.arange(10)),
                "cat": ak.Categorical(ak.array(["a", "b", "c", "c", "a", "a", "d", "d", "e", "c"])),
                "seg": ak.SegArray(ak.arange(0, 20, 2), ak.arange(20)),
                "bv": ak.BitVector(ak.arange(10)),
            }
        )
        df.register(reg_name)

        # validate registered_name set
        self.assertEqual(df.registered_name, reg_name)
        self.assertTrue(df.is_registered())

        reg = ak.list_registry()
        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(df.index.values.name in reg["Components"])
        for k, val in df.items():
            if val.objType == ak.Categorical.objType:
                self.assertTrue(val.codes.name in reg["Components"])
                self.assertTrue(val.categories.name in reg["Components"])
                self.assertTrue(val._akNAcode.name in reg["Components"])
                if val.segments is not None and val.permutation is not None:
                    self.assertTrue(val.segments.name in reg["Components"])
                    self.assertTrue(val.permutation.name in reg["Components"])
            elif val.objType == ak.SegArray.objType:
                self.assertTrue(val.segments.name in reg["Components"])
                self.assertTrue(val.values.name in reg["Components"])
            else:
                self.assertTrue(val.name in reg["Components"])

        # validate attach
        attach_df = ak.attach(reg_name)
        self.assertTrue(attach_df.is_registered())
        self.assertEqual(df.registered_name, attach_df.registered_name)
        # need to index the attached to ensure same columns order
        self.assertTrue(
            assert_frame_equal(df.to_pandas(), attach_df[df.columns.values].to_pandas()) is None
        )
        self.assertIsInstance(attach_df, ak.DataFrame)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            df.register("AlternateName")

        # validate error handling for attempt to reuse name
        df2 = ak.DataFrame({"col": ak.arange(10)})
        with self.assertRaises(RuntimeError):
            df2.register(reg_name)

        df.unregister()
        self.assertFalse(df.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_groupby_registration(self):
        reg_name = "MyGroupBy"
        g = ak.GroupBy(
            [
                ak.array([0, 1, 2, 2, 1, 1, 1]),
                ak.array(["a", "b", "a", "a", "c", "b", "c"]),
                ak.Categorical(ak.array(["a", "b", "a", "a", "c", "b", "c"])),
            ]
        )
        g.register(reg_name)
        # validate registered_name set
        self.assertEqual(g.registered_name, reg_name)
        self.assertTrue(g.is_registered())

        reg = ak.list_registry()
        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(g.segments.name in reg["Components"])
        self.assertTrue(g.permutation.name in reg["Components"])
        self.assertTrue(g._uki.name in reg["Components"])

        for k in g.keys:
            if k.objType == ak.Categorical.objType:
                self.assertTrue(k.codes.name in reg["Components"])
                self.assertTrue(k.categories.name in reg["Components"])
                self.assertTrue(k._akNAcode.name in reg["Components"])
                if k.segments is not None and k.permutation is not None:
                    self.assertTrue(k.segments.name in reg["Components"])
                    self.assertTrue(k.permutation.name in reg["Components"])
            else:
                self.assertTrue(k.name in reg["Components"])

        # validate attach
        attach_g = ak.attach(reg_name)
        self.assertTrue(attach_g.is_registered())
        self.assertEqual(g.registered_name, attach_g.registered_name)
        # need to index the attached to ensure same columns order
        self.assertListEqual(g.segments.to_list(), attach_g.segments.to_list())
        self.assertListEqual(g.permutation.to_list(), attach_g.permutation.to_list())
        self.assertListEqual(g._uki.to_list(), attach_g._uki.to_list())
        for k, attach_k in zip(g.keys, attach_g.keys):
            self.assertListEqual(k.to_list(), attach_k.to_list())
        self.assertIsInstance(attach_g, ak.GroupBy)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            g.register("AlternateName")

        # validate error handling for attempt to reuse name
        g2 = ak.GroupBy(ak.array([0, 1, 2, 1, 3, 0]))
        with self.assertRaises(RuntimeError):
            g2.register(reg_name)

        g.unregister()
        self.assertFalse(g.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_categorical_registration(self):
        reg_name = "MyCategorical"
        cat = ak.Categorical(ak.array(["a", "b", "c", "a", "c", "d", "a"]))
        cat.register(reg_name)

        # validate registered_name set
        self.assertEqual(cat.registered_name, reg_name)
        self.assertTrue(cat.is_registered())

        reg = ak.list_registry()
        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(cat.codes.name in reg["Components"])
        self.assertTrue(cat.categories.name in reg["Components"])
        self.assertTrue(cat._akNAcode.name in reg["Components"])
        if cat.segments is not None and cat.permutation is not None:
            self.assertTrue(cat.segments.name in reg["Components"])
            self.assertTrue(cat.permutation.name in reg["Components"])

        # validate attach
        attach_cat = ak.attach(reg_name)
        self.assertTrue(attach_cat.is_registered())
        self.assertEqual(cat.registered_name, attach_cat.registered_name)
        # need to index the attached to ensure same columns order
        self.assertListEqual(cat.codes.to_list(), attach_cat.codes.to_list())
        self.assertListEqual(cat.categories.to_list(), attach_cat.categories.to_list())
        self.assertListEqual(cat._akNAcode.to_list(), attach_cat._akNAcode.to_list())
        if cat.segments is not None and cat.permutation is not None:
            self.assertListEqual(cat.segments.to_list(), attach_cat.segments.to_list())
            self.assertListEqual(cat.permutation.to_list(), attach_cat.permutation.to_list())
        self.assertIsInstance(attach_cat, ak.Categorical)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            cat.register("AlternateName")

        # validate error handling for attempt to reuse name
        cat2 = ak.Categorical(ak.array(["a", "a", "b"]))
        with self.assertRaises(RuntimeError):
            cat2.register(reg_name)

        cat.unregister()
        self.assertFalse(cat.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_index_registration(self):
        reg_name = "MyIndex"
        i = ak.Index(ak.arange(10))
        i.register(reg_name)

        # validate registered_name set
        self.assertEqual(i.registered_name, reg_name)
        self.assertTrue(i.is_registered())

        reg = ak.list_registry()

        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(i.values.name in reg["Components"])

        # validate attach
        attach_i = ak.attach(reg_name)
        self.assertTrue(attach_i.is_registered())
        self.assertEqual(i.registered_name, attach_i.registered_name)
        self.assertListEqual(i.to_list(), attach_i.to_list())
        self.assertIsInstance(attach_i, ak.Index)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            i.register("AlternateName")

        # validate error handling for attempt to reuse name
        i2 = ak.Index(ak.arange(10))
        with self.assertRaises(RuntimeError):
            i2.register(reg_name)

        i.unregister()
        self.assertFalse(i.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_multi_index_registration(self):
        reg_name = "MyIndex"
        i = ak.Index.factory(
            [
                ak.arange(10),
                ak.random_strings_uniform(0, 2, 10),
                ak.Categorical(ak.random_strings_uniform(0, 2, 10)),
            ]
        )
        i.register(reg_name)

        # validate registered_name set
        self.assertEqual(i.registered_name, reg_name)
        self.assertTrue(i.is_registered())

        reg = ak.list_registry()

        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        for x in i.levels:
            if x.objType == ak.Categorical.objType:
                self.assertTrue(x.codes.name in reg["Components"])
                self.assertTrue(x.categories.name in reg["Components"])
                self.assertTrue(x._akNAcode.name in reg["Components"])
                if x.segments is not None and x.permutation is not None:
                    self.assertTrue(x.segments.name in reg["Components"])
                    self.assertTrue(x.permutation.name in reg["Components"])
            else:
                self.assertTrue(x.name in reg["Components"])

        # validate attach
        attach_i = ak.attach(reg_name)
        self.assertTrue(attach_i.is_registered())
        self.assertEqual(i.registered_name, attach_i.registered_name)
        self.assertListEqual(i.to_list(), attach_i.to_list())
        self.assertIsInstance(attach_i, ak.MultiIndex)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            i.register("AlternateName")

        # validate error handling for attempt to reuse name
        i2 = ak.Index.factory([ak.arange(10), ak.randint(0, 100, 10)])
        with self.assertRaises(RuntimeError):
            i2.register(reg_name)

        i.unregister()
        self.assertFalse(i.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_series_registration(self):
        reg_name = "MySeries"
        s = ak.Series(ak.randint(0, 3, 10))
        s.register(reg_name)

        self.assertTrue(s.is_registered())
        # validate registered_name set
        self.assertEqual(s.registered_name, reg_name)

        reg = ak.list_registry()

        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(s.index.values.name in reg["Components"])
        self.assertTrue(s.values.name in reg["Components"])

        # validate attach
        attach_s = ak.attach(reg_name)
        self.assertTrue(attach_s.is_registered())
        self.assertEqual(s.registered_name, attach_s.registered_name)
        self.assertListEqual(s.to_list(), attach_s.to_list())
        self.assertIsInstance(attach_s, ak.Series)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            s.register("AlternateName")

        # validate error handling for attempt to reuse name
        s2 = ak.Series(ak.randint(0, 3, 10))
        with self.assertRaises(RuntimeError):
            s2.register(reg_name)

        s.unregister()
        self.assertFalse(s.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_bitvector_registration(self):
        reg_name = "MyBitVector"
        b = ak.BitVector(ak.randint(0, 100, 10))
        b.register(reg_name)

        # validate registered_name set
        self.assertEqual(b.registered_name, reg_name)
        self.assertTrue(b.is_registered())

        reg = ak.list_registry()

        # assert that the object is registered
        self.assertTrue(reg_name in reg["Objects"])
        # assert that the sym entry name is recorded
        self.assertTrue(b.values.name in reg["Components"])

        # validate attach
        attach_b = ak.attach(reg_name)
        self.assertTrue(attach_b.is_registered())
        self.assertEqual(b.registered_name, attach_b.registered_name)
        self.assertListEqual(b.to_list(), attach_b.to_list())
        self.assertIsInstance(attach_b, ak.BitVector)

        # validate error handling for double registration
        with self.assertRaises(RegistrationError):
            b.register("AlternateName")

        # validate error handling for attempt to reuse name
        b2 = ak.BitVector(ak.arange(100, 150))
        with self.assertRaises(RuntimeError):
            b2.register(reg_name)

        b.unregister()
        self.assertFalse(b.is_registered())
        reg = ak.list_registry()
        self.assertEqual(len(reg["Objects"]), 0)
        self.assertEqual(len(reg["Components"]), 0)

        cleanup()

    def test_registered_component(self):
        a = ak.arange(10)
        s = ak.SegArray(ak.arange(0, 20, 2), ak.arange(20))
        df = ak.DataFrame(
            {
                "array": a,
                "SegArray": s,
            }
        )

        s.register("MySegArray")
        df.register("MyDataFrame")

        # verify that a is not a registered object, but is a registered component
        self.assertTrue(a.is_registered())

        # verify that the 2 registered entries are registered
        self.assertTrue(s.is_registered())
        self.assertTrue(df.is_registered())

        # verify that components seen as registered after original unregistered
        s.unregister()
        self.assertTrue(df["SegArray"].is_registered())

        cleanup()

    def test_error_handling(self):
        a = ak.ones(3, dtype=ak.int64)

        with self.assertRaises(
            TypeError, msg="register() should raise TypeError when user_defined_name is not a str"
        ):
            a.register(7)

        with self.assertRaises(
            TypeError, msg="attach() should raise TypeError when user_defined_name is not a str"
        ):
            ak.attach(7)

        with self.assertRaises(
            TypeError, msg="is_registered() should raise TypeError when user_defined_name is not a str"
        ):
            ak.is_registered(7)

        cleanup()

    def test_clear(self):
        a = ak.arange(10)
        b = ak.ones(15)
        a.register("MyArray")

        self.assertTrue(a.is_registered())
        self.assertFalse(b.is_registered())

        sym_tab = ak.list_symbol_table()
        self.assertTrue(a.name in sym_tab)
        self.assertTrue(b.name in sym_tab)

        ak.clear()

        sym_tab = ak.list_symbol_table()
        self.assertTrue(a.name in sym_tab)
        self.assertFalse(b.name in sym_tab)

        self.assertTrue(a.is_registered())

        cleanup()

    def test_list_registry(self):
        a = ak.arange(10)
        s = ak.SegArray(ak.arange(0, 10, 2), ak.arange(10))

        a.register("MyArray")
        s.register("MySeg")

        reg = ak.list_registry()  # access registry without object types

        # verify the objects exist in the register
        self.assertTrue(a.registered_name in reg["Objects"])
        self.assertTrue(s.registered_name in reg["Objects"])
        self.assertTrue(a.name in reg["Components"])
        self.assertTrue(s.segments.name in reg["Components"])
        self.assertTrue(s.values.name in reg["Components"])

        reg = ak.list_registry(detailed=True)  # access registry with object types
        self.assertTrue((a.registered_name, a.objType.upper()) in reg["Objects"])
        self.assertTrue((s.registered_name, s.objType.upper()) in reg["Objects"])
        self.assertTrue(a.name in reg["Components"])
        self.assertTrue(s.segments.name in reg["Components"])
        self.assertTrue(s.values.name in reg["Components"])

        cleanup()

    def test_register_attach_unregister_all(self):
        a = ak.arange(10)
        s = ak.array(["a", "b", "c"])
        c = ak.Categorical(s)

        ak.register_all({"MyArray": a, "MyStrings": s, "MyCat": c})

        # validate that all objects are registered
        self.assertEqual(a.registered_name, "MyArray")
        self.assertEqual(s.registered_name, "MyStrings")
        self.assertEqual(c.registered_name, "MyCat")

        reg = ak.list_registry()
        self.assertTrue(a.registered_name in reg["Objects"])
        self.assertTrue(s.registered_name in reg["Objects"])
        self.assertTrue(c.registered_name in reg["Objects"])

        # validate that all objects are attached
        att = ak.attach_all(["MyArray", "MyCat"])
        self.assertTrue("MyArray" in att)
        self.assertTrue("MyCat" in att)
        self.assertIsInstance(att["MyArray"], ak.pdarray)
        self.assertIsInstance(att["MyCat"], ak.Categorical)
        self.assertListEqual(att["MyArray"].to_list(), a.to_list())
        self.assertListEqual(att["MyCat"].to_list(), c.to_list())

        # validate all objects are unregistered
        ak.unregister_all(["MyStrings", "MyArray"])
        reg = ak.list_registry()
        self.assertFalse("MyArray" in reg["Objects"])
        self.assertFalse("MyStrings" in reg["Objects"])
        self.assertTrue("MyCat" in reg["Objects"])

        cleanup()

    def test_attach_weak_binding(self):
        """
        Ultimately pdarrayclass issues delete calls to the server when a bound object goes out of scope,
        if you bind to a server object more than once and one of those goes out of scope it affects
        all other references to it.
        """
        cleanup()
        a = ak.ones(3, dtype=ak.int64).register("a_reg")
        self.assertTrue(str(a), "Expected to pass")
        b = ak.attach("a_reg")
        b.unregister()
        b = None  # Force out of scope
        with self.assertRaises(RuntimeError):
            str(a)

    def test_symentry_cleanup(self):
        pda = ak.arange(10)
        self.assertTrue(len(ak.list_symbol_table()) > 0)
        pda = None
        self.assertEqual(len(ak.list_symbol_table()), 0)

        s = ak.array(["a", "b", "c"])
        self.assertTrue(len(ak.list_symbol_table()) > 0)
        s = None
        self.assertEqual(len(ak.list_symbol_table()), 0)

        cat = ak.Categorical(ak.array(["a", "b", "c"]))
        self.assertTrue(len(ak.list_symbol_table()) > 0)
        cat = None
        self.assertEqual(len(ak.list_symbol_table()), 0)

        seg = ak.SegArray(
            ak.array([0, 6, 8]), ak.array([10, 11, 12, 13, 14, 15, 20, 21, 30, 31, 32, 33])
        )
        self.assertTrue(len(ak.list_symbol_table()) > 0)
        seg = None
        self.assertEqual(len(ak.list_symbol_table()), 0)

        g = ak.GroupBy(
            [ak.arange(3), ak.array(["a", "b", "c"]), ak.Categorical(ak.array(["a", "b", "c"]))]
        )
        self.assertTrue(len(ak.list_symbol_table()) > 0)
        g = None
        self.assertEqual(len(ak.list_symbol_table()), 0)

        d = ak.DataFrame(
            {
                "pda": ak.arange(3),
                "s": ak.array(["a", "b", "c"]),
                "cat": ak.Categorical(ak.array(["a", "b", "c"])),
                "seg": ak.SegArray(
                    ak.array([0, 6, 8]), ak.array([10, 11, 12, 13, 14, 15, 20, 21, 30, 31, 32, 33])
                ),
            }
        )
        self.assertTrue(len(ak.list_symbol_table()) > 0)
        d = None
        self.assertEqual(len(ak.list_symbol_table()), 0)

        cleanup()
