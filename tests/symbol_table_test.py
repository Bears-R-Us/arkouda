import pytest

import arkouda as ak
from arkouda.pdarrayclass import RegistrationError

N = 100
UNIQUE = N // 4

DTYPES = [ak.int64, ak.uint64, ak.bool_, ak.float64, ak.bigint, ak.str_]


def clean_registry():
    ak.clear()
    for registered_name in ak.list_registry()["Objects"]:
        ak.unregister(registered_name)
    ak.clear()


@pytest.fixture(autouse=True)
def cleanup():
    clean_registry()
    yield
    clean_registry()


class TestRegistration:
    @staticmethod
    def make_pdarray(dtype, size):
        if dtype in [ak.int64, ak.uint64]:
            return ak.arange(size)
        elif dtype == ak.bigint:
            return ak.arange(size) + 2**200
        elif dtype == ak.bool_:
            return ak.randint(0, 2, size, dtype=ak.bool_)
        elif dtype == ak.float64:
            return ak.linspace(-2.5, 2.5, size)
        elif dtype == ak.str_:
            return ak.random_strings_uniform(0, 2, size)

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_pdarray_registration(self, dtype, size):
        reg_name = "MyPDArray"
        pda = self.make_pdarray(dtype, size)
        pda.register(reg_name)

        # validate registered_name set
        assert pda.registered_name == reg_name
        assert pda.is_registered()

        reg = ak.list_registry()

        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert pda.name in reg["Components"]

        # validate attach
        attach_pda = ak.attach(reg_name)
        assert attach_pda.is_registered()
        assert pda.registered_name == attach_pda.registered_name
        assert pda.to_list() == attach_pda.to_list()
        if dtype == ak.str_:
            assert isinstance(attach_pda, ak.Strings)
        else:
            assert isinstance(attach_pda, ak.pdarray)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            pda.register("AlternateName")

        # validate error handling for attempt to reuse name
        pda2 = ak.ones(10)
        with pytest.raises(RuntimeError):
            pda2.register(reg_name)

        pda.unregister()
        assert not pda.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_ipv4_registration(self, size):
        reg_name = "MyIPv4"
        ip = ak.IPv4(self.make_pdarray(ak.uint64, size))
        ip.register(reg_name)

        # validate registered_name set
        assert ip.registered_name == reg_name
        assert ip.is_registered()

        reg = ak.list_registry()
        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert ip.name in reg["Components"]

        # validate attach
        attach_ip = ak.attach(reg_name)
        assert attach_ip.is_registered()
        assert ip.registered_name == attach_ip.registered_name
        assert ip.to_list() == attach_ip.to_list()
        assert isinstance(attach_ip, ak.IPv4)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            ip.register("AlternateName")

        # validate error handling for attempt to reuse name
        ip2 = ak.IPv4(self.make_pdarray(ak.uint64, size))
        with pytest.raises(RuntimeError):
            ip2.register(reg_name)

        ip.unregister()
        assert not ip.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_datetime_registration(self, size):
        reg_name = "MyDateTime"
        dt = ak.Datetime(self.make_pdarray(ak.int64, size))
        dt.register(reg_name)

        # validate registered_name set
        assert dt.registered_name == reg_name
        assert dt.is_registered()

        reg = ak.list_registry()
        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert dt.name in reg["Components"]

        # validate attach
        attach_dt = ak.attach(reg_name)
        assert attach_dt.is_registered()
        assert dt.registered_name == attach_dt.registered_name
        assert dt.to_list() == attach_dt.to_list()
        assert isinstance(attach_dt, ak.Datetime)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            dt.register("AlternateName")

        # validate error handling for attempt to reuse name
        dt2 = ak.Datetime(self.make_pdarray(ak.int64, size))
        with pytest.raises(RuntimeError):
            dt2.register(reg_name)

        dt.unregister()
        assert not dt.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_timedelta_registration(self, size):
        reg_name = "MyTimeDelta"
        td = ak.Timedelta(self.make_pdarray(ak.int64, size))
        td.register(reg_name)

        # validate registered_name set
        assert td.registered_name == reg_name
        assert td.is_registered()

        reg = ak.list_registry()
        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert td.name in reg["Components"]

        # validate attach
        attach_td = ak.attach(reg_name)
        assert attach_td.is_registered()
        assert td.registered_name == attach_td.registered_name
        assert td.to_list() == attach_td.to_list()
        assert isinstance(attach_td, ak.Timedelta)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            td.register("AlternateName")

        # validate error handling for attempt to reuse name
        td2 = ak.Timedelta(self.make_pdarray(ak.int64, size))
        with pytest.raises(RuntimeError):
            td2.register(reg_name)

        td.unregister()
        assert not td.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_segarray_registration(self, dtype, size):
        reg_name = "MySegArray"
        segments = ak.arange(0, size, 5)
        values = self.make_pdarray(dtype, size)
        sa = ak.SegArray(segments, values)
        sa.register(reg_name)

        # validate registered_name set
        assert sa.registered_name == reg_name
        assert sa.is_registered()

        reg = ak.list_registry()
        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert sa.segments.name in reg["Components"]
        assert sa.values.name in reg["Components"]

        # validate attach
        attach_sa = ak.attach(reg_name)
        assert attach_sa.is_registered()
        assert sa.registered_name == attach_sa.registered_name
        assert sa.to_list() == attach_sa.to_list()
        assert isinstance(attach_sa, ak.SegArray)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            sa.register("AlternateName")

        # validate error handling for attempt to reuse name
        sa2 = ak.SegArray(segments, values)
        with pytest.raises(RuntimeError):
            sa2.register(reg_name)

        sa.unregister()
        assert not sa.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_dataframe_registration(self, size):
        from pandas.testing import assert_frame_equal

        reg_name = "MyDataFrame"
        # need to make segarray larger to make same size as other objects
        segments = ak.arange(0, size * 2, 2)
        values = self.make_pdarray(ak.float64, size * 2)
        df = ak.DataFrame(
            {
                "pda": self.make_pdarray(ak.float64, size),
                "str": self.make_pdarray(ak.str_, size),
                "ip": ak.IPv4(self.make_pdarray(ak.int64, size)),
                "dt": ak.Datetime(self.make_pdarray(ak.int64, size)),
                "td": ak.Timedelta(self.make_pdarray(ak.int64, size)),
                "cat": ak.Categorical(self.make_pdarray(ak.str_, size)),
                "seg": ak.SegArray(segments, values),
                "bv": ak.BitVector(self.make_pdarray(ak.int64, size)),
            }
        )
        df.register(reg_name)

        # validate registered_name set
        assert df.registered_name == reg_name
        assert df.is_registered()

        reg = ak.list_registry()
        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert df.index.values.name in reg["Components"]
        for k, val in df.items():
            if val.objType == ak.Categorical.objType:
                assert val.codes.name in reg["Components"]
                assert val.categories.name in reg["Components"]
                assert val._akNAcode.name in reg["Components"]
                if val.segments is not None and val.permutation is not None:
                    assert val.segments.name in reg["Components"]
                    assert val.permutation.name in reg["Components"]
            elif val.objType == ak.SegArray.objType:
                assert val.segments.name in reg["Components"]
                assert val.values.name in reg["Components"]
            else:
                assert val.name in reg["Components"]

        # validate attach
        attach_df = ak.attach(reg_name)
        assert attach_df.is_registered()
        assert df.registered_name == attach_df.registered_name
        # need to index the attached to ensure same columns order
        assert assert_frame_equal(df.to_pandas(), attach_df[df.columns.values].to_pandas()) is None
        assert isinstance(attach_df, ak.DataFrame)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            df.register("AlternateName")

        # validate error handling for attempt to reuse name
        df2 = ak.DataFrame({"col": self.make_pdarray(ak.int64, size)})
        with pytest.raises(RuntimeError):
            df2.register(reg_name)

        df.unregister()
        assert not df.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_groupby_registration(self, size):
        reg_name = "MyGroupBy"
        g = ak.GroupBy(
            [
                self.make_pdarray(ak.int64, size),
                self.make_pdarray(ak.str_, size),
                ak.Categorical(self.make_pdarray(ak.str_, size)),
            ]
        )
        g.register(reg_name)
        # validate registered_name set
        assert g.registered_name == reg_name
        assert g.is_registered()

        reg = ak.list_registry()
        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert g.segments.name in reg["Components"]
        assert g.permutation.name in reg["Components"]
        assert g._uki.name in reg["Components"]

        for k in g.keys:
            if k.objType == ak.Categorical.objType:
                assert k.codes.name in reg["Components"]
                assert k.categories.name in reg["Components"]
                assert k._akNAcode.name in reg["Components"]
                if k.segments is not None and k.permutation is not None:
                    assert k.segments.name in reg["Components"]
                    assert k.permutation.name in reg["Components"]
            else:
                assert k.name in reg["Components"]

        # validate attach
        attach_g = ak.attach(reg_name)
        assert attach_g.is_registered()
        assert g.registered_name == attach_g.registered_name
        # need to index the attached to ensure same columns order
        assert g.segments.to_list() == attach_g.segments.to_list()
        assert g.permutation.to_list() == attach_g.permutation.to_list()
        assert g._uki.to_list() == attach_g._uki.to_list()
        for k, attach_k in zip(g.keys, attach_g.keys):
            assert k.to_list() == attach_k.to_list()
        assert isinstance(attach_g, ak.GroupBy)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            g.register("AlternateName")

        # validate error handling for attempt to reuse name
        g2 = ak.GroupBy(self.make_pdarray(ak.uint64, size))
        with pytest.raises(RuntimeError):
            g2.register(reg_name)

        g.unregister()
        assert not g.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_categorical_registration(self, size):
        reg_name = "MyCategorical"
        cat = ak.Categorical(self.make_pdarray(ak.str_, size))
        cat.register(reg_name)

        # validate registered_name set
        assert cat.registered_name == reg_name
        assert cat.is_registered()

        reg = ak.list_registry()
        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert cat.codes.name in reg["Components"]
        assert cat.categories.name in reg["Components"]
        assert cat._akNAcode.name in reg["Components"]
        if cat.segments is not None and cat.permutation is not None:
            assert cat.segments.name in reg["Components"]
            assert cat.permutation.name in reg["Components"]

        # validate attach
        attach_cat = ak.attach(reg_name)
        assert attach_cat.is_registered()
        assert cat.registered_name == attach_cat.registered_name
        # need to index the attached to ensure same columns order
        assert cat.codes.to_list() == attach_cat.codes.to_list()
        assert cat.categories.to_list() == attach_cat.categories.to_list()
        assert cat._akNAcode.to_list() == attach_cat._akNAcode.to_list()
        if cat.segments is not None and cat.permutation is not None:
            assert cat.segments.to_list() == attach_cat.segments.to_list()
            assert cat.permutation.to_list() == attach_cat.permutation.to_list()
        assert isinstance(attach_cat, ak.Categorical)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            cat.register("AlternateName")

        # validate error handling for attempt to reuse name
        cat2 = ak.Categorical(ak.array(["a", "a", "b"]))
        with pytest.raises(RuntimeError):
            cat2.register(reg_name)

        cat.unregister()
        assert not cat.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_index_registration(self, dtype, size):
        reg_name = "MyIndex"
        i = ak.Index(self.make_pdarray(dtype, size))
        i.register(reg_name)

        # validate registered_name set
        assert i.registered_name == reg_name
        assert i.is_registered()

        reg = ak.list_registry()

        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert i.values.name in reg["Components"]

        # validate attach
        attach_i = ak.attach(reg_name)
        assert attach_i.is_registered()
        assert i.registered_name == attach_i.registered_name
        assert i.to_list() == attach_i.to_list()
        assert isinstance(attach_i, ak.Index)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            i.register("AlternateName")

        # validate error handling for attempt to reuse name
        i2 = ak.Index(self.make_pdarray(dtype, size))
        with pytest.raises(RuntimeError):
            i2.register(reg_name)

        i.unregister()
        assert not i.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_multi_index_registration(self, size):
        reg_name = "MyIndex"
        i = ak.Index.factory(
            [
                self.make_pdarray(ak.int64, size),
                self.make_pdarray(ak.str_, size),
                ak.Categorical(self.make_pdarray(ak.str_, size)),
            ]
        )
        i.register(reg_name)

        # validate registered_name set
        assert i.registered_name == reg_name
        assert i.is_registered()

        reg = ak.list_registry()

        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        for x in i.levels:
            if x.objType == ak.Categorical.objType:
                assert x.codes.name in reg["Components"]
                assert x.categories.name in reg["Components"]
                assert x._akNAcode.name in reg["Components"]
                if x.segments is not None and x.permutation is not None:
                    assert x.segments.name in reg["Components"]
                    assert x.permutation.name in reg["Components"]
            else:
                assert x.name in reg["Components"]

        # validate attach
        attach_i = ak.attach(reg_name)
        assert attach_i.is_registered()
        assert i.registered_name == attach_i.registered_name
        assert i.to_list() == attach_i.to_list()
        assert isinstance(attach_i, ak.MultiIndex)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            i.register("AlternateName")

        # validate error handling for attempt to reuse name
        i2 = ak.Index.factory([ak.arange(10), ak.randint(0, 100, 10)])
        with pytest.raises(RuntimeError):
            i2.register(reg_name)

        i.unregister()
        assert not i.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_series_registration(self, dtype, size):
        reg_name = "MySeries"
        s = ak.Series(self.make_pdarray(dtype, size))
        s.register(reg_name)

        assert s.is_registered()
        # validate registered_name set
        assert s.registered_name == reg_name

        reg = ak.list_registry()

        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert s.index.values.name in reg["Components"]
        assert s.values.name in reg["Components"]

        # validate attach
        attach_s = ak.attach(reg_name)
        assert attach_s.is_registered()
        assert s.registered_name == attach_s.registered_name
        assert s.to_list() == attach_s.to_list()
        assert isinstance(attach_s, ak.Series)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            s.register("AlternateName")

        # validate error handling for attempt to reuse name
        s2 = ak.Series(self.make_pdarray(dtype, size))
        with pytest.raises(RuntimeError):
            s2.register(reg_name)

        s.unregister()
        assert not s.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_bitvector_registration(self, size):
        reg_name = "MyBitVector"
        b = ak.BitVector(self.make_pdarray(ak.int64, size))
        b.register(reg_name)

        # validate registered_name set
        assert b.registered_name == reg_name
        assert b.is_registered()

        reg = ak.list_registry()

        # assert that the object is registered
        assert reg_name in reg["Objects"]
        # assert that the sym entry name is recorded
        assert b.values.name in reg["Components"]

        # validate attach
        attach_b = ak.attach(reg_name)
        assert attach_b.is_registered()
        assert b.registered_name, attach_b.registered_name
        assert b.to_list() == attach_b.to_list()
        assert isinstance(attach_b, ak.BitVector)

        # validate error handling for double registration
        with pytest.raises(RegistrationError):
            b.register("AlternateName")

        # validate error handling for attempt to reuse name
        b2 = ak.BitVector(self.make_pdarray(ak.int64, size))
        with pytest.raises(RuntimeError):
            b2.register(reg_name)

        b.unregister()
        assert not b.is_registered()
        reg = ak.list_registry()
        assert len(reg["Objects"]) == 0
        assert len(reg["Components"]) == 0

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_registered_component(self, dtype, size):
        a = self.make_pdarray(dtype, size)
        s = ak.SegArray(ak.arange(0, size * 2, 2), self.make_pdarray(dtype, size * 2))
        df = ak.DataFrame(
            {
                "array": a,
                "SegArray": s,
            }
        )

        s.register("MySegArray")
        df.register("MyDataFrame")

        # verify that a is not a registered object, but is a registered component
        assert a.is_registered()

        # verify that the 2 registered entries are registered
        assert s.is_registered()
        assert df.is_registered()

        # verify that components seen as registered after original unregistered
        s.unregister()
        assert df["SegArray"].values.is_registered()

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_error_handling(self, dtype):
        a = self.make_pdarray(dtype, 100)

        with pytest.raises(TypeError):
            a.register(7)

        with pytest.raises(TypeError):
            ak.attach(7)

        with pytest.raises(TypeError):
            ak.is_registered(7)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_clear(self, dtype):
        a = self.make_pdarray(dtype, 10)
        b = self.make_pdarray(dtype, 15)
        a.register("MyArray")

        assert a.is_registered()
        assert not b.is_registered()

        sym_tab = ak.list_symbol_table()
        assert a.name in sym_tab
        assert b.name in sym_tab

        ak.clear()

        sym_tab = ak.list_symbol_table()
        assert a.name in sym_tab
        assert b.name not in sym_tab

        assert a.is_registered()

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_list_registry(self, dtype):
        a = self.make_pdarray(dtype, 10)
        s = ak.SegArray(ak.arange(0, 10, 2), self.make_pdarray(dtype, 10))

        a.register("MyArray")
        s.register("MySeg")

        reg = ak.list_registry()  # access registry without object types

        # verify the objects exist in the register
        assert a.registered_name in reg["Objects"]
        assert s.registered_name in reg["Objects"]
        assert a.name in reg["Components"]
        assert s.segments.name in reg["Components"]
        assert s.values.name in reg["Components"]

        reg = ak.list_registry(detailed=True)  # access registry with object types
        assert (a.registered_name, a.objType.upper()) in reg["Objects"]
        assert (s.registered_name, s.objType.upper()) in reg["Objects"]
        assert a.name in reg["Components"]
        assert s.segments.name in reg["Components"]
        assert s.values.name in reg["Components"]

    def test_register_attach_unregister_all(self):
        a = self.make_pdarray(ak.float64, 10)
        s = self.make_pdarray(ak.str_, 20)
        c = ak.Categorical(s)

        ak.register_all({"MyArray": a, "MyStrings": s, "MyCat": c})

        # validate that all objects are registered
        assert a.registered_name == "MyArray"
        assert s.registered_name == "MyStrings"
        assert c.registered_name == "MyCat"

        reg = ak.list_registry()
        assert a.registered_name in reg["Objects"]
        assert s.registered_name in reg["Objects"]
        assert c.registered_name in reg["Objects"]

        # validate that all objects are attached
        att = ak.attach_all(["MyArray", "MyCat"])
        assert "MyArray" in att
        assert "MyCat" in att
        assert isinstance(att["MyArray"], ak.pdarray)
        assert isinstance(att["MyCat"], ak.Categorical)
        assert att["MyArray"].to_list() == a.to_list()
        assert att["MyCat"].to_list() == c.to_list()

        # validate all objects are unregistered
        ak.unregister_all(["MyStrings", "MyArray"])
        reg = ak.list_registry()
        assert "MyArray" not in reg["Objects"]
        assert "MyStrings" not in reg["Objects"]
        assert "MyCat" in reg["Objects"]

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_attach_weak_binding(self, dtype):
        """
        Ultimately pdarrayclass issues delete calls to the server when a bound object goes out of scope,
        if you bind to a server object more than once and one of those goes out of scope it affects
        all other references to it.
        """
        a = self.make_pdarray(dtype, 10).register("a_reg")
        assert str(a)
        b = ak.attach("a_reg")
        b.unregister()
        b = None  # Force out of scope
        with pytest.raises(RuntimeError):
            str(a)

    def test_symentry_cleanup(self):
        pda = self.make_pdarray(ak.int64, 10)
        assert len(ak.list_symbol_table()) > 0
        pda = None
        assert len(ak.list_symbol_table()) == 0

        s = self.make_pdarray(ak.str_, 10)
        assert len(ak.list_symbol_table()) > 0
        s = None
        assert len(ak.list_symbol_table()) == 0

        cat = ak.Categorical(self.make_pdarray(ak.str_, 10))
        assert len(ak.list_symbol_table()) > 0
        cat = None
        assert len(ak.list_symbol_table()) == 0

        seg = ak.SegArray(ak.arange(0, 10, 2), self.make_pdarray(ak.float64, 10))
        assert len(ak.list_symbol_table()) > 0
        seg = None
        assert len(ak.list_symbol_table()) == 0

        g = ak.GroupBy(
            [
                self.make_pdarray(ak.int64, 10),
                self.make_pdarray(ak.str_, 10),
                ak.Categorical(self.make_pdarray(ak.str_, 10)),
            ]
        )
        assert len(ak.list_symbol_table()) > 0
        g = None
        assert len(ak.list_symbol_table()) == 0

        d = ak.DataFrame(
            {
                "pda": self.make_pdarray(ak.int64, 10),
                "s": self.make_pdarray(ak.str_, 10),
                "cat": ak.Categorical(self.make_pdarray(ak.str_, 10)),
                "seg": ak.SegArray(ak.arange(0, 20, 2), self.make_pdarray(ak.uint64, 20)),
            }
        )
        assert len(ak.list_symbol_table()) > 0
        d = None
        assert len(ak.list_symbol_table()) == 0
