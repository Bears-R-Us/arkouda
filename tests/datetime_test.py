from base_test import ArkoudaTest
import arkouda as ak
import pandas as pd
import numpy as np

def build_op_table():
    ALL_OPS = ak.pdarray.BinOps
    table = {}
    for op in ALL_OPS:
        for firstclass in (ak.Datetime, ak.Timedelta):
            for secondclass in (ak.Datetime, ak.Timedelta, ak.pdarray):
                is_supported = op in getattr(firstclass, 'supported_with_{}'.format(secondclass.__name__.lower()))
                return_type = firstclass._get_callback(secondclass.__name__, op)
                if return_type is ak.timeclass._identity:
                    return_type = ak.pdarray
                r_is_supported = op in getattr(firstclass, 'supported_with_r_{}'.format(secondclass.__name__.lower()))
                table[(firstclass, op, secondclass)] = (is_supported, r_is_supported, return_type)
    return table
        
class DatetimeTest(ArkoudaTest):
    def setUp(self):
        ArkoudaTest.setUp(self)
        self.dtvec1 = ak.date_range(start='2021-01-01 12:00:00', periods=100, freq='s')
        self.dtvec2 = ak.Datetime(pd.date_range('2021-01-01 12:00:00', periods=100, freq='s'))
        self.dtscalar = pd.Timestamp('2021-01-01 12:00:00')
        self.tdvec1 = ak.timedelta_range(start='1 second', end='1 second', periods=100)
        self.tdvec2 = ak.Timedelta(ak.ones(100, dtype=ak.int64), unit='s')
        self.onesecond = pd.Timedelta(1, unit='s')

    def test_creation(self):
        self.assertTrue((self.dtvec1 == self.dtvec2).all())
        self.assertTrue((self.tdvec1 == self.tdvec2).all())
        
    def test_roundtrip(self):
        d = ak.Datetime(self.dtvec1.to_ndarray())
        self.assertTrue((d == self.dtvec1).all())

    def test_plus_minus(self):
        # Datetime + Datetime not supported
        with self.assertRaises(TypeError) as cm:
            self.dtvec1 + self.dtvec2
        # Datetime slice -> Datetime
        leading = self.dtvec1[1:]
        trailing = self.dtvec1[:-1]
        self.assertTrue(isinstance(leading, ak.Datetime) and isinstance(trailing, ak.Datetime))
        # Datetime - Datetime -> Timedelta
        diff = leading - trailing
        self.assertTrue(isinstance(diff, ak.Timedelta))
        self.assertTrue((diff == self.onesecond).all())
        # Datetime - DatetimeScalar -> Timedelta
        diff = self.dtvec1 - self.dtscalar
        trange = ak.timedelta_range(start=0, periods=100, freq='s')
        self.assertTrue(isinstance(diff, ak.Timedelta))
        self.assertTrue((diff == trange).all())
        # DatetimeScalar - Datetime -> Timedelta
        diff = self.dtscalar - self.dtvec1
        self.assertTrue(isinstance(diff, ak.Timedelta))
        self.assertTrue((diff == (-trange)).all())
        # Datetime + TimedeltaScalar -> Datetime
        t = (trailing + self.onesecond)
        self.assertTrue(isinstance(t, ak.Datetime))
        self.assertTrue((t == leading).all())
        # TimedeltaScalar + Datetime -> Datetime
        t = (self.onesecond + trailing)
        self.assertTrue(isinstance(t, ak.Datetime))
        self.assertTrue((t == leading).all())
        # Datetime - TimedeltaScalar -> Datetime
        t = leading - self.onesecond
        self.assertTrue(isinstance(t, ak.Datetime))
        self.assertTrue((t == trailing).all())
        # Datetime + Timedelta -> Datetime
        t = (trailing + self.tdvec1[1:])
        self.assertTrue(isinstance(t, ak.Datetime))
        self.assertTrue((t == leading).all())
        # Timedelta + Datetime -> Datetime
        t = (self.tdvec1[1:] + trailing)
        self.assertTrue(isinstance(t, ak.Datetime))
        self.assertTrue((t == leading).all())
        # Datetime - Timedelat -> Datetime
        t = (leading - self.tdvec1[1:])
        self.assertTrue(isinstance(t, ak.Datetime))
        # Timedelta + Timedelta -> Timedelta
        t = self.tdvec1 + self.tdvec1
        self.assertTrue(isinstance(t, ak.Timedelta))
        self.assertTrue((t == ak.Timedelta(2*ak.ones(100, dtype=ak.int64), unit='s')).all())
        # Timedelta + TimedeltaScalar -> Timedelta
        t = self.tdvec1 + self.onesecond
        self.assertTrue(isinstance(t, ak.Timedelta))
        self.assertTrue((t == ak.Timedelta(2*ak.ones(100, dtype=ak.int64), unit='s')).all())
        # Timedelta - Timedelta -> Timedelta
        t = self.tdvec1 - self.tdvec1
        self.assertTrue(isinstance(t, ak.Timedelta))
        self.assertTrue((t == ak.Timedelta(ak.zeros(100, dtype=ak.int64), unit='s')).all())
        # Timedelta - TimedeltaScalar -> Timedelta
        t = self.tdvec1 - self.onesecond
        self.assertTrue(isinstance(t, ak.Timedelta))
        self.assertTrue((t == ak.Timedelta(ak.zeros(100, dtype=ak.int64), unit='s')).all())

    def test_op_types(self):
        vectors = {}
        vectors[ak.Datetime] = self.dtvec1
        vectors[ak.Timedelta] = self.tdvec1
        vectors[ak.pdarray] = ak.arange(100)
        scalars = {}
        scalars[ak.Datetime] = self.dtscalar
        scalars[ak.Timedelta] = self.onesecond
        scalars[ak.pdarray] = 5
        for (firstclass, op, secondclass), (is_supported, r_is_supported, return_type) in build_op_table().items():
            fcvec = vectors[firstclass]
            fcsca = scalars[firstclass]
            scvec = vectors[secondclass]
            scsca = scalars[secondclass]
            if not is_supported:
                with self.assertRaises(TypeError):
                    eval("fcvec {} scvec".format(op))
                with self.assertRaises(TypeError):
                    eval("fcvec {} scsca".format(op))
            else:
                ret = eval("fcvec {} scvec".format(op))
                self.assertTrue(isinstance(ret, return_type))
                ret = eval("fcvec {} scsca".format(op))
                self.assertTrue(isinstance(ret, return_type))
            if not r_is_supported:
                with self.assertRaises(TypeError):
                    eval("scsca {} fcvec".format(op))
            else:
                ret = eval("scsca {} fcvec".format(op))
                self.assertTrue(isinstance(ret, return_type))
