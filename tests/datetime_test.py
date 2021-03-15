from base_test import ArkoudaTest
import arkouda as ak
import pandas as pd
import numpy as np
import logging

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
    
    def test_noop_creation(self):
        self.assertTrue((ak.Datetime(self.dtvec1) == self.dtvec1).all())
        self.assertTrue((ak.Timedelta(self.tdvec1) == self.tdvec1).all())
    
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
        vectors = {ak.Datetime: self.dtvec1,
                   ak.Timedelta: self.tdvec1,
                   ak.pdarray: ak.arange(100)}
        pdvectors = {ak.Datetime: pd.to_datetime(self.dtvec1.to_ndarray()),
                     ak.Timedelta: pd.to_timedelta(self.tdvec1.to_ndarray()),
                     ak.pdarray: pd.Series(ak.arange(100).to_ndarray())}
        scalars = {ak.Datetime: self.dtscalar,
                   ak.Timedelta: self.onesecond,
                   ak.pdarray: 5}
        metrics = {'ak_supported': 0,
                   'ak_not_supported': 0,
                   'ak_yes_pd_no':0}
        for (firstclass, op, secondclass), (is_supported, r_is_supported, return_type) in build_op_table().items():
            fcvec = vectors[firstclass]
            pdfcvec = pdvectors[firstclass]
            fcsca = scalars[firstclass]
            scvec = vectors[secondclass]
            pdscvec = pdvectors[secondclass]
            scsca = scalars[secondclass]
            if not is_supported:
                with self.assertRaises(TypeError):
                    eval("fcvec {} scvec".format(op))
                with self.assertRaises(TypeError):
                    eval("fcvec {} scsca".format(op))
                metrics['ak_not_supported'] += 1
            else:
                compareflag = True
                ret = eval("fcvec {} scvec".format(op))
                self.assertTrue(isinstance(ret, return_type))
                metrics['ak_supported'] += 1
                try:
                    pdret = eval("pdfcvec {} pdscvec".format(op))
                except TypeError:
                    logging.getLogger().info("Pandas does not support {} {} {}".format(firstclass.__name__, op, secondclass.__name__))
                    metrics['ak_yes_pd_no'] += 1
                    compareflag = False
                if compareflag:
                    # Arkouda currently does not handle NaT, so replace with zero
                    if pdret.dtype.kind == 'm':
                        pdret = pd.Series(pdret).fillna(pd.Timedelta(seconds=0))
                    else:
                        pdret = pd.Series(pdret).fillna(pd.Timestamp(0))
                    try:
                        self.assertTrue((pdret.values == ret.to_ndarray()).all())
                    except AssertionError as e:
                        logging.getLogger().error("arkouda vs pandas discrepancy in {} {} {}:\n {} {}".format(firstclass.__name__, op, secondclass.__name__, ret, pdret))
                        raise e

                compareflag = True
                ret = eval("fcvec {} scsca".format(op))
                self.assertTrue(isinstance(ret, return_type))
                try:
                    pdret = eval("pdfcvec {} scsca".format(op))
                except TypeError:
                    logging.getLogger().info("Pandas does not support {} {} {}".format(firstclass.__name__, op, secondclass.__name__))
                    compareflag = False
                if compareflag:
                    self.assertTrue((pd.Series(pdret).values == ret.to_ndarray()).all())

            if not r_is_supported:
                with self.assertRaises(TypeError):
                    eval("scsca {} fcvec".format(op))
                metrics['ak_not_supported'] += 1
            else:
                try:
                    ret = eval("scsca {} fcvec".format(op))
                except Exception as e:
                    raise TypeError("{} scalar {} {}".format(secondclass, op, firstclass)) from e
                self.assertTrue(isinstance(ret, return_type))
                metrics['ak_supported'] += 1
                compareflag = True
                try:
                    pdret = eval("scsca {} pdfcvec".format(op))
                except TypeError:
                    logging.getLogger().info("Pandas does not support {}(scalar) {} {}".format(secondclass.__name__, op, firstclass.__name__))
                    metrics['ak_yes_pd_no'] += 1
                    compareflag = False
                if compareflag:
                    try:
                        self.assertTrue((pd.Series(pdret).values == ret.to_ndarray()).all())
                    except AttributeError:
                        logging.getLogger().error("Unexpected pandas return: {}(scalar) {} {} -> {}: {}".format(secondclass, op, firstclass, type(pdret), pdret))
        logging.getLogger().info("{}".format(metrics.items()))

    def test_floor(self):
        onemin = pd.Timedelta(1, unit='min')
        floor = self.dtvec1.floor('m')
        floor2 = self.dtvec1.to_pandas().floor('min')
        self.assertTrue(isinstance(floor, ak.Datetime))
        self.assertTrue((floor.to_pandas() == floor2).all())

    def test_ceil(self):
        onemin = pd.Timedelta(1, unit='min')
        ceil = self.dtvec1.ceil('m')
        ceil2 = self.dtvec1.to_pandas().ceil('min')
        self.assertTrue(isinstance(ceil, ak.Datetime))
        self.assertTrue((ceil.to_pandas() == ceil2).all())

    def test_round(self):
        onemin = pd.Timedelta(1, unit='min')
        rd = self.dtvec1.round('m')
        rd2 = self.dtvec1.to_pandas().round('min')
        self.assertTrue(isinstance(rd, ak.Datetime))
        try:
            self.assertTrue((rd.to_pandas() == rd2).all())
        except AssertionError:
            logging.getLogger().error("{} values unequal".format((rd.to_pandas() != rd2).sum()))
            logging.getLogger().info("{} vs {}".format(rd.to_pandas(), rd2))

    def test_groupby(self):
        g = ak.GroupBy([self.dtvec1, self.tdvec1])
        self.assertTrue(isinstance(g.unique_keys[0], ak.Datetime))
        self.assertTrue(isinstance(g.unique_keys[1], ak.Timedelta))
        self.assertTrue(g.unique_keys[0].is_sorted())

    def test_reductions(self):
        self.assertEqual(self.dtvec1.min(), self.dtvec1[0])
        self.assertEqual(self.dtvec1.max(), self.dtvec1[-1])
        self.assertEqual(self.dtvec1.argmin(), 0)
        self.assertEqual(self.dtvec1.argmax(), self.dtvec1.size-1)
        with self.assertRaises(TypeError):
            self.dtvec1.sum()
        self.assertEqual(self.tdvec1.min(), self.onesecond)
        self.assertEqual(self.tdvec1.max(), self.onesecond)
        self.assertEqual(self.tdvec1.argmin(), 0)
        self.assertEqual(self.tdvec1.argmax(), 0)
        self.assertEqual(self.tdvec1.sum(), pd.Timedelta(self.tdvec1.size, unit='s'))
        self.assertTrue(((-self.tdvec1).abs() == self.tdvec1).all())

    def test_scalars(self):
        self.assertTrue((self.dtscalar <= self.dtvec1).all()) # pandas.Timestamp
        self.assertTrue((self.dtscalar.to_pydatetime() <= self.dtvec1).all()) # datetime.datetime
        # self.assertTrue((self.dtscalar.to_numpy() <= self.dtvec1).all()) # numpy.datetime64
        self.assertTrue((self.onesecond == self.tdvec1).all()) # pandas.Timedelta
        self.assertTrue((self.onesecond.to_pytimedelta() == self.tdvec1).all()) # datetime.timedelta
        # self.assertTrue((self.onesecond.to_numpy() == self.tdvec1).all()) # numpy.timedelta64

    def test_units(self):
        def check_equal(pdunit, akunit):
            pdval = pd.Timestamp(1, unit=pdunit)
            akval = ak.Datetime(ak.ones(10, dtype=ak.int64), unit=akunit)[0]
            try:
                self.assertEqual(pdval, akval)
            except AssertionError:
                logging.getLogger().error("pandas {} ({}) != arkouda {} ({})".format(pdunit, pdval, akunit, akval))
            pdval = pd.Timedelta(1, unit=pdunit)
            akval = ak.Timedelta(ak.ones(10, dtype=ak.int64), unit=akunit)[0]
            try:
                self.assertEqual(pdval, akval)
            except AssertionError:
                logging.getLogger().error("pandas {} ({}) != arkouda {} ({})".format(pdunit, pdval, akunit, akval))

        unitmap = {'W': ('weeks', 'w', 'week'),
                   'D': ('days', 'd', 'day'),
                   'h': ('hours', 'H', 'hr', 'hrs'),
                   'T': ('minutes', 'minute', 'min', 'm'),
                   'L': ('milliseconds', 'millisecond', 'milli', 'ms', 'l'),
                   'U': ('microseconds', 'microsecond', 'micro', 'us', 'u'),
                   'N': ('nanoseconds', 'nanosecond', 'nano', 'ns', 'n')}
        for pdunit, aliases in unitmap.items():
            #check_equal(pdunit, pdunit)
            for akunit in (pdunit,)+aliases:
                check_equal(pdunit, akunit)
                
