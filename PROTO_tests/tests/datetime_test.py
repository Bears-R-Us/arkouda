import numpy as np
import pandas as pd
import pytest

import arkouda as ak


def build_op_table():
    ALL_OPS = ak.pdarray.BinOps - set(("<<<", ">>>"))
    table = {}
    for op in ALL_OPS:
        for firstclass in (ak.Datetime, ak.Timedelta):
            for secondclass in (ak.Datetime, ak.Timedelta, ak.pdarray):
                is_supported = op in getattr(
                    firstclass, f"supported_with_{secondclass.__name__.lower()}"
                )
                return_type = firstclass._get_callback(secondclass.__name__, op)
                if return_type is ak.timeclass._identity:
                    return_type = ak.pdarray
                r_is_supported = op in getattr(
                    firstclass, f"supported_with_r_{secondclass.__name__.lower()}"
                )
                table[(firstclass, op, secondclass)] = (is_supported, r_is_supported, return_type)
    return table


class TestDatetime:
    @classmethod
    def setup_class(cls):
        cls.dt_vec1 = ak.date_range(start="2021-01-01 12:00:00", periods=100, freq="s")
        cls.dt_vec2 = ak.Datetime(pd.date_range("2021-01-01 12:00:00", periods=100, freq="s"))
        cls.dt_scalar = pd.Timestamp("2021-01-01 12:00:00")
        cls.td_vec1 = ak.timedelta_range(start="1 second", end="1 second", periods=100)
        cls.td_vec2 = ak.Timedelta(ak.ones(100, dtype=ak.int64), unit="s")
        cls.one_second = pd.Timedelta(1, unit="s")

    def test_creation(self):
        assert (self.dt_vec1 == self.dt_vec2).all()
        assert (self.td_vec1 == self.td_vec2).all()

    def test_noop_creation(self):
        assert (ak.Datetime(self.dt_vec1) == self.dt_vec1).all()
        assert (ak.Timedelta(self.td_vec1) == self.td_vec1).all()

    def test_roundtrip(self):
        assert (ak.Datetime(self.dt_vec1.to_ndarray()) == self.dt_vec1).all()

    def test_plus_minus(self):
        # Datetime + Datetime not supported
        with pytest.raises(TypeError):
            self.dt_vec1 + self.dt_vec2

        # Datetime slice -> Datetime
        leading = self.dt_vec1[1:]
        trailing = self.dt_vec1[:-1]
        trange = ak.timedelta_range(start=0, periods=100, freq="s")
        assert isinstance(leading, ak.Datetime) and isinstance(trailing, ak.Datetime)

        # Datetime - Datetime -> Timedelta
        diff = leading - trailing
        assert isinstance(diff, ak.Timedelta)
        assert (diff == self.one_second).all()

        # Datetime - DatetimeScalar -> Timedelta
        diff = self.dt_vec1 - self.dt_scalar
        assert isinstance(diff, ak.Timedelta)
        assert (diff == trange).all()

        # DatetimeScalar - Datetime -> Timedelta
        diff = self.dt_scalar - self.dt_vec1
        assert isinstance(diff, ak.Timedelta)
        assert (diff == (-trange)).all()

        # Datetime + TimedeltaScalar -> Datetime
        # TimedeltaScalar + Datetime -> Datetime
        # Datetime + Timedelta -> Datetime
        # Timedelta + Datetime -> Datetime
        for t in (
            trailing + self.one_second,
            self.one_second + trailing,
            trailing + self.td_vec1[1:],
            self.td_vec1[1:] + trailing,
        ):
            assert isinstance(t, ak.Datetime)
            assert (t == leading).all()

        # Datetime - TimedeltaScalar -> Datetime
        # Datetime - Timedelta -> Datetime
        for t in leading - self.one_second, leading - self.td_vec1[1:]:
            assert isinstance(t, ak.Datetime)
            assert (t == trailing).all()

        # Timedelta + Timedelta -> Timedelta
        # Timedelta + TimedeltaScalar -> Timedelta
        for t in self.td_vec1 + self.td_vec1, self.td_vec1 + self.one_second:
            assert isinstance(t, ak.Timedelta)
            assert (t == ak.Timedelta(ak.full(100, 2, dtype=ak.int64), unit="s")).all()

        # Timedelta - Timedelta -> Timedelta
        # Timedelta - TimedeltaScalar -> Timedelta
        for t in self.td_vec1 - self.td_vec1, self.td_vec1 - self.one_second:
            assert isinstance(t, ak.Timedelta)
            assert (t == ak.Timedelta(ak.zeros(100, dtype=ak.int64), unit="s")).all()

    def test_op_types(self, verbose=pytest.verbose):
        vectors = {ak.Datetime: self.dt_vec1, ak.Timedelta: self.td_vec1, ak.pdarray: ak.arange(100)}
        pdvectors = {
            ak.Datetime: pd.to_datetime(self.dt_vec1.to_ndarray()),
            ak.Timedelta: pd.to_timedelta(self.td_vec1.to_ndarray()),
            ak.pdarray: pd.Series(ak.arange(100).to_ndarray()),
        }
        scalars = {ak.Datetime: self.dt_scalar, ak.Timedelta: self.one_second, ak.pdarray: 5}
        metrics = {"ak_supported": 0, "ak_not_supported": 0, "ak_yes_pd_no": 0}
        for (firstclass, op, secondclass), (
            is_supported,
            r_is_supported,
            return_type,
        ) in build_op_table().items():
            fcvec = vectors[firstclass]  # noqa: F841
            pdfcvec = pdvectors[firstclass]  # noqa: F841
            scvec = vectors[secondclass]  # noqa: F841
            pdscvec = pdvectors[secondclass]  # noqa: F841
            scsca = scalars[secondclass]  # noqa: F841
            if not is_supported:
                with pytest.raises(TypeError):
                    eval(f"fcvec {op} scvec")
                with pytest.raises(TypeError):
                    eval(f"fcvec {op} scsca")
                metrics["ak_not_supported"] += 1
            else:
                compare_flag = True
                ret = eval(f"fcvec {op} scvec")
                assert isinstance(ret, return_type)
                metrics["ak_supported"] += 1
                try:
                    pdret = eval(f"pdfcvec {op} pdscvec")
                except TypeError:
                    if verbose:
                        print(
                            f"Pandas does not support {firstclass.__name__} {op} {secondclass.__name__}"
                        )
                    metrics["ak_yes_pd_no"] += 1
                    compare_flag = False
                if compare_flag:
                    # Arkouda currently does not handle NaT, so replace with zero
                    if pdret.dtype.kind == "m":
                        pdret = pd.Series(pdret).fillna(pd.Timedelta(seconds=0))
                    else:
                        pdret = pd.Series(pdret).fillna(pd.Timestamp(0))
                    try:
                        assert (pdret.values == ret.to_ndarray()).all()
                    except AssertionError as e:
                        if verbose:
                            print(
                                f"arkouda vs pandas discrepancy in {firstclass.__name__}"
                                f" {op} {secondclass.__name__}:\n {ret} {pdret}"
                            )
                        raise e

                compare_flag = True
                ret = eval(f"fcvec {op} scsca")
                assert isinstance(ret, return_type)
                try:
                    pdret = eval(f"pdfcvec {op} scsca")
                except TypeError:
                    if verbose:
                        print(
                            f"Pandas does not support {firstclass.__name__} {op} {secondclass.__name__}"
                        )
                    compare_flag = False
                if compare_flag:
                    assert (pd.Series(pdret).values == ret.to_ndarray()).all()

            if not r_is_supported:
                with pytest.raises(TypeError):
                    eval(f"scsca {op} fcvec")
                metrics["ak_not_supported"] += 1
            else:
                try:
                    ret = eval(f"scsca {op} fcvec")
                except Exception as e:
                    raise TypeError(f"{secondclass} scalar {op} {firstclass}") from e
                assert isinstance(ret, return_type)
                metrics["ak_supported"] += 1
                compare_flag = True
                try:
                    pdret = eval(f"scsca {op} pdfcvec")
                except TypeError:
                    if verbose:
                        print(
                            f"Pandas does not support {secondclass.__name__}(scalar) "
                            f"{op} {firstclass.__name__}"
                        )
                    metrics["ak_yes_pd_no"] += 1
                    compare_flag = False
                if compare_flag:
                    try:
                        assert (pd.Series(pdret).values == ret.to_ndarray()).all()
                    except AttributeError:
                        if verbose:
                            print(
                                f"Unexpected pandas return: {secondclass}(scalar) "
                                f"{op} {firstclass} -> {type(pdret)}: {pdret}"
                            )
        if verbose:
            print(f"{metrics.items()}")

    def test_round(self):
        for fn in "floor", "ceil", "round":
            rounded = getattr(self.dt_vec1, fn)("m")
            assert isinstance(rounded, ak.Datetime)
            assert (rounded.to_pandas() == getattr(self.dt_vec1.to_pandas(), fn)("min")).all()

    def test_groupby(self):
        g = ak.GroupBy([self.dt_vec1, self.td_vec1])
        assert isinstance(g.unique_keys[0], ak.Datetime)
        assert isinstance(g.unique_keys[1], ak.Timedelta)
        assert g.unique_keys[0].is_sorted()

    def test_reductions(self):
        assert self.dt_vec1.min() == self.dt_vec1[0]
        assert self.dt_vec1.max() == self.dt_vec1[-1]
        assert self.dt_vec1.argmin() == 0
        assert self.dt_vec1.argmax() == self.dt_vec1.size - 1
        with pytest.raises(TypeError):
            self.dt_vec1.sum()

        assert self.td_vec1.min() == self.one_second
        assert self.td_vec1.max() == self.one_second
        assert self.td_vec1.argmin() == 0
        assert self.td_vec1.argmax() == 0
        assert self.td_vec1.sum() == pd.Timedelta(self.td_vec1.size, unit="s")
        assert ((-self.td_vec1).abs() == self.td_vec1).all()

    def test_timedel_std(self):
        # pandas std uses unbiased estimator, so to compare we set ddof=1 for arkouda std
        ak_timedel_std = ak.Timedelta(ak.array([123, 456, 789]), unit="s").std(ddof=1)
        pd_timedel_std = pd.to_timedelta([123, 456, 789], unit="s").std()
        assert ak_timedel_std == pd_timedel_std

    def test_scalars(self):
        for scal in self.dt_scalar, self.dt_scalar.to_pydatetime(), self.dt_scalar.to_numpy():
            assert (scal <= self.dt_vec1).all()
        for scal in self.one_second, self.one_second.to_pytimedelta(), self.one_second.to_numpy():
            assert (scal == self.td_vec1).all()

    def test_units(self, verbose=pytest.verbose):
        unitmap = {
            "W": ("weeks", "w", "week"),
            "D": ("days", "d", "day"),
            "h": ("hours", "H", "hr", "hrs"),
            "m": ("minutes", "minute", "min", "m"),
            "ms": ("milliseconds", "millisecond", "milli", "ms", "l"),
            "us": ("microseconds", "microsecond", "micro", "us", "u"),
            "ns": ("nanoseconds", "nanosecond", "nano", "ns", "n"),
        }
        for pdunit, aliases in unitmap.items():
            for akunit in (pdunit,) + aliases:
                for pdclass, akclass in (pd.Timestamp, ak.Datetime), (pd.Timedelta, ak.Timedelta):
                    pdval = pdclass(1, unit=pdunit)
                    akval = akclass(ak.ones(10, dtype=ak.int64), unit=akunit)[0]
                    try:
                        assert pdval == akval
                    except AssertionError:
                        if verbose:
                            print(f"pandas {pdunit} ({pdval}) != arkouda {akunit} ({akval})")

    def date_time_attribute_helper(self, pd_dt, ak_dt):
        assert (pd_dt.date == ak_dt.date.to_pandas()).all()

        for attr_name in (
            "nanosecond",
            "microsecond",
            "second",
            "minute",
            "hour",
            "day",
            "month",
            "year",
            "day_of_week",
            "dayofweek",
            "weekday",
            "day_of_year",
            "dayofyear",
            "is_leap_year",
        ):
            assert getattr(pd_dt, attr_name).to_list() == getattr(ak_dt, attr_name).to_list()

        assert pd_dt.isocalendar().week.to_list() == ak_dt.week.to_list()
        assert pd_dt.isocalendar().week.to_list() == ak_dt.weekofyear.to_list()
        assert ((pd_dt.isocalendar() == ak_dt.isocalendar().to_pandas()).all()).all()

    def test_date_time_accessors(self):
        self.date_time_attribute_helper(
            pd.Series(pd.date_range("2021-01-01 00:00:00", periods=100)).dt,
            ak.Datetime(ak.date_range("2021-01-01 00:00:00", periods=100)),
        )
        self.date_time_attribute_helper(
            pd.Series(pd.date_range("2000-01-01 12:00:00", periods=100, freq="d")).dt,
            ak.Datetime(ak.date_range("2000-01-01 12:00:00", periods=100, freq="d")),
        )
        self.date_time_attribute_helper(
            pd.Series(pd.date_range("1980-01-01 12:00:00", periods=100, freq="YE")).dt,
            ak.Datetime(ak.date_range("1980-01-01 12:00:00", periods=100, freq="YE")),
        )

    def time_delta_attribute_helper(self, pd_td, ak_td):
        assert ((pd_td.components == ak_td.components.to_pandas()).all()).all()
        assert np.allclose(pd_td.total_seconds(), ak_td.total_seconds().to_ndarray())
        for attr_name in "nanoseconds", "microseconds", "seconds", "days":
            assert getattr(pd_td, attr_name).to_list() == getattr(ak_td, attr_name).to_list()

    def test_time_delta_accessors(self):
        self.time_delta_attribute_helper(
            pd.Series(pd.to_timedelta(np.arange(10**12 + 1000, (10**12 + 1100)), unit="us")).dt,
            ak.Timedelta(ak.arange(10**12 + 1000, (10**12 + 1100)), unit="us"),
        )
        self.time_delta_attribute_helper(
            pd.Series(pd.to_timedelta(np.arange(10**12 + 1000, (10**12 + 1100)), unit="ns")).dt,
            ak.Timedelta(ak.arange(10**12 + 1000, (10**12 + 1100)), unit="ns"),
        )
        self.time_delta_attribute_helper(
            pd.Series(pd.to_timedelta(np.arange(2000, 2100), unit="W")).dt,
            ak.Timedelta(ak.arange(2000, 2100), unit="W"),
        )

    def test_woy_boundary(self):
        # make sure weeks at year boundaries are correct, modified version of pandas test at
        # https://github.com/pandas-dev/pandas/blob/main/pandas/tests/scalar/timestamp/test_timestamp.py
        for date in "2013-12-31", "2008-12-28", "2009-12-31", "2010-01-01", "2010-01-03":
            ak_week = ak.Datetime(ak.date_range(date, periods=10, freq="W")).week.to_list()
            pd_week = (
                pd.Series(pd.date_range(date, periods=10, freq="W")).dt.isocalendar().week.to_list()
            )
            assert ak_week == pd_week

        for date in "2000-01-01", "2005-01-01":
            ak_week = ak.Datetime(ak.date_range(date, periods=10, freq="d")).week.to_list()
            pd_week = (
                pd.Series(pd.date_range(date, periods=10, freq="d")).dt.isocalendar().week.to_list()
            )
            assert ak_week == pd_week
