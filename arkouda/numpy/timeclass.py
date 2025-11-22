import datetime
import json

from typing import TYPE_CHECKING, Optional, TypeVar, Union

import numpy as np

from pandas import Series as pdSeries
from pandas import Timedelta as pdTimedelta
from pandas import Timestamp as pdTimestamp
from pandas import date_range as pd_date_range
from pandas import timedelta_range as pd_timedelta_range
from pandas import to_datetime, to_timedelta

from arkouda.numpy.dtypes import int64, int_scalars, intTypes, isSupportedInt
from arkouda.numpy.pdarrayclass import RegistrationError, create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import from_series


if TYPE_CHECKING:
    from arkouda.client import generic_msg
else:
    generic_msg = TypeVar("generic_msg")

__all__ = [
    "Datetime",
    "Timedelta",
    "date_range",
    "timedelta_range",
]


_BASE_UNIT = "ns"

_unit2normunit = {
    "weeks": "w",
    "days": "d",
    "hours": "h",
    "hrs": "h",
    "minutes": "m",
    "t": "m",
    "milliseconds": "ms",
    "l": "ms",
    "microseconds": "us",
    "u": "us",
    "nanoseconds": "ns",
    "n": "ns",
}

_unit2factor = {
    "w": 7 * 24 * 60 * 60 * 10**9,
    "d": 24 * 60 * 60 * 10**9,
    "h": 60 * 60 * 10**9,
    "m": 60 * 10**9,
    "s": 10**9,
    "ms": 10**6,
    "us": 10**3,
    "ns": 1,
}


def _get_factor(unit: str) -> int:
    unit = unit.lower()
    if unit in _unit2factor:
        return _unit2factor[unit]
    else:
        for key, normunit in _unit2normunit.items():
            if key.startswith(unit):
                return _unit2factor[normunit]
        raise ValueError(
            f"Argument must be one of {set(_unit2factor.keys()) | set(_unit2normunit.keys())}"
        )


def _identity(x, **kwargs):
    return x


class _Timescalar:
    def __init__(self, scalar):
        if isinstance(scalar, np.datetime64) or isinstance(scalar, datetime.datetime):
            scalar = to_datetime(scalar).to_numpy()
        elif isinstance(scalar, np.timedelta64) or isinstance(scalar, datetime.timedelta):
            scalar = to_timedelta(scalar).to_numpy()
        self.unit = np.datetime_data(scalar.dtype)[0]
        self._factor = _get_factor(self.unit)
        # int64 in nanoseconds
        self.value = self._factor * scalar.astype("int64")


class _AbstractBaseTime(pdarray):
    """Base class for Datetime and Timedelta; not user-facing. Arkouda handles
    time similar to Pandas (albeit with less functionality), in that all absolute
    and relative times are represented in nanoseconds as int64 behind the scenes.
    Datetime and Timedelta can be constructed from Arkouda, NumPy, or Pandas arrays;
    in each case, the input values are normalized to nanoseconds on initialization,
    so that all resulting operations are transparent.
    """

    special_objType = "Time"

    def __init__(self, pda, unit: str = _BASE_UNIT):
        from arkouda.numpy import cast as akcast

        if isinstance(pda, Datetime) or isinstance(pda, Timedelta):
            self.unit: str = pda.unit
            self._factor: int = pda._factor
            # Make a copy to avoid unknown symbol errors
            self.values: pdarray = akcast(pda.values, int64)
        # Convert the input to int64 pdarray of nanoseconds
        elif isinstance(pda, pdarray):
            if pda.dtype not in intTypes:
                raise TypeError(f"{self.__class__.__name__} array must have int64 dtype")
            # Already int64 pdarray, just scale
            self.unit = unit
            self._factor = _get_factor(self.unit)
            # This makes a copy of the input array, to leave input unchanged
            self.values = akcast(self._factor * pda, int64)  # Mimics a datetime64[ns] array
        elif hasattr(pda, "dtype"):
            # Handles all pandas and numpy datetime/timedelta arrays
            if pda.dtype.kind not in ("M", "m"):
                # M = datetime64, m = timedelta64
                raise TypeError(f"Invalid dtype: {pda.dtype.name}")
            if isinstance(pda, pdSeries):
                # Pandas Datetime and Timedelta
                # Get units of underlying numpy datetime64 array
                self.unit = np.datetime_data(pda.values.dtype)[0]  # type: ignore [arg-type]
                self._factor = _get_factor(self.unit)
                # Create pdarray
                self.values = from_series(pda)
                # Scale if necessary
                # This is futureproofing; it will not be used unless pandas
                # changes its Datetime implementation
                if self._factor != 1:
                    # Scale inplace because we already created a copy
                    self.values *= self._factor
            elif isinstance(pda, np.ndarray):
                # Numpy datetime64 and timedelta64
                # Force through pandas.Series
                self.__init__(to_datetime(pda).to_series())  # type: ignore
            elif hasattr(pda, "to_series"):
                # Pandas DatetimeIndex
                # Force through pandas.Series
                self.__init__(pda.to_series())  # type: ignore
        else:
            raise TypeError(f"Unsupported type: {type(pda)}")
        # Now that self.values is correct, init self with same metadata
        super().__init__(
            self.values.name,
            self.values.dtype,
            self.values.size,
            self.values.ndim,
            self.values.shape,
            self.values.itemsize,
        )
        self._data = self.values
        self._is_populated = False

    @classmethod
    def _get_callback(cls, other, op):
        # Will be overridden by all children
        return _identity

    def floor(self, freq):
        """Round times down to the nearest integer of a given frequency.

        Parameters
        ----------
        freq : str {'d', 'm', 'h', 's', 'ms', 'us', 'ns'}
            Frequency to round to

        Returns
        -------
        self.__class__
            Values rounded down to nearest frequency
        """
        f = _get_factor(freq)
        return self.__class__(self.values // f, unit=freq)

    def ceil(self, freq):
        """Round times up to the nearest integer of a given frequency.

        Parameters
        ----------
        freq : str {'d', 'm', 'h', 's', 'ms', 'us', 'ns'}
            Frequency to round to

        Returns
        -------
        self.__class__
            Values rounded up to nearest frequency
        """
        f = _get_factor(freq)
        return self.__class__((self.values + (f - 1)) // f, unit=freq)

    def round(self, freq):
        """Round times to the nearest integer of a given frequency. Midpoint
        values will be rounded to nearest even integer.

        Parameters
        ----------
        freq : str {'d', 'm', 'h', 's', 'ms', 'us', 'ns'}
            Frequency to round to

        Returns
        -------
        self.__class__
            Values rounded to nearest frequency
        """
        f = _get_factor(freq)
        offset = self.values + ((f + 1) // 2)
        rounded = offset // f
        # Halfway values are supposed to round to the nearest even integer
        # Need to figure out which ones ended up odd and fix them
        decrement = ((offset % f) == 0) & ((rounded % 2) == 1)
        rounded[decrement] = rounded[decrement] - 1
        return self.__class__(rounded, unit=freq)

    def to_ndarray(self):
        __doc__ = super().to_ndarray.__doc__  # noqa
        return np.array(
            self.values.to_ndarray(),
            dtype="{}64[ns]".format(self.__class__.__name__.lower()),
        )

    def tolist(self):
        __doc__ = super().tolist().__doc__  # noqa
        return self.to_ndarray().tolist()

    def to_hdf(
        self,
        prefix_path: str,
        dataset: str = "array",
        mode: str = "truncate",
        file_type: str = "distribute",
    ):
        """Override of the pdarray to_hdf to store the special dtype."""
        from arkouda.client import generic_msg
        from arkouda.pandas.io import _file_type_to_int, _mode_str_to_int

        return generic_msg(
            cmd="tohdf",
            args={
                "values": self,
                "dset": dataset,
                "write_mode": _mode_str_to_int(mode),
                "filename": prefix_path,
                "dtype": self.dtype,
                "objType": self.special_objType,
                "file_format": _file_type_to_int(file_type),
            },
        )

    def update_hdf(self, prefix_path: str, dataset: str = "array", repack: bool = True):
        """Override the pdarray implementation so that the special object type will be used."""
        from arkouda.client import generic_msg
        from arkouda.pandas.io import (
            _file_type_to_int,
            _get_hdf_filetype,
            _mode_str_to_int,
            _repack_hdf,
        )

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        generic_msg(
            cmd="tohdf",
            args={
                "values": self,
                "dset": dataset,
                "write_mode": _mode_str_to_int("append"),
                "filename": prefix_path,
                "dtype": self.dtype,
                "objType": self.special_objType,
                "file_format": _file_type_to_int(file_type),
                "overwrite": True,
            },
        )

        if repack:
            _repack_hdf(prefix_path)

    def __str__(self):
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            vals = [f"'{self[i]}'" for i in range(self.size)]
        else:
            vals = [f"'{self[i]}'" for i in range(3)]
            vals.append("... ")
            vals.extend([f"'{self[i]}'" for i in range(self.size - 3, self.size)])
        spaces = " " * (len(self.__class__.__name__) + 1)
        return "{}([{}],\n{}dtype='{}64[ns]')".format(
            self.__class__.__name__,
            ",\n{} ".format(spaces).join(vals),
            spaces,
            self.__class__.__name__.lower(),
        )

    def __repr__(self) -> str:
        return self.__str__()

    def _binop(self, other, op):
        # Need to do 2 things:
        #  1) Determine return type, based on other's class
        #  2) Get other's int64 data to combine with self's data
        if isinstance(other, Datetime) or self._is_datetime_scalar(other):
            if op not in self.supported_with_datetime:
                raise TypeError(f"{op} not supported between {self.__class__.__name__} and Datetime")
            otherclass = "Datetime"
            if self._is_datetime_scalar(other):
                otherdata = _Timescalar(other).value
            else:
                otherdata = other.values
        elif isinstance(other, Timedelta) or self._is_timedelta_scalar(other):
            if op not in self.supported_with_timedelta:
                raise TypeError(f"{op} not supported between {self.__class__.__name__} and Timedelta")
            otherclass = "Timedelta"
            if self._is_timedelta_scalar(other):
                otherdata = _Timescalar(other).value
            else:
                otherdata = other.values
        elif (isinstance(other, pdarray) and other.dtype in intTypes) or isSupportedInt(other):
            if op not in self.supported_with_pdarray:
                raise TypeError(f"{op} not supported between {self.__class__.__name__} and integer")
            otherclass = "pdarray"
            otherdata = other
        else:
            return NotImplemented
        # Determines return type (Datetime, Timedelta, or pdarray)
        callback = self._get_callback(otherclass, op)
        # Actual operation evaluates on the underlying int64 data
        return callback(self.values._binop(otherdata, op))

    def _r_binop(self, other, op):
        # Need to do 2 things:
        #  1) Determine return type, based on other's class
        #  2) Get other's int64 data to combine with self's data

        # First case is pdarray <op> self
        if isinstance(other, pdarray) and other.dtype in intTypes:
            if op not in self.supported_with_r_pdarray:
                raise TypeError(f"{op} not supported between int64 and {self.__class__.__name__}")
            callback = self._get_callback("pdarray", op)
            # Need to use other._binop because self.values._r_binop can only handle scalars
            return callback(other._binop(self.values, op))
        # All other cases are scalars, so can use self.values._r_binop
        elif self._is_datetime_scalar(other):
            if op not in self.supported_with_r_datetime:
                raise TypeError(
                    f"{op} not supported between scalar datetime and {self.__class__.__name__}"
                )
            otherclass = "Datetime"
            otherdata = _Timescalar(other).value
        elif self._is_timedelta_scalar(other):
            if op not in self.supported_with_r_timedelta:
                raise TypeError(
                    f"{op} not supported between scalar timedelta and {self.__class__.__name__}"
                )
            otherclass = "Timedelta"
            otherdata = _Timescalar(other).value
        elif isSupportedInt(other):
            if op not in self.supported_with_r_pdarray:
                raise TypeError(f"{op} not supported between int64 and {self.__class__.__name__}")
            otherclass = "pdarray"
            otherdata = other
        else:
            # If here, type is not handled
            return NotImplemented
        callback = self._get_callback(otherclass, op)
        return callback(self.values._r_binop(otherdata, op))

    def opeq(self, other, op):
        if isinstance(other, Timedelta) or self._is_timedelta_scalar(other):
            if op not in self.supported_opeq:
                raise TypeError(f"{self.__class__.__name__} {op} Timedelta not supported")
            if self._is_timedelta_scalar(other):
                otherdata = _Timescalar(other).value
            else:
                otherdata = other.values
            self.values.opeq(otherdata, op)
        elif isinstance(other, Datetime) or self._is_datetime_scalar(other):
            raise TypeError(f"{self.__class__.__name__} {op} datetime not supported")
        else:
            return NotImplemented

    @staticmethod
    def _is_datetime_scalar(scalar):
        return (
            isinstance(scalar, pdTimestamp)
            or (isinstance(scalar, np.datetime64) and np.isscalar(scalar))
            or isinstance(scalar, datetime.datetime)
        )

    @staticmethod
    def _is_timedelta_scalar(scalar):
        return (
            isinstance(scalar, pdTimedelta)
            or (isinstance(scalar, np.timedelta64) and np.isscalar(scalar))
            or isinstance(scalar, datetime.timedelta)
        )

    def _scalar_callback(self, key):
        # Will be overridden in all children
        return key

    def __getitem__(self, key):
        if isSupportedInt(key):
            # Single integer index will return a pandas scalar
            return self._scalar_callback(self.values[key])
        else:
            # Slice or array index should return same class
            return self.__class__(self.values[key])

    def __setitem__(self, key, value):
        # RHS can only be vector or scalar of same class
        if isinstance(value, self.__class__):
            # Value.values is already in nanoseconds, so self.values
            # can be set directly
            self.values[key] = value.values
        elif self._is_supported_scalar(value):
            # _Timescalar takes care of normalization to nanoseconds
            normval = _Timescalar(value)
            self.values[key] = normval.value
        else:
            return NotImplemented

    def min(self):
        __doc__ = super().min.__doc__  # noqa
        # Return type is pandas scalar
        return self._scalar_callback(self.values.min())

    def max(self):
        __doc__ = super().max.__doc__  # noqa
        # Return type is pandas scalar
        return self._scalar_callback(self.values.max())

    def mink(self, k):
        __doc__ = super().mink.__doc__  # noqa
        # Return type is same class
        return self.__class__(self.values.mink(k))

    def maxk(self, k):
        __doc__ = super().maxk.__doc__  # noqa
        # Return type is same class
        return self.__class__(self.values.maxk(k))


class Datetime(_AbstractBaseTime):
    """Represents a date and/or time.

    Datetime is the Arkouda analog to pandas DatetimeIndex and
    other timeseries data types.

    Parameters
    ----------
    pda : int64 pdarray, pd.DatetimeIndex, pd.Series, or np.datetime64 array
    unit : str, default 'ns'
        For int64 pdarray, denotes the unit of the input. Ignored for pandas
        and numpy arrays, which carry their own unit. Not case-sensitive;
        prefixes of full names (like 'sec') are accepted.

        Possible values:

        * 'weeks' or 'w'
        * 'days' or 'd'
        * 'hours' or 'h'
        * 'minutes', 'm', or 't'
        * 'seconds' or 's'
        * 'milliseconds', 'ms', or 'l'
        * 'microseconds', 'us', or 'u'
        * 'nanoseconds', 'ns', or 'n'

        Unlike in pandas, units cannot be combined or mixed with integers

    Notes
    -----
    The ``.values`` attribute is always in nanoseconds with int64 dtype.
    """

    supported_with_datetime = frozenset(("==", "!=", "<", "<=", ">", ">=", "-"))
    supported_with_r_datetime = frozenset(("==", "!=", "<", "<=", ">", ">=", "-"))
    supported_with_timedelta = frozenset(("+", "-", "/", "//", "%"))
    supported_with_r_timedelta = frozenset(("+"))
    supported_opeq = frozenset(("+=", "-="))
    supported_with_pdarray = frozenset(())  # type: ignore
    supported_with_r_pdarray = frozenset(())  # type: ignore

    special_objType = "Datetime"

    def _ensure_components(self):
        from arkouda.client import generic_msg

        if self._is_populated:
            return
        # lazy initialize all attributes in one server call
        attribute_dict = json.loads(generic_msg(cmd="dateTimeAttributes", args={"values": self.values}))
        self._ns = create_pdarray(attribute_dict["nanosecond"])
        self._us = create_pdarray(attribute_dict["microsecond"])
        self._ms = create_pdarray(attribute_dict["millisecond"])
        self._s = create_pdarray(attribute_dict["second"])
        self._min = create_pdarray(attribute_dict["minute"])
        self._hour = create_pdarray(attribute_dict["hour"])
        self._day = create_pdarray(attribute_dict["day"])
        self._month = create_pdarray(attribute_dict["month"])
        self._year = create_pdarray(attribute_dict["year"])
        self._iso_year = create_pdarray(attribute_dict["isoYear"])
        self._day_of_week = create_pdarray(attribute_dict["dayOfWeek"])
        self._week_of_year = create_pdarray(attribute_dict["weekOfYear"])
        self._day_of_year = create_pdarray(attribute_dict["dayOfYear"])
        self._is_leap_year = create_pdarray(attribute_dict["isLeapYear"])
        self._date = self.floor("d")
        self._is_populated = True

    @property
    def nanosecond(self):
        self._ensure_components()
        return self._ns

    @property
    def microsecond(self):
        self._ensure_components()
        return self._us

    @property
    def millisecond(self):
        self._ensure_components()
        return self._ms

    @property
    def second(self):
        self._ensure_components()
        return self._s

    @property
    def minute(self):
        self._ensure_components()
        return self._min

    @property
    def hour(self):
        self._ensure_components()
        return self._hour

    @property
    def day(self):
        self._ensure_components()
        return self._day

    @property
    def month(self):
        self._ensure_components()
        return self._month

    @property
    def year(self):
        self._ensure_components()
        return self._year

    @property
    def day_of_year(self):
        self._ensure_components()
        return self._day_of_year

    @property
    def dayofyear(self):
        return self.day_of_year

    @property
    def day_of_week(self):
        self._ensure_components()
        return self._day_of_week

    @property
    def dayofweek(self):
        return self.day_of_week

    @property
    def weekday(self):
        return self.day_of_week

    @property
    def week(self):
        self._ensure_components()
        return self._week_of_year

    @property
    def weekofyear(self):
        return self.week

    @property
    def date(self):
        # no need to call _ensure_components for the date
        # if _date has been set, return it. Otherwise set it first
        if not hasattr(self, "_date"):
            self._date = self.floor("d")
        return self._date

    @property
    def is_leap_year(self):
        self._ensure_components()
        return self._is_leap_year

    def isocalendar(self):
        from arkouda import DataFrame

        self._ensure_components()
        return DataFrame(
            {
                "year": self._iso_year,
                "week": self._week_of_year,
                "day": self._day_of_week + 1,
            }
        )

    @classmethod
    def _get_callback(cls, otherclass, op):
        callbacks = {
            ("Datetime", "-"): Timedelta,  # Datetime - Datetime -> Timedelta
            ("Timedelta", "+"): cls,  # Datetime + Timedelta -> Datetime
            ("Timedelta", "-"): cls,  # Datetime - Timedelta -> Datetime
            ("Timedelta", "%"): Timedelta,
        }  # Datetime % Timedelta -> Timedelta
        # Every other supported op returns an int64 pdarray, so callback is identity
        return callbacks.get((otherclass, op), _identity)

    def _scalar_callback(self, scalar):
        # Formats a scalar return value as pandas Timestamp
        return pdTimestamp(int(scalar), unit=_BASE_UNIT)

    @staticmethod
    def _is_supported_scalar(self, scalar):
        # Tests whether scalar has compatible type with self's elements
        return self.is_datetime_scalar(scalar)

    def to_pandas(self):
        """Convert array to a pandas DatetimeIndex. Note: if the array size
        exceeds client.maxTransferBytes, a RuntimeError is raised.

        See Also
        --------
        to_ndarray
        """
        return to_datetime(self.to_ndarray())

    def sum(self):
        raise TypeError("Cannot sum datetime64 values")

    def register(self, user_defined_name):
        """
        Register this Datetime object and underlying components with the Arkouda server.

        Parameters
        ----------
        user_defined_name : str
            user defined name the Datetime is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        Datetime
            The same Datetime which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different Datetimes with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the Datetimes with the user_defined_name

        See Also
        --------
        unregister, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        """
        from arkouda.client import generic_msg

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")
        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.special_objType,
                "array": self.values,
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this Datetime object in the arkouda server which was previously
        registered using register() and/or attached to using attach().

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See Also
        --------
        register, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        """
        from arkouda.numpy.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    def is_registered(self) -> np.bool_:
        """
         Return True iff the object is contained in the registry or is a component of a
         registered object.

        Returns
        -------
        numpy.bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mis-match of registered components

        See Also
        --------
        register, attach, unregister

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.numpy.util import is_registered

        if self.registered_name is None:
            return np.bool_(is_registered(self.values.name, as_component=True))
        else:
            return np.bool_(is_registered(self.registered_name))


class Timedelta(_AbstractBaseTime):
    """Represents a duration, the difference between two dates or times.

    Timedelta is the Arkouda equivalent of pandas.TimedeltaIndex.

    Parameters
    ----------
    pda : int64 pdarray, pd.TimedeltaIndex, pd.Series, or np.timedelta64 array
    unit : str, default 'ns'
        For int64 pdarray, denotes the unit of the input. Ignored for pandas
        and numpy arrays, which carry their own unit. Not case-sensitive;
        prefixes of full names (like 'sec') are accepted.

        Possible values:

        * 'weeks' or 'w'
        * 'days' or 'd'
        * 'hours' or 'h'
        * 'minutes', 'm', or 't'
        * 'seconds' or 's'
        * 'milliseconds', 'ms', or 'l'
        * 'microseconds', 'us', or 'u'
        * 'nanoseconds', 'ns', or 'n'

        Unlike in pandas, units cannot be combined or mixed with integers

    Notes
    -----
    The ``.values`` attribute is always in nanoseconds with int64 dtype.
    """

    supported_with_datetime = frozenset(("+"))
    supported_with_r_datetime = frozenset(("+", "-", "/", "//", "%"))
    supported_with_timedelta = frozenset(("==", "!=", "<", "<=", ">", ">=", "+", "-", "/", "//", "%"))
    supported_with_r_timedelta = frozenset(("==", "!=", "<", "<=", ">", ">=", "+", "-", "/", "//", "%"))
    supported_opeq = frozenset(("+=", "-=", "%="))
    supported_with_pdarray = frozenset(("*", "//"))
    supported_with_r_pdarray = frozenset(("*"))

    special_objType = "Timedelta"

    def _ensure_components(self):
        from arkouda.client import generic_msg

        if self._is_populated:
            return
        # lazy initialize all attributes in one server call
        attribute_dict = json.loads(generic_msg(cmd="timeDeltaAttributes", args={"values": self.values}))
        self._ns = create_pdarray(attribute_dict["nanosecond"])
        self._us = create_pdarray(attribute_dict["microsecond"])
        self._ms = create_pdarray(attribute_dict["millisecond"])
        self._s = create_pdarray(attribute_dict["second"])
        self._m = create_pdarray(attribute_dict["minute"])
        self._h = create_pdarray(attribute_dict["hour"])
        self._d = create_pdarray(attribute_dict["day"])
        self._nanoseconds = self._ns
        self._microseconds = self._ms * 1000 + self._us
        self._seconds = self._h * 3600 + self._m * 60 + self._s
        self._days = self._d
        self._total_seconds = self._days * (24 * 3600) + self._seconds + (self._microseconds / 10**6)
        self._is_populated = True

    @property
    def nanoseconds(self):
        self._ensure_components()
        return self._nanoseconds

    @property
    def microseconds(self):
        self._ensure_components()
        return self._microseconds

    @property
    def seconds(self):
        self._ensure_components()
        return self._seconds

    @property
    def days(self):
        self._ensure_components()
        return self._days

    def total_seconds(self):
        self._ensure_components()
        return self._total_seconds

    @property
    def components(self):
        from arkouda import DataFrame

        self._ensure_components()
        return DataFrame(
            {
                "days": self._d,
                "hours": self._h,
                "minutes": self._m,
                "seconds": self._s,
                "milliseconds": self._ms,
                "microseconds": self._us,
                "nanoseconds": self._ns,
            }
        )

    @classmethod
    def _get_callback(cls, otherclass, op):
        callbacks = {
            ("Timedelta", "-"): cls,  # Timedelta - Timedelta -> Timedelta
            ("Timedelta", "+"): cls,  # Timedelta + Timedelta -> Timedelta
            ("Datetime", "+"): Datetime,  # Timedelta + Datetime -> Datetime
            ("Datetime", "-"): Datetime,  # Datetime - Timedelta -> Datetime
            ("Timedelta", "%"): cls,  # Timedelta % Timedelta -> Timedelta
            ("pdarray", "//"): cls,  # Timedelta // pdarray -> Timedelta
            ("pdarray", "*"): cls,
        }  # Timedelta * pdarray -> Timedelta
        # Every other supported op returns an int64 pdarray, so callback is identity
        return callbacks.get((otherclass, op), _identity)

    def _scalar_callback(self, scalar):
        # Formats a returned scalar as a pandas.Timedelta
        return pdTimedelta(int(scalar), unit=_BASE_UNIT)

    @staticmethod
    def _is_supported_scalar(self, scalar):
        return self.is_timedelta_scalar(scalar)

    def to_pandas(self):
        """Convert array to a pandas TimedeltaIndex. Note: if the array size
        exceeds client.maxTransferBytes, a RuntimeError is raised.

        See Also
        --------
        to_ndarray
        """
        return to_timedelta(self.to_ndarray())

    def std(
        self,
        ddof: int_scalars = 0,
        axis: Optional[Union[None, int, tuple]] = None,
        keepdims: Optional[bool] = False,
    ):
        """Returns the standard deviation as a pd.Timedelta object, with args compatible with ak.std."""
        return self._scalar_callback(self.values.std(ddof, axis, keepdims))

    def sum(self):
        # Sum as a pd.Timedelta
        return self._scalar_callback(self.values.sum())

    def abs(self):
        """Absolute value of time interval."""
        from arkouda.numpy import abs as akabs
        from arkouda.numpy.numeric import cast as akcast

        return self.__class__(akcast(akabs(self.values), "int64"))

    def register(self, user_defined_name):
        """
        Register this Timedelta object and underlying components with the Arkouda server.

        Parameters
        ----------
        user_defined_name : str
            user defined name the timedelta is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        Timedelta
            The same Timedelta which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different Timedeltas with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the timedelta with the user_defined_name

        See Also
        --------
        unregister, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        """
        from arkouda.client import generic_msg

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")
        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.special_objType,
                "array": self.values,
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this timedelta object in the arkouda server which was previously
        registered using register() and/or attached to using attach().

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See Also
        --------
        register, attach, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        """
        from arkouda.numpy.util import unregister

        if not self.registered_name:
            raise RegistrationError("This object is not registered")
        unregister(self.registered_name)
        self.registered_name = None

    def is_registered(self) -> np.bool_:
        """
         Return True iff the object is contained in the registry or is a component of a
         registered object.

        Returns
        -------
        numpy.bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mis-match of registered components

        See Also
        --------
        register, attach, unregister

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda.numpy.util import is_registered

        if self.registered_name is None:
            return np.bool_(is_registered(self.values.name, as_component=True))
        else:
            return np.bool_(is_registered(self.registered_name))


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize=False,
    name=None,
    inclusive="both",
    **kwargs,
):
    """
    Create a fixed frequency Datetime range. Alias for
    ``ak.Datetime(pd.date_range(args))``. Subject to size limit
    imposed by client.maxTransferBytes.

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5H'. See
        timeseries.offset_aliases for a list of
        frequency aliases.
    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex, for example
        'Asia/Hong_Kong'. By default, the resulting DatetimeIndex is
        timezone-naive.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries. Whether to set each bound as closed or open.
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    rng : DatetimeIndex

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``DatetimeIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    """
    return Datetime(
        pd_date_range(
            start,
            end,
            periods,
            freq,
            tz,
            normalize,
            name,
            inclusive=inclusive,
            **kwargs,
        )
    )


def timedelta_range(start=None, end=None, periods=None, freq=None, name=None, closed=None, **kwargs):
    """
    Return a fixed frequency TimedeltaIndex, with day as the default
    frequency. Alias for ``ak.Timedelta(pd.timedelta_range(args))``.
    Subject to size limit imposed by client.maxTransferBytes.

    Parameters
    ----------
    start : str or timedelta-like, default None
        Left bound for generating timedeltas.
    end : str or timedelta-like, default None
        Right bound for generating timedeltas.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5H'.
    name : str, default None
        Name of the resulting TimedeltaIndex.
    closed : str, default None
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None).

    Returns
    -------
    rng : TimedeltaIndex

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``TimedeltaIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.
    """
    return Timedelta(pd_timedelta_range(start, end, periods, freq, name, closed, **kwargs))
