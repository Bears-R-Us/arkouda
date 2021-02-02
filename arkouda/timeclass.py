from arkouda.pdarrayclass import pdarray
from pandas import Series, Timestamp, Timedelta as pdTimedelta, date_range as pd_date_range, timedelta_range as pd_timedelta_range, to_datetime, to_timedelta # type: ignore
from arkouda.dtypes import int64, isSupportedInt
from arkouda.pdarraycreation import from_series, array as ak_array
from arkouda.numeric import cast, abs as akabs
import numpy as np # type: ignore
import datetime

_BASE_UNIT = 'ns'

_unit2factor = {'d': 24*60*60*10**9,
               'h': 60*60*10**9,
               'm': 60*10**9,
               's': 10**9,
               'ms': 10**6,
               'us': 10**3,
               'ns': 1}

def _get_factor(unit : str) -> int:
    unit = unit.lower()
    if unit not in _unit2factor:
        raise ValueError("Argument must be one of {}".format(set(_unit2factor.keys())))
    return _unit2factor[unit]

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
        self._data = self._factor * scalar.astype('int64')
    

class _AbstractBaseTime(pdarray):
    
    def __init__(self, array, unit : str=_BASE_UNIT): # type: ignore
        # Convert the input to int64 pdarray of nanoseconds
        if isinstance(array, pdarray):
            if array.dtype != int64:
                raise TypeError("{} array must have int64 dtype".format(self.__class__.__name__))
            # Already int64 pdarray, just scale
            self.unit = unit
            self._factor = _get_factor(self.unit)
            # This makes a copy of the input array, to leave input unchanged
            self._data = self._factor * array # Mimics a datetime64[ns] array         
        elif hasattr(array, 'dtype'):
            # Handles all pandas and numpy datetime/timedelta arrays
            if array.dtype.kind not in ('M', 'm'):
                # M = datetime64, m = timedelta64
                raise TypeError("Invalid dtype: {}".format(array.dtype.name))
            if isinstance(array, Series):
                # Pandas Datetime and Timedelta
                # Get units of underlying numpy datetime64 array
                self.unit = np.datetime_data(array.values.dtype)[0]
                self._factor = _get_factor(self.unit)
                # Create pdarray
                self._data = from_series(array)
                # Scale if necessary
                # This is futureproofing; it will not be used unless pandas
                # changes its Datetime implementation
                if self._factor != 1:
                    # Scale inplace because we already created a copy
                    self._data *= self._factor
            elif isinstance(array, np.ndarray):
                # Numpy datetime64 and timedelta64
                # Force through pandas.Series
                self.__init__(to_datetime(array).to_series()) # type: ignore
            elif hasattr(array, 'to_series'):
                # Pandas DatetimeIndex
                # Force through pandas.Series
                self.__init__(array.to_series()) # type: ignore
        else:
            raise TypeError("Unsupported type: {}".format(type(array)))
        # Now that self._data is correct, init self with same metadata
        super().__init__(self._data.name, self._data.dtype.name, self._data.size, self._data.ndim, self._data.shape, self._data.itemsize)

    @classmethod
    def _get_callback(cls, other, op):
        # Will be overridden by all children
        return _identity
    
    def _validate_unit(unit):
        if unit.lower() not in _unit2factor:
            raise ValueError("Argument must be one of {}".format(set(_unit2factor.keys())))
        
    def floor(self, freq):
        f = _get_factor(freq)
        return self.__class__(self._data // f, unit=freq)

    def ceil(self, freq):
        f = _get_factor(freq)
        return self.__class__((self._data + (f - 1)) // f, unit=freq)

    def round(self, freq):
        f = _get_factor(freq)
        offset = self._data + ((f + 1) // 2)
        rounded = offset // f
        # Halfway values are supposed to round to the nearest even integer
        # Need to figure out which ones ended up odd and fix them
        decrement = ((offset % f) == 0) & ((rounded % 2) == 1)
        rounded[decrement] = rounded[decrement] - 1
        return self.__class__(rounded, unit=freq)

    def to_ndarray(self):
        return np.array(self._data.to_ndarray(), dtype="{}64[ns]".format(self.__class__.__name__.lower()))

    def __str__(self):
        from arkouda.client import pdarrayIterThresh
        if self.size <= pdarrayIterThresh:
            vals = ["'{}'".format(self[i]) for i in range(self.size)]
        else:
            vals = ["'{}'".format(self[i]) for i in range(3)]
            vals.append('... ')
            vals.extend(["'{}'".format(self[i]) for i in range(self.size-3, self.size)])
        spaces = ' '*(len(self.__class__.__name__)+1)
        return "{}([{}],\n{}dtype='{}64[ns]')".format(self.__class__.__name__,
                                                      ',\n{} '.format(spaces).join(vals),
                                                      spaces,
                                                      self.__class__.__name__.lower())
    
    def __repr__(self) -> str:
        return self.__str__()

    def _binop(self, other, op):
        if isinstance(other, Datetime) or self._is_datetime_scalar(other):
            if op not in self.supported_with_datetime:
                raise TypeError("{} not supported between {} and Datetime".format(op, self.__class__.__name__))
            otherclass = 'Datetime'
            if self._is_datetime_scalar(other):
                otherdata = _Timescalar(other)._data
            else:
                otherdata = other._data
        elif isinstance(other, Timedelta) or self._is_timedelta_scalar(other):
            if op not in self.supported_with_timedelta:
                raise TypeError("{} not supported between {} and Timedelta".format(op, self.__class__.__name__))
            otherclass = 'Timedelta'
            if self._is_timedelta_scalar(other):
                otherdata = _Timescalar(other)._data
            else:
                otherdata = other._data
        elif (isinstance(other, pdarray) and other.dtype == int64) or isSupportedInt(other):
            if op not in self.supported_with_pdarray:
                raise TypeError("{} not supported between {} and integer".format(op, self.__class__.__name__))
            otherclass = 'pdarray'
            otherdata = other
        else:
            return NotImplemented
        callback = self._get_callback(otherclass, op)
        return callback(self._data._binop(otherdata, op))

    def _r_binop(self, other, op):
        # First case is pdarray <op> self
        if (isinstance(other, pdarray) and other.dtype == int64):
            if op not in self.supported_with_r_pdarray:
                raise TypeError("{} not supported between int64 and {}".format(op, self.__class__.__name__))
            callback = self._get_callback('pdarray', op)
            # Need to use other._binop because self._data._r_binop can only handle scalars
            return callback(other._binop(self._data, op))
        # All other cases are scalars, so can use self._data._r_binop
        elif self._is_datetime_scalar(other):
            if op not in self.supported_with_r_datetime:
                raise TypeError("{} not supported between scalar datetime and {}".format(op, self.__class__.__name__))
            otherclass = 'Datetime'
            otherdata = _Timescalar(other)._data
        elif self._is_timedelta_scalar(other):
            if op not in self.supported_with_r_timedelta:
                raise TypeError("{} not supported between scalar timedelta and {}".format(op, self.__class__.__name__))
            otherclass = 'Timedelta'
            otherdata = _Timescalar(other)._data
        elif isSupportedInt(other):
            if op not in self.supported_with_r_pdarray:
                raise TypeError("{} not supported between int64 and {}".format(op, self.__class__.__name__))
            otherclass = 'pdarray'
            otherdata = other
        else:
            # If here, type is not handled
            return NotImplemented
        callback = self._get_callback(otherclass, op)
        return callback(self._data._r_binop(otherdata, op))

    def opeq(self, other, op):
        if isinstance(other, Timedelta) or self._is_timedelta_scalar(other):
            if op not in self.supported_opeq:
                raise TypeError("{} {} Timedelta not supported".format(self.__class__.__name__, op))
            if self._is_timedelta_scalar(other):
                other = _Timescalar(other)
            self._data.opeq(other._data, op)
        elif isinstance(other, Datetime) or self._is_datetime_scalar(other):
            raise TypeError("{} {} datetime not supported".format(self.__class__.__name__, op))
        else:
            return NotImplemented    

    @staticmethod
    def _is_datetime_scalar(scalar):
        return (isinstance(scalar, Timestamp) or
                (isinstance(scalar, np.datetime64) and np.isscalar(scalar)) or
                isinstance(scalar, datetime.datetime))

    @staticmethod
    def _is_timedelta_scalar(scalar):
        return (isinstance(scalar, pdTimedelta) or
                (isinstance(scalar, np.timedelta64) and np.isscalar(scalar)) or
                isinstance(scalar, datetime.timedelta))

    # def _normalize_scalar(self, scalar):
    #     scalar = _Timescalar(scalar)
    #     if scalar._factor < self._factor:
    #         warnings.warn("Discarding precision in assignment")
    #     return (scalar._data * scalar._factor) // self._factor

    def _scalar_callback(self, key):
        # Will be overridden in all children
        return key
        
    def __getitem__(self, key):
        if isSupportedInt(key):
            return self._scalar_callback(self._data[key])
        else:
            # Slice or array index should return same class
            return self.__class__(self._data[key])

    def __setitem__(self, key, value):
        if isinstance(value, self.__class__):
            if self._factor != other._factor:
                if other._factor < self._factor:
                    warnings.warn("Discarding precision in assignment")
                self._data[key] = (value._factor * value._data) // self._factor
            else:
                self._data[key] = value._data
        elif self._is_supported_scalar(value):
            normval = self._Timescalar(value)
            self._data[key] = normval._data
        else:
            return NotImplemented

    def min(self):
        return self._scalar_callback(self._data.min())

    def max(self):
        return self._scalar_callback(self._data.max())

    def mink(self, k):
        return self.__class__(self._data.mink(k))

    def maxk(self, k):
        return self.__class__(self._data.maxk(k))

class Datetime(_AbstractBaseTime):

    supported_with_datetime = frozenset(('==', '!=', '<', '<=', '>', '>=', '-'))
    supported_with_r_datetime = frozenset(('==', '!=', '<', '<=', '>', '>=', '-'))
    supported_with_timedelta = frozenset(('+', '-', '/', '//', '%'))
    supported_with_r_timedelta = frozenset(('+'))
    supported_opeq = frozenset(('+=', '-='))
    supported_with_pdarray = frozenset(())  # type: ignore
    supported_with_r_pdarray = frozenset(()) # type: ignore

    @classmethod
    def _get_callback(cls, otherclass, op): 
        callbacks = {('Datetime', '-'): Timedelta,
                     ('Timedelta', '+'): cls,
                     ('Timedelta', '-'): cls,
                     ('Timedelta', '%'): Timedelta}
        return callbacks.get((otherclass, op), _identity)
    
    def _scalar_callback(self, scalar):
        # Formats a scalar return value
        return Timestamp(int(scalar), unit=_BASE_UNIT)

    @staticmethod
    def _is_supported_scalar(scalar):
        return self.is_datetime_scalar(scalar)

    def to_pandas(self):
        return to_datetime(self.to_ndarray())

    def sum(self):
        raise TypeError("Cannot sum datetime64 values")
                
class Timedelta(_AbstractBaseTime):

    supported_with_datetime = frozenset(('+'))
    supported_with_r_datetime = frozenset(('+', '-', '/', '//', '%'))
    supported_with_timedelta = frozenset(('==', '!=', '<', '<=', '>', '>=', '+', '-', '/', '//', '%'))
    supported_with_r_timedelta = frozenset(('==', '!=', '<', '<=', '>', '>=', '+', '-', '/', '//', '%'))
    supported_opeq = frozenset(('+=', '-=', '%='))
    supported_with_pdarray = frozenset(('*', '//'))
    supported_with_r_pdarray = frozenset(('*'))

    @classmethod
    def _get_callback(cls, otherclass, op):
        callbacks = {('Timedelta', '-'): cls,
                     ('Timedelta', '+'): cls,
                     ('Datetime', '+'): Datetime,
                     ('Datetime', '-'): Datetime,
                     ('Timedelta', '%'): cls,
                     ('pdarray', '//'): cls,   # Td // pdarray -> Td
                     ('pdarray', '*'): cls}
        return callbacks.get((otherclass, op), _identity)
    
    def _scalar_callback(self, scalar):
        return pdTimedelta(int(scalar), unit=_BASE_UNIT)

    @staticmethod
    def _is_supported_scalar(scalar):
        return self.is_timedelta_scalar(scalar)

    def to_pandas(self):
        return to_timedelta(self.to_ndarray())

    def sum(self):
        return self._scalar_callback(self._data.sum())

    def abs(self):
        return self.__class__(cast(akabs(self._data), 'int64'))
    
    
def date_range(start=None, end=None, periods=None, freq=None,
               tz=None, normalize=False, name=None, closed=None, **kwargs):
    return Datetime(pd_date_range(start, end, periods, freq,
                                  tz, normalize, name, closed, **kwargs))

def timedelta_range(start=None, end=None, periods=None, freq=None,
                    name=None, closed=None, **kwargs):
    return Timedelta(pd_timedelta_range(start, end, periods, freq,
                                  name, closed, **kwargs))
