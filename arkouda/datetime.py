from arkouda.pdarrayclass import pdarray
from pandas import Timestamp

unit2factor = {'d': 24*60*60*10**9,
               'h': 60*60*10**9,
               'm': 60*10**9,
               's': 10**9,
               'ms': 10**6,
               'us': 10**3,
               'ns': 1}


class Datetime(pdarray):
    supported_with_datetime = frozenset(('==', '!=', '<', '<=', '>', '>='))
    supported_with_timedelta = frozenset(('+', '-', '/', '//', '%'))
    
    def __init__(self, array, unit='ns'):
        if array.dtype != int64:
            raise TypeError("Datetime array must have int64 dtype")
        if array.min() < 0:
            raise ValueError("Datetime values cannot be negative")
        if unit not in unit2factor:
            raise ValueError("Datetime unit must be one of {}".format(set(unit2factor.keys())))
        super().__init__(self, array.name, array.dtype.name, array.size, array.ndim, array.shape, array.itemsize)
        self._data = array
        self.unit = unit
        self._factor = unit2factor[self.unit]

    def _validate_unit(unit):
        if unit not in unit2factor:
            raise ValueError("Argument must be one of {}".format(set(unit2factor.keys())))
        
    def floor(self, freq):
        _validate_unit(freq)
        freqfactor = unit2factor[freq]
        f = freqfactor // self._factor
        return Datetime(f*(self._data // f), unit=self.unit)

    def ceil(self, freq):
        _validate_unit(freq)
        freqfactor = unit2factor[freq]
        f = freqfactor // self._factor
        return Datetime(f*((self._data + (f - 1)) // f), unit=self.unit)

    def round(self, freq):
        _validate_unit(freq)
        freqfactor = unit2factor[freq]
        f = freqfactor // self._factor
        offset = (f+1)//2
        return Datetime(f*((self._data + offset) // f), unit=self.unit)

    def to_ndarray(self):
        return np.array(self.to_ndarray(), dtype="datetime64[{}]".format(self.unit))

    def __str__(self):
        from arkouda.client import pdarrayIterThresh
        if self.size <= pdarrayIterThresh:
            vals = ["'{}'".format(self[i]) for i in range(self.size)]
        else:
            vals = ["'{}'".format(self[i]) for i in range(3)]
            vals.append('... ')
            vals.extend(["'{}'".format(self[i]) for i in range(self.size-3, self.size)])
        return "[{}]".format(', '.join(vals))
    
    def __repr__(self) -> str:
        return "array({}, dtype='datetime64[{}]')".format(self.__str__(), self.unit)

    def _binop(self, other, op):
        def adjust_and_compute():
            if self._factor > other._factor:
                adjustment = self._factor // other._factor
                return Datetime(super()._binop(adjustment*self, other, op), unit=other.unit)
            elif self._factor < other._factor:
                adjustment = other._factor // self._factor
                return Datetime(super()._binop(self, adjustment*other, op), unit=self.unit)
            else:
                return Datetime(super()._binop(self, other, op), unit=self.unit)
        if isinstance(other, Datetime):
            if op not in self.supported_with_datetime:
                raise TypeError("{} not supported between two Datetime objects".format(op))
            adjust_and_compute()
        
        elif isinstance(other, Timedelta):
            if op not in self.supported_with_timedelta:
                raise TypeError("{} not supported between Datetime and Timedelta".format(op))
            adjust_and_compute()
        else:
            # super()._binop(self, other, op)
            return NotImplemented

    def opeq(self, other, op):
        def adjust_and_compute():
            if self._factor != other._factor:
                super().opeq(self, (other._factor * other) // self._factor, op)
                return self
            else:
                super().opeq(self, other, op)
                return self
        if isinstance(other, Timedelta):
            if op not in self.supported_with_timedelta:
                raise TypeError("{} not supported between Datetime and Timedelta")
            adjust_and_compute()
        else:
            return NotImplemented    

    def __getitem__(self, key):
        if isSupportedInt(key):
            return Timestamp(super().__getitem__(self, key), unit=self.unit).to_datetime64()
        else:
            return Datetime(super().__getitem__(self, key), unit=self.unit)

    def __setitem__(self, key, value):
        if isinstance(value, Datetime):
            if self._factor != other._factor:
                super().__setitem__(self, key, (value._factor * value) // self._factor)
            else:
                super()._binop(self, key, value)
        elif isSupportedDatetime(value):
            value = normalize_datetime_scalar(value)
            super().__setitem__(self, key, value.value // self._factor)
        else:
            return NotImplemented
                
