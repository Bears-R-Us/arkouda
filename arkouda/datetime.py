from arkouda.pdarrayclass import pdarray

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
        super().__init__(self, array.name, array.dtype.name, array.size, array.ndim, array.shape, array.itemsize)      
        self.unit = unit
        self._factor = unit2factor[self.unit]

    def floor(self, freq):
        freqfactor = unit2factor[freq]
        f = freqfactor // self._factor
        return self // f

    def _binop(self, other, op):
        def adjust_and_compute():
            if self._factor > other._factor:
                adjustment = self._factor // other._factor
                return super()._binop(self, adjustment*other, op)
            elif self._factor < other._factor:
                adjustment = other._factor // self._factor
                return super()._binop(adjustment*self, other, op)
            else:
                return super()._binop(self, other, op)
        if isinstance(other, Datetime):
            if op not in self.supported_with_datetime:
                raise TypeError("{} not supported between two Datetime objects".format(op))
            adjust_and_compute()
        
        elif isinstance(other, Timedelta):
            if op not in self.supported_with_timedelta:
                raise TypeError("{} not supported between Datetime and Timedelta".format(op))
            adjust_and_compute()
        else:
            super()._binop(self, other, op)

    def __getitem__(self, key):
        if isinstace(key, int):
            return pd.Timestamp(super().__getitem__(self, key), unit=self.unit)
        else:
            return super().__getitem__(self, key)
