from __future__ import annotations

import json
import operator
from builtins import str as builtin_str
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import pandas as pd
from pandas._config import get_option
from typeguard import typechecked

import arkouda.pandas.dataframe
from arkouda.accessor import CachedAccessor, DatetimeAccessor, StringAccessor
from arkouda.alignment import lookup
from arkouda.numpy.dtypes import bool_scalars, dtype, float64, int64
from arkouda.numpy.pdarrayclass import (
    RegistrationError,
    any,
    argmaxk,
    create_pdarray,
    pdarray,
)
from arkouda.numpy.pdarraycreation import arange, array, full, zeros
from arkouda.numpy.pdarraysetops import argsort, concatenate, in1d, indexof1d
from arkouda.numpy.strings import Strings
from arkouda.numpy.util import get_callback, is_float
from arkouda.pandas.categorical import Categorical
from arkouda.pandas.groupbyclass import GroupBy, groupable_element_type
from arkouda.pandas.index import Index, MultiIndex

if TYPE_CHECKING:
    from arkouda.numpy import cast as akcast
    from arkouda.numpy import isnan, value_counts
    from arkouda.numpy.segarray import SegArray
else:
    SegArray = TypeVar("SegArray")
    akcast = TypeVar("akcast")
    isnan = TypeVar("isnan")
    value_counts = TypeVar("value_counts")

# pd.set_option("display.max_colwidth", 65) is being called in DataFrame.py. This will resolve BitVector
# truncation issues. If issues arise, that's where to look for it.

__all__ = [
    "Series",
]


supported_scalars = Union[int, float, bool, builtin_str, np.int64, np.float64, np.bool_, np.str_]


def is_supported_scalar(x) -> bool:
    return isinstance(x, (int, float, bool, builtin_str, np.int64, np.float64, np.bool_, np.str_))


def natural_binary_operators(cls) -> type:
    for name, op in {
        "__add__": operator.add,
        "__sub__": operator.sub,
        "__mul__": operator.mul,
        "__truediv__": operator.truediv,
        "__floordiv__": operator.floordiv,
        "__and__": operator.and_,
        "__or__": operator.or_,
        "__xor__": operator.xor,
        "__eq__": operator.eq,
        "__ge__": operator.ge,
        "__gt__": operator.gt,
        "__le__": operator.le,
        "__lshift__": operator.lshift,
        "__lt__": operator.lt,
        "__mod__": operator.mod,
        "__ne__": operator.ne,
        "__rshift__": operator.rshift,
        "__pow__": operator.pow,
    }.items():
        setattr(cls, name, cls._make_binop(op))

    return cls


def unary_operators(cls) -> type:
    for name, op in {
        "__invert__": operator.invert,
        "__neg__": operator.neg,
    }.items():
        setattr(cls, name, cls._make_unaryop(op))

    return cls


def aggregation_operators(cls) -> type:
    for name in ["max", "min", "mean", "sum", "std", "var", "argmax", "argmin", "prod"]:
        setattr(cls, name, cls._make_aggop(name))
    return cls


@unary_operators
@aggregation_operators
@natural_binary_operators
class Series:
    """
    One-dimensional arkouda array with axis labels.

    Parameters
    ----------
    index : pdarray, Strings
        an array of indices associated with the data array.
        If empty, it will default to a range of ints whose size match the size of the data.
        optional
    data : Tuple, List, groupable_element_type, Series, SegArray
        a 1D array. Must not be None.

    Raises
    ------
    TypeError
        Raised if index is not a pdarray or Strings object
        Raised if data is not a pdarray, Strings, or Categorical object
    ValueError
        Raised if the index size does not match data size

    Notes
    -----
    The Series class accepts either positional arguments or keyword arguments.
    If entering positional arguments,
        2 arguments entered:
            argument 1 - data
            argument 2 - index
        1 argument entered:
            argument 1 - data
    If entering 1 positional argument, it is assumed that this is the data argument.
    If only 'data' argument is passed in, Index will automatically be generated.
    If entering keywords,
        'data' (see Parameters)
        'index' (optional) must match size of 'data'

    """

    objType = "Series"

    @typechecked
    def __init__(
        self,
        data: Union[
            Tuple,
            List,
            groupable_element_type,
            Series,
            SegArray,
            pd.Series,
            pd.Categorical,
        ],
        name=None,
        index: Optional[Union[pdarray, Strings, Tuple, List, Index]] = None,
    ):
        if isinstance(data, pd.Categorical):
            data = Categorical(data)

        self.registered_name: Optional[str] = None

        if index is None and isinstance(data, (tuple, list)) and len(data) == 2:
            # handles the previous `ar_tuple` case
            if not isinstance(data[0], (pdarray, Index, Strings, Categorical, list, tuple)):
                raise TypeError("indices must be a pdarray, Strings, Categorical, List, or Tuple")
            if not isinstance(data[1], (pdarray, Strings, Categorical, Series, SegArray)):
                raise TypeError("values must be a pdarray, Strings, SegArray, or Categorical")
            self.values = data[1] if not isinstance(data[1], Series) else data[1].values
            self.index = Index.factory(index) if index else Index.factory(data[0])
        elif isinstance(data, pd.Series):
            if isinstance(data.values, pd.Categorical):
                self.values = Categorical(data.values)
            else:
                self.values = array(data.values)
            self.index = Index(data.index)
            self.name = data.name
        elif isinstance(data, tuple) and len(data) != 2:
            raise TypeError("Series initialization requries a tuple of (index, values)")
        else:
            # When only 1 positional argument it will be treated as data and not index
            if isinstance(data, Series):
                self.values = data.values
            elif isinstance(data, List):
                self.values = array(data)
            else:
                self.values = data
            self.index = Index.factory(index) if index is not None else Index(arange(self.values.size))

        if self.index.size != self.values.size:
            raise ValueError(
                "Index size does not match data size: {} != {}".format(self.index.size, self.values.size)
            )
        if name is None and isinstance(data, (Series, pd.Series)):
            self.name = data.name
        else:
            self.name = name
        self.size = self.index.size

    def __len__(self):
        return self.values.size

    def __repr__(self):
        """Return ascii-formatted version of the series."""
        if len(self) == 0:
            return "Series([ -- ][ 0 values : 0 B])"

        maxrows = pd.get_option("display.max_rows")
        if len(self) <= maxrows:
            prt = self.to_pandas()
            length_str = ""
        else:
            prt = pd.concat(
                [
                    self.head(maxrows // 2 + 2).to_pandas(),
                    self.tail(maxrows // 2).to_pandas(),
                ]
            )
            length_str = f"\nLength {len(self)}"
        return (
            prt.to_string(
                dtype=prt.dtype,
                min_rows=get_option("display.min_rows"),
                max_rows=maxrows,
                length=False,
            )
            + length_str
        )

    def validate_key(
        self,
        key: Union[Series, pdarray, Strings, Categorical, List, supported_scalars, SegArray],
    ) -> Union[pdarray, Strings, Categorical, supported_scalars, SegArray]:
        """
        Validate type requirements for keys when reading or writing the Series.
        Also converts list and tuple arguments into pdarrays.

        Parameters
        ----------
        key : Series, pdarray, Strings, Categorical, List, supported_scalars, or SegArray
            The key or container of keys that might be used to index into the Series.

        Returns
        -------
        The validated key(s), with lists and tuples converted to pdarrays

        Raises
        ------
        TypeError
            Raised if keys are not boolean values or the type of the labels
            Raised if key is not one of the supported types
        KeyError
            Raised if container of keys has keys not present in the Series
        IndexError
            Raised if the length of a boolean key array is different
            from the Series

        """
        if isinstance(key, list):
            return self.validate_key(array(key))
        if isinstance(key, tuple):
            raise TypeError("Series does not support tuple keys")
        if isinstance(key, Series):
            # @TODO align the series indexes
            return self.validate_key(key.values)

        if is_supported_scalar(key):
            if dtype(type(key)) != self.index.dtype:
                raise TypeError(
                    "Unexpected key type. Received {} but expected {}. key: {}".format(
                        dtype(type(key)), self.index.dtype, key
                    )
                )
        elif isinstance(key, Strings):
            if self.index.dtype != dtype(str):
                raise TypeError(
                    "Unexpected key type. Received Strings but expected {}".format(self.index.dtype)
                )
            if any(~in1d(key, self.index.values)):
                raise KeyError("{} not in index".format(key[~in1d(key, self.index.values)]))
        elif isinstance(key, pdarray):
            if key.dtype == self.index.dtype:
                if any(~in1d(key, self.index.values)):
                    raise KeyError("{} not in index".format(key[~in1d(key, self.index.values)]))
            elif key.dtype == "bool_":
                if key.size != self.index.size:
                    raise IndexError(
                        "Boolean index has wrong length: {} instead of {}".format(key.size, self.size)
                    )
            else:
                raise TypeError(
                    "Unexpected key type. Received {} but expected {}".format(
                        dtype(type(key)), self.index.dtype
                    )
                )
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.size
            stride = key.step if key.step is not None else 1
            if start < 0:
                raise IndexError("Slice start must be non-negative")
            if stop > self.size:
                raise IndexError("Slice stop must be less than or equal to the size of the Series")
            if start > stop:
                raise IndexError("Slice start must be less than or equal to the stop")
            key = arange(start, stop, stride)
        else:
            raise TypeError(
                "Series [] only supports indexing by scalars, lists of scalars, "
                "and arrays of scalars. Received {}".format(type(key))
            )
        return key

    @typechecked
    def __getitem__(self, _key: Union[supported_scalars, pdarray, Strings, List, Series]):
        """
        Get values from Series.

        Parameters
        ----------
        _key : supported_scalars, pdarray, Strings, List, or Series
            The key or container of keys to get entries for.

        Returns
        -------
        Series with all entries with matching labels. If only one entry in the
        Series is accessed, returns a scalar.

        """
        key = self.validate_key(_key)
        if is_supported_scalar(key):
            return self[array([key])]
        assert isinstance(key, (pdarray, Strings))
        if isinstance(key, pdarray) and key.dtype == "bool_":
            # boolean array indexes without sorting
            return Series(index=self.index[key], data=self.values[key])
        indices = indexof1d(key, self.index.values)
        if len(indices) == 1:
            return self.values[indices[0]]
        else:
            return Series(index=self.index[indices], data=self.values[indices])

    def validate_val(
        self, val: Union[pdarray, Strings, supported_scalars, List]
    ) -> Union[pdarray, Strings, supported_scalars]:
        """
        Validate type requirements for values being written into the Series.
        Also converts list and tuple arguments into pdarrays.

        Parameters
        ----------
        val : pdarray, Strings, supported_scalars, or List
            The value or container of values that might be assigned into the Series.

        Returns
        -------
        The validated value, with lists converted to pdarrays

        Raises
        ------
        TypeError
            Raised if val is not the same type or a container with elements
              of the same time as the Series
            Raised if val is a string or Strings type.
            Raised if val is not one of the supported types

        """
        if isinstance(val, list):
            val = array(val)
        if is_supported_scalar(val):
            if dtype(type(val)) != self.values.dtype:
                raise TypeError(
                    "Unexpected value type. Received {} but expected {}".format(
                        dtype(type(val)), self.values.dtype
                    )
                )
            if isinstance(val, str):
                raise TypeError("Cannot modify string type dataframes")
        elif isinstance(val, Strings):
            raise TypeError("Cannot modify string type dataframes")
        elif isinstance(val, pdarray):
            if val.dtype != self.values.dtype:
                raise TypeError(
                    "Unexpected value type. Received {} but expected {}".format(
                        dtype(type(val)), self.values.dtype
                    )
                )
        else:
            raise TypeError("cannot set with unsupported value type: {}".format(type(val)))
        return val

    def __setitem__(
        self,
        key: Union[pdarray, Strings, Categorical, Series, List, supported_scalars, SegArray],
        val: Union[pdarray, Strings, List, supported_scalars],
    ) -> None:
        """
        Set or adds entries in a Series by label.

        Parameters
        ----------
        key : pdarray, Strings, Categorical, Series, List, supported_scalars, or SegArray
            The key or container of keys to set entries for.

        val : pdarray, Strings, List, or supported_scalars
            The value or values to set/add to the Series.

        Raises
        ------
        ValueError
            Raised when setting multiple values to a Series with repeated labels
            Raised when number of values provided does not match the number of
            entries to set.

        """
        val = self.validate_val(val)
        key = self.validate_key(key)

        if isinstance(key, (pdarray, Strings)) and len(key) > 1 and self.has_repeat_labels():
            raise ValueError("Cannot set with multiple keys for Series with repeated labels.")

        indices = None
        if is_supported_scalar(key):
            indices = self.index == key
        else:
            indices = in1d(self.index.values, key)
        tf, counts = GroupBy(indices).size()
        update_count = counts[1] if len(counts) == 2 else 0
        if update_count == 0:
            # adding a new entry
            if isinstance(val, (pdarray, Strings)):
                raise ValueError("Cannot set. Too many values provided")
            new_index_values = concatenate([self.index.values, array([key])])
            self.index = Index.factory(new_index_values)
            self.values = concatenate([self.values, array([val])])
            return
        if is_supported_scalar(val):
            self.values[indices] = val
            return
        else:
            val_array = cast(Union[pdarray, Strings], val)
            if val_array.size == 1 and is_supported_scalar(key):
                self.values[indices] = val_array[0]
                return
            if update_count != val_array.size:
                raise ValueError(
                    "Cannot set using a list-like indexer with a different length from the value"
                )
            self.values[indices] = val
            return

    def memory_usage(self, index: bool = True, unit: Literal["B", "KB", "MB", "GB"] = "B") -> int:
        """
        Return the memory usage of the Series.

        The memory usage can optionally include the contribution of
        the index.

        Parameters
        ----------
        index : bool, default=True
            Specifies whether to include the memory usage of the Series index.
        unit : {"B", "KB", "MB", "GB"}, default = "B"
            Unit to return. One of {'B', 'KB', 'MB', 'GB'}.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        arkouda.numpy.pdarrayclass.nbytes
        arkouda.Index.memory_usage
        arkouda.pandas.series.Series.memory_usage
        arkouda.pandas.datafame.DataFrame.memory_usage

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.series import Series
        >>> s = ak.Series(ak.arange(3))
        >>> s.memory_usage()
        48

        Not including the index gives the size of the rest of the data, which
        is necessarily smaller:

        >>> s.memory_usage(index=False)
        24

        Select the units:

        >>> s = ak.Series(ak.arange(3000))
        >>> s.memory_usage(unit="KB")
        46.875

        """
        from arkouda.numpy.util import convert_bytes

        v = cast(int, convert_bytes(self.values.nbytes, unit=unit))
        if index:
            v += self.index.memory_usage(unit=unit)
        return v

    def has_repeat_labels(self) -> bool:
        """Return whether the Series has any labels that appear more than once."""
        tf, counts = GroupBy(self.index.values).size()
        return counts.size != self.index.size

    def to_ndarray(self) -> np.ndarray:
        return self.values.to_ndarray()

    @property
    def ndim(self) -> int:
        return 1

    @property
    def loc(self) -> _LocIndexer:
        """
        Accesses entries of a Series by label.

        Returns
        -------
        _LocIndexer
            An indexer for label-based access to Series entries.

        """
        return _LocIndexer(self)

    @property
    def at(self) -> _LocIndexer:
        """
        Accesses entries of a Series by label.

        Returns
        -------
        _LocIndexer
            An indexer for label-based access to Series entries.

        """
        return _LocIndexer(self)

    @property
    def iloc(self) -> _iLocIndexer:
        """
        Accesses entries of a Series by position.

        Returns
        -------
        _iLocIndexer
            An indexer for position-based access to Series entries.

        """
        return _iLocIndexer("iloc", self)

    @property
    def iat(self) -> _iLocIndexer:
        """
        Accesses entries of a Series by position.

        Returns
        -------
        _iLocIndexer
            An indexer for position-based access to a single element.

        """
        return _iLocIndexer("iat", self)

    dt = CachedAccessor("dt", DatetimeAccessor)
    str = CachedAccessor("str", StringAccessor)

    @property
    def shape(self) -> Tuple[int]:
        # mimic the pandas return of series shape property
        return (self.values.size,)

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    @typechecked
    def isin(self, lst: Union[pdarray, Strings, List]) -> Series:
        """
        Find Series elements whose values are in the specified list.

        Parameters
        ----------
        lst : pdarray, Strings, or List
            Either a Python list or an Arkouda array to check membership against.

        Returns
        -------
        Series
            A Series of booleans that is True for elements found in the list,
            and False otherwise.

        """
        if isinstance(lst, list):
            lst = array(lst)

        boolean = in1d(self.values, lst)
        return Series(data=boolean, index=self.index)

    @typechecked
    def locate(self, key: Union[int, pdarray, Index, Series, List, Tuple]) -> Series:
        """
        Lookup values by index label.

        Parameters
        ----------
        key : int, pdarray, Index, Series, List, or Tuple
            The key or keys to look up. This can be:
            - A scalar
            - A list of scalars
            - A list of lists (for MultiIndex)
            - A Series (in which case labels are preserved, and its values are used as keys)

            Keys will be converted to Arkouda arrays as needed.

        Returns
        -------
        Series
            A Series containing the values corresponding to the key.

        """
        if isinstance(key, Series):
            # special case, keep the index values of the Series, and lookup the values
            return Series(index=key.index, data=lookup(self.index.index, self.values, key.values))
        elif isinstance(key, MultiIndex):
            idx = self.index.lookup(key.index)
        elif isinstance(key, Index):
            idx = self.index.lookup(key.index)
        elif isinstance(key, pdarray):
            idx = self.index.lookup(key)
        elif isinstance(key, (list, tuple)):
            key0 = key[0]
            if isinstance(key0, list) or isinstance(key0, tuple):
                # nested list. check if already arkouda arrays
                if not isinstance(key0[0], pdarray):
                    # convert list of lists to list of pdarrays
                    key = [array(a) for a in np.array(key).T.copy()]

            elif not isinstance(key0, pdarray):
                # a list of scalers, convert into arkouda array
                try:
                    val = array(key)
                    if isinstance(val, pdarray):
                        key = val
                except Exception:
                    raise TypeError("'key' parameter must be convertible to pdarray")

            # else already list if arkouda array, use as is
            idx = self.index.lookup(key)
        else:
            # scalar value
            idx = self.index == key
        return Series(index=self.index[idx], data=self.values[idx])

    @classmethod
    def _make_binop(cls, operator):
        def binop(self, other) -> Series:
            if isinstance(other, Series):
                if self.index._check_aligned(other.index):
                    return cls((self.index, operator(self.values, other.values)))
                else:
                    idx = self.index._merge(other.index).index
                    a = lookup(self.index.index, self.values, idx, fillvalue=0)
                    b = lookup(other.index.index, other.values, idx, fillvalue=0)
                    return cls((idx, operator(a, b)))
            else:
                return cls((self.index, operator(self.values, other)))

        return binop

    @classmethod
    def _make_unaryop(cls, operator):
        def unaryop(self) -> Series:
            return cls((self.index, operator(self.values)))

        return unaryop

    @classmethod
    def _make_aggop(cls, name):
        def aggop(self) -> Series:
            return getattr(self.values, name)()

        return aggop

    @typechecked
    def add(self, b: Series) -> Series:
        index = self.index.concat(b.index).index

        values = concatenate([self.values, b.values], ordered=False)

        idx, vals = GroupBy(index).sum(values)
        return Series(data=vals, index=idx)

    @typechecked
    def topn(self, n: int = 10) -> Series:
        """
        Return the top values of the Series.

        Parameters
        ----------
        n : int, default=10
            Number of values to return. The default of 10 returns the top 10 values.

        Returns
        -------
        Series
            A new Series containing the top `n` values.

        """
        k = self.index
        v = self.values

        idx = argmaxk(v, n)
        idx = idx[-1 : -n - 1 : -1]

        return Series(index=k.index[idx], data=v[idx])

    def _reindex(self, idx):
        if isinstance(self.index, MultiIndex):
            new_index = MultiIndex(self.index[idx].levels, name=self.index.name, names=self.index.names)
        elif isinstance(self.index, Index):
            new_index = Index(self.index[idx], name=self.index.name)
        else:
            new_index = Index(self.index[idx])

        return Series(index=new_index, data=self.values[idx])

    @typechecked
    def sort_index(self, ascending: bool = True) -> Series:
        """
        Sort the Series by its index.

        Parameters
        ----------
        ascending : bool, default=True
            Whether to sort the index in ascending (default) or descending order.

        Returns
        -------
        Series
            A new Series sorted by index.

        """
        idx = self.index.argsort(ascending=ascending)
        return self._reindex(idx)

    @typechecked
    def sort_values(self, ascending: bool = True) -> Series:
        """
        Sort the Series by its values.

        Parameters
        ----------
        ascending : bool, default=True
            Whether to sort values in ascending (default) or descending order.

        Returns
        -------
        Series
            A new Series sorted by its values.

        """
        if not ascending:
            if isinstance(self.values, pdarray) and self.values.dtype in (
                int64,
                float64,
            ):
                # For numeric values, negation reverses sort order
                idx = argsort(-self.values)
            else:
                # For non-numeric values, need the descending arange because reverse slicing
                # is not supported
                idx = argsort(self.values)[arange(self.values.size - 1, -1, -1)]
        else:
            idx = argsort(self.values)
        return self._reindex(idx)

    @typechecked
    def tail(self, n: int = 10) -> Series:
        """Return the last n values of the series"""
        idx_series = self.index[-n:]
        return Series(index=idx_series.index, data=self.values[-n:])

    @typechecked
    def head(self, n: int = 10) -> Series:
        """Return the first n values of the series"""
        idx_series = self.index[0:n]
        return Series(index=idx_series.index, data=self.values[0:n])

    @typechecked
    def to_pandas(self) -> pd.Series:
        """Convert the series to a local PANDAS series"""
        import copy

        from arkouda.numpy.segarray import SegArray

        idx = self.index.to_pandas()

        if isinstance(self.values, Categorical):
            val = self.values.to_pandas()
        elif isinstance(self.values, SegArray):
            # pandas errors when ndarray formatted like a segarray is
            # passed into Series but works when it's just a list of lists
            val = self.values.to_list()
        else:
            val = self.values.to_ndarray()

        if isinstance(self.name, str):
            name = copy.copy(self.name)
            return pd.Series(val, index=idx, name=name)
        else:
            return pd.Series(val, index=idx)

    def to_markdown(self, mode="wt", index=True, tablefmt="grid", storage_options=None, **kwargs):
        r"""
        Print Series in Markdown-friendly format.

        Parameters
        ----------
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.
        tablefmt: str = "grid"
            Table format to call from tablulate:
            https://pypi.org/project/tabulate/
        storage_options: dict, optional
            Extra options that make sense for a particular storage connection,
            e.g. host, port, username, password, etc., if using a URL that will be parsed by fsspec,
            e.g., starting “s3://”, “gcs://”.
            An error will be raised if providing this argument with a non-fsspec URL.
            See the fsspec and backend storage implementation docs for the set
            of allowed keys and values.

        **kwargs
            These parameters will be passed to tabulate.

        Note
        ----
        This function should only be called on small Series as it calls pandas.Series.to_markdown:
        https://pandas.pydata.org/docs/reference/api/pandas.Series.to_markdown.html

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> s = ak.Series(["elk", "pig", "dog", "quetzal"], name="animal")
        >>> print(s.to_markdown())
        |    | animal   |
        |---:|:---------|
        |  0 | elk      |
        |  1 | pig      |
        |  2 | dog      |
        |  3 | quetzal  |

        Output markdown with a tabulate option.

        >>> print(s.to_markdown(tablefmt="grid"))
        +----+----------+
        |    | animal   |
        +====+==========+
        |  0 | elk      |
        +----+----------+
        |  1 | pig      |
        +----+----------+
        |  2 | dog      |
        +----+----------+
        |  3 | quetzal  |
        +----+----------+

        """
        return self.to_pandas().to_markdown(
            mode=mode,
            index=index,
            tablefmt=tablefmt,
            storage_options=storage_options,
            **kwargs,
        )

    @typechecked()
    def to_list(self) -> list:
        p = self.to_pandas()
        return p.to_list()

    @typechecked
    def value_counts(self, sort: bool = True) -> Series:
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        sort : bool, default=True
            Whether to sort the result by count in descending order. If False,
            the order of the results is not guaranteed.

        Returns
        -------
        Series
            A Series where the index contains the unique values and the values are
            their counts in the original Series.

        """
        from arkouda.numpy import value_counts

        dtype = get_callback(self.values)
        idx, vals = value_counts(self.values)
        s = Series(index=idx, data=vals)
        if sort:
            s = s.sort_values(ascending=False)
        s.index.set_dtype(dtype)
        return s

    @typechecked
    def diff(self) -> Series:
        """
        Diffs consecutive values of the series.

        Returns a new series with the same index and length.  First value is set to NaN.
        """
        values = zeros(len(self), "float64")
        if not isinstance(self.values, Categorical):
            values[1:] = akcast(self.values[1:] - self.values[:-1], "float64")
            values[0] = np.nan
        else:
            raise TypeError("Diff not supported on Series built from Categorical.")

        return Series(data=values, index=self.index)

    @typechecked
    def to_dataframe(
        self,
        index_labels: Union[List[builtin_str], None] = None,
        value_label: Union[builtin_str, None] = None,
    ) -> arkouda.pandas.dataframe.DataFrame:
        """
        Convert the Series to an Arkouda DataFrame.

        Parameters
        ----------
        index_labels : list of str or None, optional
            Column name(s) to label the index.
        value_label : str or None, optional
            Column name to label the values.

        Returns
        -------
        DataFrame
            An Arkouda DataFrame representing the Series.

        """
        list_value_label = [value_label] if isinstance(value_label, str) else value_label

        return Series.concat([self], axis=1, index_labels=index_labels, value_labels=list_value_label)

    @typechecked
    def register(self, user_defined_name: builtin_str):
        """
        Register this Series object and underlying components with the Arkouda server

        Parameters
        ----------
        user_defined_name : str
            user defined name the Series is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        Series
            The same Series which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different Series with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the Series with the user_defined_name

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
                "objType": self.objType,
                "num_idxs": 1,
                "idx_names": [
                    (
                        json.dumps(
                            {
                                "codes": self.index.values.codes.name,
                                "categories": self.index.values.categories.name,
                                "NA_codes": self.index.values._akNAcode.name,
                                **(
                                    {"permutation": self.index.values.permutation.name}
                                    if self.index.values.permutation is not None
                                    else {}
                                ),
                                **(
                                    {"segments": self.index.values.segments.name}
                                    if self.index.values.segments is not None
                                    else {}
                                ),
                            }
                        )
                        if isinstance(self.index.values, Categorical)
                        else self.index.values.name
                    )
                ],
                "idx_types": [self.index.values.objType],
                "values": (
                    json.dumps(
                        {
                            "codes": self.values.codes.name,
                            "categories": self.values.categories.name,
                            "NA_codes": self.values._akNAcode.name,
                            **(
                                {"permutation": self.values.permutation.name}
                                if self.values.permutation is not None
                                else {}
                            ),
                            **(
                                {"segments": self.values.segments.name}
                                if self.values.segments is not None
                                else {}
                            ),
                        }
                    )
                    if isinstance(self.values, Categorical)
                    else self.values.name
                ),
                "val_type": self.values.objType,
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this Series object in the arkouda server which was previously
        registered using register() and/or attached to using attach()

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

    @typechecked
    def is_registered(self) -> bool:
        """
         Return True iff the object is contained in the registry or is a component of a
         registered object.

        Returns
        -------
        bool
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
            return False
        else:
            return is_registered(self.registered_name)

    @classmethod
    @typechecked
    def from_return_msg(cls, repMsg: builtin_str) -> Series:
        """
        Return a Series instance pointing to components created by the arkouda server.
        The user should not call this function directly.

        Parameters
        ----------
        repMsg : str
            + delimited string containing the values and indexes

        Returns
        -------
        Series
            A Series representing a set of pdarray components on the server

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown in the process of creating
            the Series instance

        """
        data = json.loads(repMsg)
        val_comps = data["value"].split("+|+")
        if val_comps[0] == Categorical.objType.upper():
            values = Categorical.from_return_msg(val_comps[1])
        elif val_comps[0] == Strings.objType.upper():
            values = Strings.from_return_msg(val_comps[1])  # type: ignore
        else:
            values = create_pdarray(val_comps[1])

        index = Index.from_return_msg(data["index"])

        return cls(values, index)

    @staticmethod
    @typechecked
    def _all_aligned(array: List) -> bool:
        """Is an array of Series indexed aligned?"""
        itor = iter(array)
        a1 = next(itor).index
        for a2 in itor:
            if a1._check_aligned(a2.index) is False:
                return False
        return True

    @staticmethod
    @typechecked
    def concat(
        arrays: List,
        axis: int = 0,
        index_labels: Union[List[builtin_str], None] = None,
        value_labels: Union[List[builtin_str], None] = None,
        ordered: bool = False,
    ) -> Union[arkouda.pandas.dataframe.DataFrame, Series]:
        """
        Concatenate a list of Arkouda Series or grouped arrays horizontally or vertically.

        If a list of grouped Arkouda arrays is passed, they are converted to Series. Each grouping
        is a 2-tuple where the first item is the key(s) and the second is the value. If concatenating
        horizontally (axis=1), all series/groupings must have the same length and the same index.
        The index is converted to a column in the resulting DataFrame; if it's a MultiIndex,
        each level is converted to a separate column.

        Parameters
        ----------
        arrays : List
            A list of Series or groupings (tuples of index and values) to concatenate.
        axis : int, default=0
            The axis to concatenate along:
            - 0 = vertical (stack series into one)
            - 1 = horizontal (align by index and produce a DataFrame)
        index_labels : List of str or None, optional
            Column name(s) to label the index when axis=1.
        value_labels : List of str or None, optional
            Column names to label the values of each Series.
        ordered : bool, default=False
            Unused parameter. Reserved for future support of deterministic
            vs. performance-optimized concatenation.

        Returns
        -------
        Series or DataFrame
            - If axis=0: a new Series
            - If axis=1: a new DataFrame

        """
        if len(arrays) == 0:
            raise IndexError("Array length must be non-zero")

        types = {type(x) for x in arrays}
        if len(types) != 1:
            raise TypeError(f"Items must all have same type: {types}")

        if isinstance(arrays[0], tuple):
            arrays = [Series(i) for i in arrays]

        if axis == 1:
            # Horizontal concat
            if value_labels is None:
                value_labels = [f"val_{i}" for i in range(len(arrays))]

            if Series._all_aligned(arrays):
                data = next(iter(arrays)).index.to_dict(index_labels)

                if value_labels is not None:
                    # Expect value_labels to always be not None; were doing the check for mypy
                    for col, label in zip(arrays, value_labels):
                        data[str(label)] = col.values

            else:
                aitor = iter(arrays)
                idx = next(aitor).index
                idx = idx._merge_all([i.index for i in aitor])

                data = idx.to_dict(index_labels)

                if value_labels is not None:
                    # Expect value_labels to always be not None; were doing the check for mypy
                    for col, label in zip(arrays, value_labels):
                        data[str(label)] = lookup(col.index.index, col.values, idx.index, fillvalue=0)

            return arkouda.pandas.dataframe.DataFrame(data)
        else:
            # Vertical concat
            idx = arrays[0].index
            v = arrays[0].values
            for other in arrays[1:]:
                idx = idx.concat(other.index)
                v = concatenate([v, other.values], ordered=True)
            return Series(index=idx.index, data=v)

    def map(self, arg: Union[dict, Series]) -> Series:
        """
        Map values of Series according to an input mapping.

        Parameters
        ----------
        arg : dict or Series
            The mapping correspondence.

        Returns
        -------
        Series
            A new series with the same index as the caller.
            When the input Series has Categorical values,
            the return Series will have Strings values.
            Otherwise, the return type will match the input type.

        Raises
        ------
        TypeError
            Raised if arg is not of type dict or arkouda.Series.
            Raised if series values not of type pdarray, Categorical, or Strings.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> s = ak.Series(ak.array([2, 3, 2, 3, 4]))
        >>> s

        +----+-----+
        |    | 0   |
        +====+=====+
        |  0 | 2   |
        +----+-----+
        |  1 | 3   |
        +----+-----+
        |  2 | 2   |
        +----+-----+
        |  3 | 3   |
        +----+-----+
        |  4 | 4   |
        +----+-----+

        >>> s.map({4: 25.0, 2: 30.0, 1: 7.0, 3: 5.0})

        +----+-----+
        |    | 0   |
        +====+=====+
        |  0 | 30.0|
        +----+-----+
        |  1 | 5.0 |
        +----+-----+
        |  2 | 30.0|
        +----+-----+
        |  3 | 5.0 |
        +----+-----+
        |  4 | 25.0|
        +----+-----+

        >>> s2 = ak.Series(ak.array(["a","b","c","d"]), index = ak.array([4,2,1,3]))
        >>> s.map(s2)

        +----+-----+
        |    | 0   |
        +====+=====+
        |  0 | b   |
        +----+-----+
        |  1 | d   |
        +----+-----+
        |  2 | b   |
        +----+-----+
        |  3 | d   |
        +----+-----+
        |  4 | a   |
        +----+-----+

        """
        from arkouda import Series
        from arkouda.numpy.util import map

        return Series(map(self.values, arg), index=self.index)

    def isna(self) -> Series:
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA. NA values,
        such as numpy.NaN, gets mapped to True values.
        Everything else gets mapped to False values.
        Characters such as empty strings '' are not considered NA values.

        Returns
        -------
        Series
            Mask of bool values for each element in Series
            that indicates whether an element is an NA value.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> from arkouda import Series
        >>> import numpy as np

        >>> s = Series(ak.array([1, 2, np.nan]), index = ak.array([1, 2, 4]))
        >>> s.isna()

        +----+---------+
        |    |   0     |
        +====+=========+
        |  1 |   False |
        +----+---------+
        |  2 |   False |
        +----+---------+
        |  4 |   True  |
        +----+---------+

        """
        from arkouda.numpy import isnan

        if not is_float(self.values):
            return Series(full(self.values.size, False, dtype=bool), index=self.index)

        return Series(isnan(self.values), index=self.index)

    def isnull(self) -> Series:
        """
        Series.isnull is an alias for Series.isna.

        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA. NA values,
        such as numpy.NaN, gets mapped to True values.
        Everything else gets mapped to False values.
        Characters such as empty strings '' are not considered NA values.

        Returns
        -------
        Series
            Mask of bool values for each element in Series
            that indicates whether an element is an NA value.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> from arkouda import Series
        >>> import numpy as np

        >>> s = Series(ak.array([1, 2, np.nan]), index = ak.array([1, 2, 4]))
        >>> s.isnull()

        +----+---------+
        |    |   0     |
        +====+=========+
        |  1 |   False |
        +----+---------+
        |  2 |   False |
        +----+---------+
        |  4 |   True  |
        +----+---------+

        """
        return self.isna()

    def notna(self) -> Series:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True.
        Characters such as empty strings '' are not considered NA values.
        NA values, such as numpy.NaN, get mapped to False values.

        Returns
        -------
        Series
            Mask of bool values for each element in Series
            that indicates whether an element is not an NA value.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> from arkouda import Series
        >>> import numpy as np

        >>> s = Series(ak.array([1, 2, np.nan]), index = ak.array([1, 2, 4]))
        >>> s.notna()

        +----+---------+
        |    |   0     |
        +====+=========+
        |  1 |   True  |
        +----+---------+
        |  2 |   True  |
        +----+---------+
        |  4 |   False |
        +----+---------+

        """
        from arkouda.numpy import isnan

        if not is_float(self.values):
            return Series(full(self.values.size, True, dtype=bool), index=self.index)

        return Series(~isnan(self.values), index=self.index)

    def notnull(self) -> Series:
        """
        Series.notnull is an alias for Series.notna.

        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True.
        Characters such as empty strings '' are not considered NA values.
        NA values, such as numpy.NaN, get mapped to False values.

        Returns
        -------
        Series
            Mask of bool values for each element in Series
            that indicates whether an element is not an NA value.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> from arkouda import Series
        >>> import numpy as np

        >>> s = Series(ak.array([1, 2, np.nan]), index = ak.array([1, 2, 4]))
        >>> s.notnull()

        +----+---------+
        |    |   0     |
        +====+=========+
        |  1 |   True  |
        +----+---------+
        |  2 |   True  |
        +----+---------+
        |  4 |   False |
        +----+---------+

        """
        return self.notna()

    def hasnans(self) -> bool_scalars:
        """
        Return True if there are any NaNs.

        Returns
        -------
        bool

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> from arkouda import Series
        >>> import numpy as np

        >>> s = ak.Series(ak.array([1, 2, 3, np.nan]))
        >>> s

        +----+-------+
        |    |   0   |
        +====+=======+
        |  0 |   1.0 |
        +----+-------+
        |  1 |   2.0 |
        +----+-------+
        |  2 |   3.0 |
        +----+-------+
        |  3 |   nan |
        +----+-------+

        >>> s.hasnans()
        True

        """
        from arkouda.numpy import isnan

        if is_float(self.values):
            result = any(isnan(self.values))
            if isinstance(result, (bool, np.bool_)):
                return result

        return False

    def fillna(self, value: Union[supported_scalars, Series, pdarray]) -> Series:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : supported_scalars, Series, or pdarray
            Value to use to fill holes (e.g. 0), alternately a
            Series of values specifying which value to use for
            each index.  Values not in the Series will not be filled.
            This value cannot be a list.

        Returns
        -------
        Series
            Object with missing values filled.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> from arkouda import Series

        >>> data = ak.Series([1, np.nan, 3, np.nan, 5])
        >>> data

        +----+-------+
        |    |   0   |
        +====+=======+
        |  0 |   1.0 |
        +----+-------+
        |  1 |   nan |
        +----+-------+
        |  2 |   3.0 |
        +----+-------+
        |  3 |   nan |
        +----+-------+
        |  4 |   5.0 |
        +----+-------+

        >>> fill_values1 = ak.ones(5)
        >>> data.fillna(fill_values1)

        +----+-------+
        |    |   0   |
        +====+=======+
        |  0 |   1.0 |
        +----+-------+
        |  1 |   1.0 |
        +----+-------+
        |  2 |   3.0 |
        +----+-------+
        |  3 |   1.0 |
        +----+-------+
        |  4 |   5.0 |
        +----+-------+

        >>> fill_values2 = Series(ak.ones(5))
        >>> data.fillna(fill_values2)

        +----+-------+
        |    |   0   |
        +====+=======+
        |  0 |   1.0 |
        +----+-------+
        |  1 |   1.0 |
        +----+-------+
        |  2 |   3.0 |
        +----+-------+
        |  3 |   1.0 |
        +----+-------+
        |  4 |   5.0 |
        +----+-------+

        >>> fill_values3 = 100.0
        >>> data.fillna(fill_values3)

        +----+---------+
        |    |     0   |
        +====+=========+
        |  0 |     1.0 |
        +----+---------+
        |  1 |   100.0 |
        +----+---------+
        |  2 |     3.0 |
        +----+---------+
        |  3 |   100.0 |
        +----+---------+
        |  4 |     5.0 |
        +----+---------+

        """
        from arkouda.numpy import isnan, where

        if isinstance(value, Series):
            value = value.values

        if isinstance(self.values, pdarray) and is_float(self.values):
            return Series(where(isnan(self.values), value, self.values), index=self.index)
        else:
            return Series(self.values, index=self.index)

    @staticmethod
    @typechecked
    def pdconcat(
        arrays: List, axis: int = 0, labels: Union[Strings, None] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Concatenate a list of Arkouda Series or grouped arrays, returning a local pandas object.

        If a list of grouped Arkouda arrays is passed, they are converted to Series. Each grouping
        is a 2-tuple with the first item being the key(s) and the second the value.

        If `axis=1` (horizontal), each Series or grouping must have the same length and the same index.
        The index is converted to a column in the resulting DataFrame. If it is a MultiIndex,
        each level is converted to a separate column.

        Parameters
        ----------
        arrays : List
            A list of Series or groupings (tuples of index and values) to concatenate.
        axis : int, default=0
            The axis along which to concatenate:
            - 0 = vertical (stack into a Series)
            - 1 = horizontal (align by index into a DataFrame)
        labels : Strings or None, optional
            Names to assign to the resulting columns in the DataFrame.

        Returns
        -------
        Series or DataFrame
            - If axis=0: a local pandas Series
            - If axis=1: a local pandas DataFrame

        """
        if len(arrays) == 0:
            raise IndexError("Array length must be non-zero")

        types = {type(x) for x in arrays}
        if len(types) != 1:
            raise TypeError(f"Items must all have same type: {types}")

        if isinstance(arrays[0], tuple):
            arrays = [Series(i) for i in arrays]

        if axis == 1:
            idx = arrays[0].index.to_pandas()

            cols = []
            for col in arrays:
                cols.append(pd.Series(data=col.values.to_ndarray(), index=idx))
            retval = pd.concat(cols, axis=1)
            if labels is not None:
                retval.columns = pd.Index(labels)
        else:
            retval = pd.concat([s.to_pandas() for s in arrays])

        return retval


class _LocIndexer:
    def __init__(self, series):
        self.series = series

    def __getitem__(self, key):
        return self.series[key]

    def __setitem__(self, key, val):
        self.series[key] = val


class _iLocIndexer:
    def __init__(self, method_name, series):
        self.name = method_name
        self.series = series

    def validate_key(self, key) -> Union[pdarray, int]:
        if isinstance(key, list):
            key = array(key)
        if isinstance(key, tuple):
            raise TypeError(".{} does not support tuple arguments".format(self.name))
        if isinstance(key, pdarray):
            if len(key) == 0:
                raise ValueError("Cannot index using 0-length iterables.")
            if key.dtype != int64 and key.dtype != bool:
                raise TypeError(".{} requires integer keys".format(self.name))

            if key.dtype == "bool_" and key.size != self.series.size:
                raise IndexError(
                    "Boolean index has wrong length: {} instead of {}".format(key.size, self.series.size)
                )
            elif any(key >= self.series.size):
                raise IndexError("{} cannot enlarge its target object.".format(self.name))

        elif isinstance(key, int):
            if key >= self.series.size:
                raise IndexError("{} cannot enlarge its target object.".format(self.name))
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.series.size
            stride = key.step if key.step is not None else 1
            if start < 0:
                raise IndexError("Slice start must be non-negative")
            if stop > self.series.size:
                raise IndexError("Slice stop must be less than or equal to the size of the Series")
            if start > stop:
                raise IndexError("Slice start must be less than or equal to the stop")
            key = arange(start, stop, stride)
        else:
            raise TypeError(".{} requires integer keys".format(self.name))

        return key

    def validate_val(self, val) -> Union[pdarray, supported_scalars]:
        return self.series.validate_val(val)

    def __getitem__(self, key):
        key = self.validate_key(key)
        if is_supported_scalar(key):
            key = array([key])
        return Series(index=self.series.index[key], data=self.series.values[key])

    def __setitem__(self, key, val):
        key = self.validate_key(key)
        val = self.validate_val(val)

        if is_supported_scalar(val):
            self.series.values[key] = val
            return
        else:
            if is_supported_scalar(key):
                self.series.values[key] = val
                return
            if key.dtype == int64 and len(val) != len(key):
                raise ValueError(
                    "cannot set using a list-like indexer with a different length than the value"
                )
        self.series.values[key] = val
