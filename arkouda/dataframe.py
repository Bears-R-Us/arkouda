from __future__ import annotations

from collections import UserDict
from warnings import warn
import pandas as pd  # type: ignore
import random

from arkouda.segarray import SegArray
from arkouda.pdarrayclass import pdarray
from arkouda.categorical import Categorical
from arkouda.strings import Strings
from arkouda.pdarraycreation import arange, array
from arkouda.groupbyclass import GroupBy
from arkouda.pdarraysetops import concatenate, unique, intersect1d, in1d
from arkouda.pdarrayIO import save_all
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import float64 as akfloat64
from arkouda.sorting import argsort, coargsort
from arkouda.numeric import where
from arkouda.client import maxTransferBytes
from arkouda.row import Row

# This is necessary for displaying DataFrames with BitVector columns,
# because pandas _html_repr automatically truncates the number of displayed bits
pd.set_option('display.max_colwidth', 64)

__all__ = [
    "DataFrame",
    "sorted",
    "intersect",
    "invert_permutation",
    "intx",
]


def groupby_operators(cls):
    for name in ['all', 'any', 'argmax', 'argmin', 'max', 'mean', 'min', 'nunique', 'prod', 'sum', 'OR', 'AND', 'XOR']:
        setattr(cls, name, cls._make_aggop(name))
    return cls


"""
DataFrame structure based on Arkouda arrays.
"""


class DataFrame(UserDict):
    """
    A DataFrame structure based on arkouda arrays.

    Examples
    --------

    Create an empty DataFrame and add a column of data:

    >>> import arkouda as ak
    >>> import numpy as np
    >>> df = ak.DataFrame()
    >>> df['a'] = ak.array([1,2,3])

    Create a new DataFrame using a dictionary of data:

    >>> userName = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    >>> userID = ak.array([111, 222, 111, 333, 222, 111])
    >>> item = ak.array([0, 0, 1, 1, 2, 0])
    >>> day = ak.array([5, 5, 6, 5, 6, 6])
    >>> amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    >>> df = ak.DataFrame({'userName': userName, 'userID': userID,
    >>>            'item': item, 'day': day, 'amount': amount})
    >>> df
    DataFrame(['userName', 'userID', 'item', 'day', 'amount'] [6 rows : 224 B])

    Indexing works slightly differently than with pandas:
    >>> df[0]
    {'userName': 'Alice', 'userID': 111, 'item': 0, 'day': 5, 'amount': 0.5}
    >>> df['userID']
    array([111, 222, 111, 333, 222, 111])
    >>> df['userName']
    array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    >>> df[[1,5,7]]
      userName  userID  item  day  amount
    1      Bob     222     0    5     0.6
    2    Alice     111     1    6     1.1
    3    Carol     333     1    5     1.2

    Note that strides are not implemented except for stride = 1.
    >>> df[1:5:1]
    DataFrame(['userName', 'userID', 'item', 'day', 'amount'] [4 rows : 148 B])
    >>> df[ak.array([1,2,3])]
    DataFrame(['userName', 'userID', 'item', 'day', 'amount'] [3 rows : 112 B])
    >>> df[['userID', 'day']]
    DataFrame(['userID', 'day'] [6 rows : 96 B])
    """

    COLUMN_CLASSES = (pdarray, Strings, Categorical, SegArray)

    def __init__(self, initialdata=None):
        super().__init__()

        if isinstance(initialdata, DataFrame):
            # Copy constructor
            self._size = initialdata._size
            self._bytes = initialdata._bytes
            self._empty = initialdata._empty
            self._columns = initialdata._columns
            self.data = initialdata.data
            self.update_size()
            return

        # Some metadata about this dataframe.
        self._size = 0
        self._bytes = 0
        self._empty = True

        # Initial attempts to keep an order on the columns
        self._columns = ['index']
        self.data['index'] = None

        # Add data to the DataFrame if there is any
        if initialdata is not None:
            # Used to prevent uneven array length in initialization.
            sizes = set()

            # Initial data is a dictionary of arkouda arrays
            if type(initialdata) == dict:
                for key, val in initialdata.items():
                    if not isinstance(val, self.COLUMN_CLASSES):
                        raise ValueError(f"Values must be one of {self.COLUMN_CLASSES}.")
                    sizes.add(val.size)
                    if len(sizes) > 1:
                        raise ValueError("Input arrays must have equal size.")
                    self._empty = False
                    UserDict.__setitem__(self, key, val)
                    # Update the column index
                    if key != 'index':
                        self._columns.append(key)

            # Initial data is a list of arkouda arrays
            elif type(initialdata) == list:
                # Create string IDs for the columns
                keys = [str(x) for x in range(len(initialdata))]
                for key, col in zip(keys, initialdata):
                    if not isinstance(col, self.COLUMN_CLASSES):
                        raise ValueError(f"Values must be one of {self.COLUMN_CLASSES}.")
                    sizes.add(col.size)
                    if len(sizes) > 1:
                        raise ValueError("Input arrays must have equal size.")
                    self._empty = False
                    UserDict.__setitem__(self, key, col)
                    # Update the column index
                    self._columns.append(key)

            # Initial data is invalid.
            else:
                raise ValueError(f"Initialize with dict or list of {self.COLUMN_CLASSES}.")

            # Update the dataframe indices and metadata.
            if len(sizes) > 0:
                self._size = sizes.pop()
            # If the index column was passed in, use that instead of
            # creating a new one.
            if self.data['index'] is None:
                self.data['index'] = arange(0, self._size, 1)

            self.update_size()

    def __delitem__(self, key):
        # This function is a backdoor to messing up the indices and columns.
        # I needed to reimplement it to prevent bad behavior
        if key == 'index':
            raise KeyError('The index column may be reset, but not dropped.')
        else:
            UserDict.__delitem__(self, key)
            self._columns.remove(key)

        # If removing this column emptied the dataframe
        if len(self._columns) == 1:
            self.data['index'] = None
            self._empty = True
        self.update_size()

    def __getitem__(self, key):
        # Select rows using an integer pdarray
        if isinstance(key, pdarray):
            result = {}
            for k in self._columns:
                result[k] = UserDict.__getitem__(self, k)[key]
            return DataFrame(initialdata=result)

        # Select rows or columns using a list
        if isinstance(key, list):
            result = DataFrame()
            if len(key) <= 0:
                return result
            if len({type(x) for x in key}) > 1:
                raise TypeError("Invalid selector: too many types in list.")
            if type(key[0]) == int:
                rows = array(key)
                for k in self.data.keys():
                    result.data[k] = UserDict.__getitem__(self, k)[rows]
                    result._columns.append(k)
                result._empty = False
                return result
            elif type(key[0]) == str:
                # Grab the index column as well.
                result.data['index'] = UserDict.__getitem__(self, 'index')
                for k in key:
                    result.data[k] = UserDict.__getitem__(self, k)
                    result._columns.append(k)
                result._empty = False
                return result

        # Select a single row using an integer
        if isinstance(key, int):
            result = {}
            row = array([key])
            for k in self.data.keys():
                result[k] = (UserDict.__getitem__(self, k)[row])[0]
            return Row(result)

        # Select a single column using a string
        elif isinstance(key, str):
            if key not in self.keys():
                raise KeyError("Invalid column name '{}'.".format(key))
            return UserDict.__getitem__(self, key)

        # Select rows using a slice
        elif isinstance(key, slice):
            # result = DataFrame()
            data = {}
            s = key
            for k in self._columns:
                data[k] = UserDict.__getitem__(self, k)[s.start:s.stop:s.step]
            # result._columns.append(k)
            # result._empty = False
            return DataFrame(initialdata=data)

        else:
            raise IndexError("Invalid selector: unknown error.")

    def __setitem__(self, key, value):
        self.update_size()

        # If this is the first column added, we must create an index column.
        add_index = False
        if self._empty:
            add_index = True

        # Set a single row in the dataframe using a dict of values
        if type(key) == int:
            for k in self.data.keys():
                if isinstance(self.data[k], Strings):
                    raise ValueError("This DataFrame has a column of type ak.Strings;"
                                     " so this DataFrame is immutable. This feature could change"
                                     " if arkouda supports mutable Strings in the future.")
            if self._empty:
                raise ValueError("Initial data must be dict of arkouda arrays.")
            elif not isinstance(value, (dict, UserDict)):
                raise ValueError("Expected dict or Row type.")
            elif key >= self._size:
                raise KeyError("The row index is out of range.")
            else:
                for k, v in value.items():
                    if k == 'index':
                        continue
                    self[k][key] = v

        # Set a single column in the dataframe using a an arkouda array
        elif type(key) == str:
            if not isinstance(value, self.COLUMN_CLASSES):
                raise ValueError(f"Column must be one of {self.COLUMN_CLASSES}.")
            elif self._size is not None and self._size != value.size:
                raise ValueError("Expected size {} but received size {}.".format(self.size, value.size))
            else:
                self._empty = False
                UserDict.__setitem__(self, key, value)
                # Update the index values
                if key not in self._columns:
                    self._columns.append(key)

        # Do nothing and return if there's no valid data
        else:
            raise ValueError("No valid data received.")

        # Update the dataframe indices and metadata.
        if add_index:
            self.update_size()
            self.data['index'] = arange(0, self._size, 1)

    def __len__(self):
        """
        Return the number of rows
        """
        return self.size

    def _ncols(self):
        """
        Only count the non-index columns.
        """
        return len(list(self.data.keys())) - 1

    def __str__(self):
        """
        Returns a summary string of this dataframe.
        """

        self.update_size()

        if self._empty:
            return 'DataFrame([ -- ][ 0 rows : 0 B])'

        keys = [str(key) for key in list(self.data.keys())]
        keys = [("'" + key + "'") for key in keys]
        keystr = ", ".join(keys)

        # first call to memory_usage() initializes self._bytes
        mem = self.memory_usage()

        # Get units that make the most sense.
        if self._bytes < 1024:
            mem = self.memory_usage(unit='B')
        elif self._bytes < 1024 ** 2:
            mem = self.memory_usage(unit='KB')
        elif self._bytes < 1024 ** 3:
            mem = self.memory_usage(unit='MB')
        else:
            mem = self.memory_usage(unit='GB')
        rows = " rows"
        if self._size == 1:
            rows = " row"
        return 'DataFrame([' + keystr + '], {:,}'.format(self._size) + rows + ', ' + str(mem) + ')'

    def _get_head_tail(self):
        if self._empty:
            return pd.DataFrame()
        self.update_size()
        maxrows = pd.get_option('display.max_rows')
        if self._size <= maxrows:
            newdf = DataFrame()
            for col in self._columns:
                if isinstance(self[col], Categorical):
                    newdf[col] = self[col].categories[self[col].codes]
                else:
                    newdf[col] = self[col]
            return newdf.to_pandas(retain_index=True)
        # Being 1 above the threshold caises the PANDAS formatter to split the data frame vertically
        idx = array(list(range(maxrows // 2 + 1)) + list(range(self._size - (maxrows // 2), self._size)))
        newdf = DataFrame()
        for col in self._columns[1:]:
            if isinstance(self[col], Categorical):
                newdf[col] = self[col].categories[self[col].codes[idx]]
            else:
                newdf[col] = self[col][idx]
        newdf['index'] = self['index'][idx]
        return newdf.to_pandas(retain_index=True)

    def _shape_str(self):
        return "{} rows x {} columns".format(self.size, self._ncols())

    def __repr__(self):
        """
        Return ascii-formatted version of the dataframe.
        """

        prt = self._get_head_tail()
        with pd.option_context("display.show_dimensions", False):
            retval = prt.__repr__()
        retval += self._shape_str()
        return retval

    def _repr_html_(self):
        """
        Return html-formatted version of the dataframe.
        """

        prt = self._get_head_tail()
        with pd.option_context("display.show_dimensions", False):
            retval = prt._repr_html_()
        retval += "<p>" + self._shape_str() + "</p>"
        return retval

    def _ipython_key_completions_(self):
        return self._columns

    def drop(self, keys):
        """
        Drop a column or columns from the dataframe, in-place.

        Parameters
        ----------
        keys : str or list
            The column(s) to be dropped.
        """

        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            # Do not allow the user to drop the index column
            if key == 'index':
                raise KeyError('The index column may be reset, but not dropped.')

            del self[key]

        # If the dataframe just became empty...
        if len(self._columns) == 1:
            self.data['index'] = None
            self._empty = True
        self.update_size()

    def drop_duplicates(self, subset=None, keep='first'):
        """
        Drops duplcated rows and returns resulting DataFrame.

        If a subset of the columns are provided then only one instance of each
        duplicated row will be returned (keep determines which row).

        Parameters
        ----------
        subset : Iterable of column names to use to dedupe.
        keep : {'first', 'last'}, default 'first'
            Determines which duplicates (if any) to keep.

        Returns
        -------
        DataFrame
            DataFrame with duplicates removed.
        """
        if self._empty:
            return self

        if not subset:
            subset = self._columns[1:]

        if len(subset) == 1:
            if not subset[0] in self.data:
                raise KeyError("{} is not a column in the DataFrame.".format(subset[0]))
            _ = GroupBy(self.data[subset[0]])

        else:
            for col in subset:
                if not col in self.data:
                    raise KeyError("{} is not a column in the DataFrame.".format(subset[0]))

            _ = GroupBy([self.data[col] for col in subset])

        if keep == 'last':
            _segment_ends = concatenate([_.segments[1:] - 1, array([_.permutation.size - 1])])
            return self[_.permutation[_segment_ends]]
        else:
            return self[_.permutation[_.segments]]

    @property
    def size(self):
        """
        Returns the number of bytes on the arkouda server.
        """

        self.update_size()
        if self._size == None:
            return 0
        return self._size

    @property
    def dtypes(self):
        dtypes = []
        keys = []
        for key, val in self.items():
            keys.append(key)
            if isinstance(val, pdarray):
                dtypes.append(str(val.dtype))
            elif isinstance(val, Strings):
                dtypes.append('str')
        res = Row({key: dtype for key, dtype in zip(keys, dtypes)})
        return res

    @property
    def empty(self):
        return self._empty

    @property
    def shape(self):
        self.update_size()
        num_cols = len(self._columns) - 1
        num_rows = self._size
        return (num_rows, num_cols)

    @property
    def columns(self):
        return self._columns

    @property
    def index(self):
        return self.data['index']

    def reset_index(self, size=False):
        """
        Set the index to an integer range.

        Useful if this dataframe is the result of a slice operation from
        another dataframe, or if you have permuted the rows and no longer need
        to keep that ordering on the rows.

        Parameters
        ----------
        size : int
            If size is passed, do not attempt to determine size based on
            existing column sizes. Assume caller handles consistency correctly.
        """

        if not size:
            self.update_size()
            self.data['index'] = arange(0, self._size)
        else:
            self.data['index'] = arange(size)

    @property
    def info(self):
        """
        Returns a summary string of this dataframe.
        """

        self.update_size()

        if self._size is None:
            return 'DataFrame([ -- ][ 0 rows : 0 B])'

        keys = [str(key) for key in list(self.data.keys())]
        keys = [("'" + key + "'") for key in keys]
        keystr = ", ".join(keys)

        # first call to memory_usage() initializes self._bytes
        mem = self.memory_usage()

        # Get units that make the most sense.
        if self._bytes < 1024:
            mem = self.memory_usage(unit='B')
        elif self._bytes < 1024 ** 2:
            mem = self.memory_usage(unit='KB')
        elif self._bytes < 1024 ** 3:
            mem = self.memory_usage(unit='MB')
        else:
            mem = self.memory_usage(unit='GB')
        rows = " rows"
        if self._size == 1:
            rows = " row"
        return 'DataFrame([' + keystr + '], {:,}'.format(self._size) + rows + ', ' + str(mem) + ')'

    def update_size(self):
        """
        Computes the number of bytes on the arkouda server.
        """

        sizes = set()
        for key, val in self.items():
            if val is not None:
                sizes.add(val.size)
        if len(sizes) > 1:
            raise ValueError("Size mismatch in DataFrame columns.")
        if len(sizes) == 0:
            self._size = None
        else:
            self._size = sizes.pop()

    def rename(self, mapper):
        """
        Rename columns in-place according to a mapping.

        Parameters
        ----------
        mapper : callable or dict-like
            Function or dictionary mapping existing column names to
            new column names. Nonexistent names will not raise an
            error.

        Returns
        -------
        self
            Renaming occurs in-place, but result is also returned,
            for compatibility.
        """

        if callable(mapper):
            # Do not rename index, start at 1
            for i in range(1, len(self._columns)):
                oldname = self._columns[i]
                newname = mapper(oldname)
                # Only rename if name has changed
                if newname != oldname:
                    self._columns[i] = newname
                    self.data[newname] = self.data[oldname]
                    del self.data[oldname]
        elif isinstance(mapper, dict):
            for oldname, newname in mapper.items():
                # Only rename if name has changed
                if newname != oldname:
                    try:
                        i = self._columns.index(oldname)
                        self._columns[i] = newname
                        self.data[newname] = self.data[oldname]
                        del self.data[oldname]
                    except:
                        pass
        else:
            raise TypeError("Argument must be callable or dict-like")
        return self

    def append(self, other, ordered=True):
        # TODO - remove error and define function once akutil/util.py moved to arkouda
        raise NotImplementedError("Append functionality is not yet available for dataframes in arkouda. For updates, please visit https://github.com/Bears-R-Us/arkouda/issues/1126")

    @classmethod
    def concat(cls, items, ordered=True):
        # TODO - remove error and define function once akutil/util.py moved to arkouda
        raise NotImplementedError("Append functionality is not yet available for dataframes in arkouda. For updates, please visit https://github.com/Bears-R-Us/arkouda/issues/1126")

    def head(self, n=5):
        """
        Return the first `n` rows.

        This function returns the first `n` rows of the the dataframe. It is
        useful for quickly verifying data, for example, after sorting or
        appending rows.

        Parameters
        ----------
        n : int
            Number of rows to select.

        Returns
        -------
        akutil.DataFrame
            The first `n` rows of the DataFrame.

        See Also
        --------
        tail
        """

        return self[:n]

    def tail(self, n=5):
        """
        Return the last `n` rows.

        This function returns the last `n` rows for the dataframe. It is
        useful for quickly testing if your object has the right type of data in
        it.

        Parameters
        ----------
        n : int (default=5)
            Number of rows to select.

        Returns
        -------
        akutil.DataFrame
            The last `n` rows of the DataFrame.

        See Also
        --------
        akutil.dataframe.head
        """

        self.update_size()
        if self._size <= n:
            return self
        return self[self._size - n:]

    def sample(self, n=5):
        """
        Return a random sample of `n` rows.

        Parameters
        ----------
        n : int (default=5)
            Number of rows to return.

        Returns
        -------
        akutil.DataFrame
            The sampled `n` rows of the DataFrame.
        """
        self.update_size()
        if self._size <= n:
            return self
        return self[array(random.sample(range(self._size), n))]

    def GroupBy(self, keys, use_series=False):
        """
        Group the dataframe by a column or a list of columns.

        Parameters
        ----------
        keys : string or list
            An (ordered) list of column names or a single string to group by.
        use_series : If True, returns an akutil.GroupBy oject. Otherwise an arkouda GroupBy object

        Returns
        -------
        GroupBy
            Either an akutil GroupBy or an arkouda GroupBy object.

        See Also
        --------
        arkouda.GroupBy
        """

        self.update_size()
        if isinstance(keys, str):
            cols = self.data[keys]
        elif not isinstance(keys, list):
            raise TypeError("keys must be a colum name or a list of column names")
        elif len(keys) == 1:
            cols = self.data[keys[0]]
        else:
            cols = [self.data[col] for col in keys]
        gb = GroupBy(cols)
        if use_series:
            # TODO - remove the error and implement once series is configured
            #	gb = GroupBy(gb, self)
            raise NotImplementedError("akutil GroupBy functionality using series has not yet been implemented in "
                                      "Arkouda. For updates, please visit https://github.com/Bears-R-Us/arkouda/issues/1128")
        return gb

    def memory_usage(self, unit='GB'):
        """
        Print the size of this DataFrame.

        Parameters
        ----------
        unit : str
            Unit to return. One of {'KB', 'MB', 'GB'}.

        Returns
        -------
        int
            The number of bytes used by this DataFrame in [unit]s.
        """

        KB = 1024
        MB = KB * KB
        GB = MB * KB
        self._bytes = 0
        for key, val in self.items():
            if isinstance(val, pdarray):
                self._bytes += (val.dtype).itemsize * val.size
            elif isinstance(val, Strings):
                self._bytes += val.nbytes
        if unit == 'B':
            return "{:} B".format(int(self._bytes))
        elif unit == 'MB':
            return "{:} MB".format(int(self._bytes / MB))
        elif unit == 'KB':
            return "{:} KB".format(int(self._bytes / KB))
        return "{:.2f} GB".format(self._bytes / GB)

    def to_pandas(self, datalimit=maxTransferBytes, retain_index=False):
        """
        Send this DataFrame to a pandas DataFrame.

        Parameters
        ----------
        datalimit : int (default=arkouda.client.maxTransferBytes)
            The maximum number size, in megabytes to transfer. The requested
            DataFrame will be converted to a pandas DataFrame only if the
            estimated size of the DataFrame does not exceed this value.

        retain_index : book (default=False)
            Normally, to_pandas() creates a new range index object. If you want
            to keep the index column, set this to True.

        Returns
        -------
        pandas.DataFrame
            The result of converting this DataFrame to a pandas DataFrame.
        """

        self.update_size()

        # Estimate how much memory would be required for this DataFrame
        nbytes = 0
        for key, val in self.items():
            if isinstance(val, pdarray):
                nbytes += (val.dtype).itemsize * self._size
            elif isinstance(val, Strings):
                nbytes += val.nbytes

        KB = 1024
        MB = KB * KB
        GB = MB * KB

        # Get units that make the most sense.
        msg = ""
        if nbytes < KB:
            msg = "{:,} B".format(nbytes)
        elif nbytes < MB:
            msg = "{:,} KB".format(int(nbytes / KB))
        elif nbytes < GB:
            msg = "{:,} MB".format(int(nbytes / MB))
            print(f"This transfer will use " + msg + ".")
        else:
            msg = "{:,} GB".format(int(nbytes / GB))
            print(f"This will transfer " + msg + " from arkouda to pandas.")
        # If the total memory transfer requires more than `datalimit` per
        # column, we will warn the user and return.
        if nbytes > (datalimit * len(self._columns) * MB):
            msg = f"This operation would transfer more than {datalimit} bytes."
            warn(msg, UserWarning)
            return None

        # Proceed with conversion if possible, ignore index column
        pandas_data = {}
        for key in self._columns[1:]:
            val = self[key]
            try:
                pandas_data[key] = val.to_ndarray()
            except TypeError as e:
                raise IndexError("Bad index type or format.")

        # Return a new dataframe with original indices if requested.
        if retain_index and 'index' in self:
            index = self.data['index'].to_ndarray()
            return pd.DataFrame(data=pandas_data, index=index)

        else:
            return pd.DataFrame(data=pandas_data)

    def save(self, path, index=False):
        """
        Save DataFrame to disk, preserving column names.

        Parameters
        ----------
        path : str
            File path to save data
        index : bool
            If True, save the index column. By default, do not save the index.

        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.
        """
        tosave = {k: v for k, v in self.data.items() if (index or k != "index")}
        save_all(tosave, path)

    def argsort(self, key, ascending=True):
        """
        Return the permutation that sorts the dataframe by `key`.

        Parameters
        ----------
        key : str
            The key to sort on.

        Returns
        -------
        ak.pdarray
            The permutation array that sorts the data on `key`.
        """

        if self._empty:
            return array([], dtype=akint64)
        if ascending:
            return argsort(self[key])
        else:
            if isinstance(self[key], pdarray) and self[key].dtype in (akint64, akfloat64):
                return argsort(-self[key])
            else:
                return argsort(self[key])[arange(self.size - 1, -1, -1)]

    def coargsort(self, keys, ascending=True):
        """
        Return the permutation that sorts the dataframe by `keys`.

        Sorting using Strings may not yield correct results

        Parameters
        ----------
        keys : list
            The keys to sort on.

        Returns
        -------
        ak.pdarray
            The permutation array that sorts the data on `keys`.
        """

        if self._empty:
            return array([], dtype=akint64)
        arrays = []
        for key in keys:
            arrays.append(self[key])
        i = coargsort(arrays)
        if not ascending:
            i = i[arange(self.size - 1, -1, -1)]
        return i

    def sort_values(self, by=None, ascending=True):
        """
        Sort the DataFrame by one or more columns.

        If no column is specified, all columns are used.

        Note: Fails on sorting ak.Strings when multiple columns being sorted

        Parameters
        ----------
        by : str or list/tuple of str
            The name(s) of the column(s) to sort by.
        ascending : bool
            Sort values in ascending (default) or descending order.

        See Also
        --------
        apply_permutation, sorted
        """

        if self._empty:
            return array([], dtype=akint64)
        if by is None:
            if len(self._columns) == 2:
                i = self.argsort(self._columns[1], ascending=ascending)
            else:
                i = self.coargsort(self._columns[1:], ascending=ascending)
        elif isinstance(by, str):
            i = self.argsort(by, ascending=ascending)
        elif isinstance(by, (list, tuple)):
            i = self.coargsort(by, ascending=ascending)
        else:
            raise TypeError("Column name(s) must be str or list/tuple of str")
        return self[i]

    def apply_permutation(self, perm):
        """
        Apply a permutation to an entire DataFrame.

        This may be useful if you want to unsort an DataFrame, or even to
        apply an arbitrary permutation such as the inverse of a sorting
        permutation.

        Parameters
        ----------
        perm : ak.pdarray
            A permutation array. Should be the same size as the data
            arrays, and should consist of the integers [0,size-1] in
            some order. Very minimal testing is done to ensure this
            is a permutation.

        See Also
        --------
        sort
        """

        if (perm.min() != 0) or (perm.max() != perm.size - 1):
            raise ValueError("The indicated permutation is invalid.")
        if unique(perm).size != perm.size:
            raise ValueError("The indicated permutation is invalid.")
        for key, val in self.data.items():
            self[key] = self[key][perm]

    def filter_by_range(self, keys, low=1, high=None):
        """
        Find all rows where the value count of the items in a given set of
        columns (keys) is within the range [low, high].

        To filter by a specific value, set low == high.

        Parameters
        ----------
        keys : list or str
            The names of the columns to group by
        low : int (default=1)
            The lowest value count.
        high : int (default=None)
            The highest value count, default to unlimited.

        Returns
        -------
        pdarray
            An array of boolean values for qualified rows in this DataFrame.

        See Also
        --------
        filter_by_count
        """

        if isinstance(keys, str):
            keys = [keys]
        gb = self.GroupBy(keys, use_series=False)
        vals, cts = gb.count()
        if not high:
            positions = where(cts >= low, 1, 0)
        else:
            positions = where(((cts >= low) & (cts <= high)), 1, 0)

        broadcast = gb.broadcast(positions, permute=False)
        broadcast = (broadcast == 1)
        return broadcast[invert_permutation(gb.permutation)]

    def copy(self, deep=True):
        """
        Make a copy of this object's data.

        When `deep = True` (default), a new object will be created with a copy of
        the calling object's data. Modifications to the data of the copy will not
        be reflected in the original object.


        When `deep = False` a new object will be created without copying the
        calling object's data. Any changes to the data of the original object will
        be reflected in the shallow copy, and vice versa.

        Parameters
        ----------
        deep : bool (default=True)
            When True, return a deep copy. Otherwise, return a shallow copy.

        Returns
        -------
        aku.DataFrame
            A deep or shallow copy according to caller specification.
        """

        if deep:
            res = DataFrame()
            res._size = self._size
            res._bytes = self._bytes
            res._empty = self._empty
            res._columns = self._columns

            for key, val in self.items():
                res[key] = val[:]

            return res
        else:
            return DataFrame(self)

    def groupby(self, keys, use_series=True):
        """Group the dataframe by a column or a list of columns.  Alias for GroupBy

        Parameters
        ----------
        keys : a single column name or a list of column names
        use_series : Change return type to Arkouda Groupby object.

        Returns
        -------
        An arkouda Groupby instance
        """

        return self.GroupBy(keys, use_series)


def sorted(df, column=False):
    """
    Analogous to other python 'sorted(obj)' functions in that it returns
    a sorted copy of the DataFrame.

    If no sort key is specified, sort by the first key returned.

    Note: This fails on sorting ak.Strings, as does DataFrame.sort().

    Parameters
    ----------
    df : akutil.dataframe.DataFrame
        The DataFrame to sort.

    column : str
        The name of the column to sort by.

    Returns
    -------
    akutil.dataframe.DataFrame
        A sorted copy of the original DataFrame.
    """

    if not isinstance(df, DataFrame):
        raise TypeError("The sorted operation requires an DataFrame.")
    result = DataFrame(df.data)
    result.sort(column)
    return result


def intx(a, b):
    # TODO - define function once akutil/alignment.py is moved into arkouda
    raise NotImplementedError("intx functionality from akutil is not yet available in arkouda. For updates, visit https://github.com/Bears-R-Us/arkouda/issues/1127")


def intersect(a, b, positions=True, unique=False):
    """
    Find the intersection of two arkouda arrays.

    This function can be especially useful when `positions=True` so
    that the caller gets the indices of values present in both arrays.

    Parameters
    ----------
    a : ak.Strings or ak.pdarray
        An array of strings

    b : ak.Strings or ak.pdarray
        An array of strings

    positions : bool (default=True)
        Return tuple of boolean pdarrays that indicate positions in a and b
        where the values are in the intersection.

    unique : bool (default=False)
        If the number of distinct values in `a` (and `b`) is equal to the size of
        `a` (and `b`), there is a more efficient method to compute the intersection.

    Returns
    -------
    (ak.pdarray, ak.pdarray)
        The indices of `a` and `b` where any element occurs at least once in both
        arrays.
    """

    # To ensure compatibility with all types of arrays:
    if (isinstance(a, pdarray) and isinstance(b, pdarray)):
        intx = intersect1d(a, b)
        if not positions:
            return intx
        else:
            maska = in1d(a, intx)
            maskb = in1d(b, intx)
            return (maska, maskb)

    # It takes more effort to do this with ak.Strings arrays.
    elif (isinstance(a, Strings) and isinstance(b, Strings)):

        # Hash the two arrays first
        hash_a00, hash_a01 = a.hash()
        hash_b00, hash_b01 = b.hash()

        # a and b do not have duplicate entries, so the hashes are distinct
        if unique:
            hash0 = concatenate([hash_a00, hash_b00])
            hash1 = concatenate([hash_a01, hash_b01])

            # Group by the unique hashes
            gb = GroupBy([hash0, hash1])
            val, cnt = gb.count()

            # Hash counts, in groupby order
            counts = gb.broadcast(cnt, permute=False)

            # Same, in original order
            tmp = counts[:]
            counts[gb.permutation] = tmp
            del tmp

            # Masks
            maska = (counts > 1)[:a.size]
            maskb = (counts > 1)[a.size:]

            # The intersection for each array of hash values
            if positions:
                return (maska, maskb)
            else:
                return a[maska]

        # a and b may have duplicate entries, so get the unique hash values
        else:
            gba = GroupBy([hash_a00, hash_a01])
            gbb = GroupBy([hash_b00, hash_b01])

            # Take the unique keys as the hash we'll work with
            a0, a1 = gba.unique_keys
            b0, b1 = gbb.unique_keys
            hash0 = concatenate([a0, b0])
            hash1 = concatenate([a1, b1])

            # Group by the unique hashes
            gb = GroupBy([hash0, hash1])
            val, cnt = gb.count()

            # Hash counts, in groupby order
            counts = gb.broadcast(cnt, permute=False)

            # Restore the original order
            tmp = counts[:]
            counts[gb.permutation] = tmp
            del tmp

            # Broadcast back up one more level
            countsa = counts[:a0.size]
            countsb = counts[a0.size:]
            counts2a = gba.broadcast(countsa, permute=False)
            counts2b = gbb.broadcast(countsb, permute=False)

            # Restore the original orders
            tmp = counts2a[:]
            counts2a[gba.permutation] = tmp
            del tmp
            tmp = counts2b[:]
            counts2b[gbb.permutation] = tmp
            del tmp

            # Masks
            maska = (counts2a > 1)
            maskb = (counts2b > 1)

            # The intersection for each array of hash values
            if positions:
                return (maska, maskb)
            else:
                return a[maska]


def invert_permutation(perm):
    """
    Find the inverse of a permutation array.

    Parameters
    ----------
    perm : ak.pdarray
        The permutation array.

    Returns
    -------
    ak.pdarray
        The inverse of the permutation array.
    """

    # Test if the array is actually a permutation
    rng = perm.max() - perm.min()
    if (unique(perm).size != perm.size) and (perm.size != rng + 1):
        raise ValueError("The array is not a permutation.")
    return coargsort([perm, arange(0, perm.size)])
