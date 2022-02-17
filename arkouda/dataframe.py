from __future__ import annotations

from datetime import datetime
from ipaddress import ip_address
from collections import UserDict
from tabulate import tabulate
from warnings import warn
import akutil as aku
import pandas as pd
import numpy as np
import random

from arkouda.segarray import SegArray
from arkouda.pdarrayclass import pdarray
from arkouda.categorical import Categorical
from arkouda.strings import Strings
from arkouda.pdarraycreation import arange, array
from arkouda.groupbyclass import GroupBy
from arkouda.pdarraysetops import concatenate

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
        for name in [ 'all','any','argmax','argmin','max','mean','min','nunique','prod','sum','OR', 'AND', 'XOR' ] :
            setattr(cls,name, cls._make_aggop(name))
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
    >>> import akutil as aku
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = aku.DataFrame()
    >>> df['a'] = ak.array([1,2,3])

    Create a new DataFrame using a dictionary of data:

    >>> userName = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    >>> userID = ak.array([111, 222, 111, 333, 222, 111])
    >>> item = ak.array([0, 0, 1, 1, 2, 0])
    >>> day = ak.array([5, 5, 6, 5, 6, 6])
    >>> amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    >>> df = aku.DataFrame({'userName': userName, 'userID': userID,
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
            return aku.Row(result)

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
                for k,v in value.items():
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
                UserDict.__setitem__(self,key,value)
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
        elif self._bytes < 1024**2:
            mem = self.memory_usage(unit='KB')
        elif self._bytes < 1024**3:
            mem = self.memory_usage(unit='MB')
        else:
            mem = self.memory_usage(unit='GB')
        rows = " rows"
        if self._size == 1:
            rows = " row"
        return 'DataFrame(['+keystr+'], {:,}'.format(self._size)+rows+', '+str(mem)+')'

    def _get_head_tail(self):
        if self._empty:
            return pd.DataFrame()
        self.update_size()
        maxrows = pd.get_option('display.max_rows')
        if self._size <= maxrows:
            newdf = aku.DataFrame()
            for col in self._columns:
                if isinstance(self[col], Categorical):
                    newdf[col] = self[col].categories[self[col].codes]
                else:
                    newdf[col] = self[col]
            return newdf.to_pandas(retain_index=True)
        # Being 1 above the threshold caises the PANDAS formatter to split the data frame vertically
        idx = array(list(range(maxrows//2+1)) + list(range(self._size - (maxrows//2), self._size)))
        newdf = aku.DataFrame()
        for col in self._columns[1:]:
            if isinstance(self[col], Categorical):
                newdf[col] = self[col].categories[self[col].codes[idx]]
            else:
                newdf[col] = self[col][idx]
        newdf['index'] = self['index'][idx]
        return newdf.to_pandas(retain_index=True)

    def _shape_str(self):
        return "{} rows x {} columns".format(self.size,self._ncols() )

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
        res = aku.Row({key: dtype for key, dtype in zip(keys, dtypes)})
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