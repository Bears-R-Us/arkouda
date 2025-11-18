# flake8: noqa
# mypy: ignore-errors
from _typeshed import Incomplete


from collections import UserDict


class DataFrame(UserDict):
    r'''

    A DataFrame structure based on arkouda arrays.

    Parameters
    ----------
    initialdata : List or dictionary of lists, tuples, or pdarrays
        Each list/dictionary entry corresponds to one column of the data and
        should be a homogenous type. Different columns may have different
        types. If using a dictionary, keys should be strings.

    index : Index, pdarray, or Strings
        Index for the resulting frame. Defaults to an integer range.

    columns : List, tuple, pdarray, or Strings
        Column labels to use if the data does not include them. Elements must
        be strings. Defaults to an stringified integer range.

    Examples
    --------
    >>> import arkouda as ak

    Create an empty DataFrame and add a column of data:
    >>> import arkouda as ak
    >>> df = ak.DataFrame()
    >>> df['a'] = ak.array([1,2,3])
    >>> df
       a
    0  1
    1  2
    2  3 (3 rows x 1 columns)

    Create a new DataFrame using a dictionary of data:

    >>> userName = ak.array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])
    >>> userID = ak.array([111, 222, 111, 333, 222, 111])
    >>> item = ak.array([0, 0, 1, 1, 2, 0])
    >>> day = ak.array([5, 5, 6, 5, 6, 6])
    >>> amount = ak.array([0.5, 0.6, 1.1, 1.2, 4.3, 0.6])
    >>> df = ak.DataFrame({
    ...     'userName': userName,
    ...     'userID': userID,
    ...     'item': item,
    ...     'day': day,
    ...     'amount': amount
    ... })
    >>> df
      userName  userID  item  day  amount
    0    Alice     111     0    5     0.5
    1      Bob     222     0    5     0.6
    2    Alice     111     1    6     1.1
    3    Carol     333     1    5     1.2
    4      Bob     222     2    6     4.3
    5    Alice     111     0    6     0.6 (6 rows x 5 columns)

    Indexing works slightly differently than with pandas:
    >>> df[0]
    {'userName': np.str_('Alice'), 'userID': np.int64(111), 'item': np.int64(0),
    'day': np.int64(5), 'amount': np.float64(0.5)}
    >>> df['userID']
    array([111 222 111 333 222 111])

    >>> df['userName']
    array(['Alice', 'Bob', 'Alice', 'Carol', 'Bob', 'Alice'])

    >>> df[ak.array([1,3,5])]
      userName  userID  item  day  amount
    1      Bob     222     0    5     0.6
    3    Carol     333     1    5     1.2
    5    Alice     111     0    6     0.6 (3 rows x 5 columns)

    Compute the stride:
    >>> df[1:5:1]
      userName  userID  item  day  amount
    1      Bob     222     0    5     0.6
    2    Alice     111     1    6     1.1
    3    Carol     333     1    5     1.2
    4      Bob     222     2    6     4.3 (4 rows x 5 columns)

    >>> df[ak.array([1,2,3])]
      userName  userID  item  day  amount
    1      Bob     222     0    5     0.6
    2    Alice     111     1    6     1.1
    3    Carol     333     1    5     1.2 (3 rows x 5 columns)

    >>> df[['userID', 'day']]
       userID  day
    0     111    5
    1     222    5
    2     111    6
    3     333    5
    4     222    6
    5     111    6 (6 rows x 2 columns)


    '''
    ...

    def GroupBy(self, keys, use_series=False, as_index=True, dropna=True) -> 'Union[DataFrameGroupBy, GroupBy_class]':
        r'''

        Group the dataframe by a column or a list of columns.

        Parameters
        ----------
        keys : str or list of str
            An (ordered) list of column names or a single string to group by.
        use_series : bool, default=False
            If True, returns an arkouda.pandas.dataframe.DataFrameGroupBy object.
            Otherwise an arkouda.pandas.groupbyclass.GroupBy object.
        as_index: bool, default=True
            If True, groupby columns will be set as index
            otherwise, the groupby columns will be treated as DataFrame columns.
        dropna : bool, default=True
            If True, and the groupby keys contain NaN values,
            the NaN values together with the corresponding row will be dropped.
            Otherwise, the rows corresponding to NaN values will be kept.

        Returns
        -------
        arkouda.pandas.dataframe.DataFrameGroupBy or arkouda.pandas.groupbyclass.GroupBy
            If use_series = True, returns an arkouda.pandas.dataframe.DataFrameGroupBy object.
            Otherwise returns an arkouda.pandas.groupbyclass.GroupBy object.

        See Also
        --------
        arkouda.GroupBy

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1.0, 1.0, 2.0, np.nan], 'col2': [4, 5, 6, 7]})
        >>> df
           col1  col2
        0   1.0     4
        1   1.0     5
        2   2.0     6
        3   NaN     7 (4 rows x 2 columns)

        >>> df.GroupBy("col1") # doctest: +SKIP
        <arkouda.groupbyclass.GroupBy object at 0x7dbc23462510>
        >>> df.GroupBy("col1").size()
        (array([1.00000000000000000 2.00000000000000000]), array([2 1]))

        >>> df.GroupBy("col1",use_series=True).size()
        col1
        1.0    2
        2.0    1
        dtype: int64
        >>> df.GroupBy("col1",use_series=True, as_index = False).size()
           col1  size
        0   1.0     2
        1   2.0     1 (2 rows x 2 columns)


        '''
        ...

    def GroupBy_class(self, keys: 'Optional[groupable]' = None, assume_sorted: 'bool' = False, dropna: 'bool' = True, **kwargs):
        r'''

        Group an array or list of arrays by value.

        Usually in preparation
        for aggregating the within-group values of another array.

        Parameters
        ----------
        keys : (list of) pdarray, Strings, or Categorical
            The array to group by value, or if list, the column arrays to group by row
        assume_sorted : bool
            If True, assume keys is already sorted (Default: False)

        Attributes
        ----------
        nkeys : int
            The number of key arrays (columns)
        permutation : pdarray
            The permutation that sorts the keys array(s) by value (row)
        unique_keys : pdarray, Strings, or Categorical
            The unique values of the keys array(s), in grouped order
        ngroups : int_scalars
            The length of the unique_keys array(s), i.e. number of groups
        segments : pdarray
            The start index of each group in the grouped array(s)
        logger : ArkoudaLogger
            Used for all logging operations
        dropna : bool (default=True)
            If True, and the groupby keys contain NaN values,
            the NaN values together with the corresponding row will be dropped.
            Otherwise, the rows corresponding to NaN values will be kept.
            The default is True

        Raises
        ------
        TypeError
            Raised if keys is a pdarray with a dtype other than int64

        Notes
        -----
        Integral pdarrays, Strings, and Categoricals are natively supported, but
        float64 and bool arrays are not.

        For a user-defined class to be groupable, it must inherit from pdarray
        and define or overload the grouping API:
          1) a ._get_grouping_keys() method that returns a list of pdarrays
             that can be (co)argsorted.
          2) (Optional) a .group() method that returns the permutation that
             groups the array
        If the input is a single array with a .group() method defined, method 2
        will be used; otherwise, method 1 will be used.


        '''
        ...

    def _drop_column(self, keys):
        r'''

        Drop a column or columns from the dataframe, in-place.

        keys : list
            The labels to be dropped on the given axis

        '''
        ...

    def _drop_row(self, keys):
        r'''

        Drop a row or rows from the dataframe, in-place.

        keys : list
            The indexes to be dropped on the given axis

        '''
        ...

    def _get_head_tail(self):

        ...

    def _get_head_tail_server(self):

        ...

    def _ipython_key_completions_(self):

        ...

    def _ncols(self):
        r'''

        Return number of columns.

        If index appears, we now want to utilize this
        because the actual index has been moved to a property.


        '''
        ...

    def _parse_col_name(self, entryName, dfName):
        r'''

        Parse the registered name of the data component and pull out the column type and column name.

        Helper method used by from_return_msg.

        Parameters
        ----------
        entryName : string
            The full registered name of the data component

        dfName : string
            The name of the DataFrame

        Returns
        -------
        tuple
            (columnName, columnType)


        '''
        ...

    def _prep_data(self, index=False, columns=None):

        ...

    def _reindex(self, idx):

        ...

    def _rename_column(self, mapper: 'Union[Callable, Dict]', inplace: 'bool' = False) -> 'Optional[DataFrame]':
        r'''

        Rename columns within the dataframe.

        Parameters
        ----------
        mapper : callable or dict-like
            Function or dictionary mapping existing columns to new columns.
            Nonexistent names will not raise an error.
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.

        Returns
        -------
        DataFrame or None
            DateFrame when `inplace=False`
            None when `inplace=True`

        See Also
        --------
        ak.DataFrame._rename_index
        ak.DataFrame.rename


        '''
        ...

    def _rename_index(self, mapper: 'Union[Callable, Dict]', inplace: 'bool' = False) -> 'Optional[DataFrame]':
        r'''

        Rename indexes within the dataframe.

        Parameters
        ----------
        mapper : callable or dict-like
            Function or dictionary mapping existing indexes to new indexes.
            Nonexistent names will not raise an error.
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.

        Returns
        -------
        DataFrame or None
            DateFrame when `inplace=False`
            None when `inplace=True`

        See Also
        --------
            ak.DataFrame._rename_column
            ak.DataFrame.rename

        Notes
        -----
            This does not function exactly like pandas. The replacement value here must be
            the same type as the existing value.


        '''
        ...

    def _repr_html_(self):
        r'''
        Return html-formatted version of the dataframe.
        '''
        ...

    def _set_index(self, value):

        ...

    def _shape_str(self):

        ...

    def _to_hdf_snapshot(self, path, dataset='DataFrame', mode='truncate', file_type='distribute'):
        r'''

        Save a dataframe as a group with columns within the group.

        This allows saving other
        datasets in the HDF5 file without impacting the integrity of the dataframe.
        This is only used for the snapshot workflow.

        Parameters
        ----------
        path : str
            File path to save data
        dataset: str
            Name to save the dataframe under within the file
            Only used when as_dataset=True
        mode: str (truncate | append), default=truncate
            Indicates whether the dataset should truncate the file and write or append
            to the file
            Only used when as_dataset=True
        file_type: str (single | distribute), default=distribute
            Whether to save to a single file or distribute across Locales
            Only used when as_dataset=True

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray


        '''
        ...

    def all(self, axis=0) -> 'Union[Series, bool]':
        r'''

        Return whether all elements are True, potentially over an axis.

        Returns True unless there at least one element along a Dataframe axis that is False.

        Currently, will ignore any columns that are not type bool.
        This is equivalent to the pandas option bool_only=True.

        Parameters
        ----------
        axis: {0 or ‘index’, 1 or ‘columns’, None}, default = 0

            Indicate which axis or axes should be reduced.

            0 / ‘index’ : reduce the index, return a Series whose index is the original column labels.

            1 / ‘columns’ : reduce the columns, return a Series whose index is the original index.

            None : reduce all axes, return a scalar.

        Returns
        -------
        arkouda.pandas.series.Series or bool

        Raises
        ------
        ValueError
            Raised if axis does not have a value in {0 or ‘index’, 1 or ‘columns’, None}.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"A":[True,True,True,False],"B":[True,True,True,False],
        ...          "C":[True,False,True,False],"D":[True,True,True,True]})
        >>> df
               A      B      C     D
        0   True   True   True  True
        1   True   True  False  True
        2   True   True   True  True
        3  False  False  False  True (4 rows x 4 columns)

        >>> df.all(axis=0)
        A    False
        B    False
        C    False
        D     True
        dtype: bool
        >>> df.all(axis=1)
        0     True
        1    False
        2     True
        3    False
        dtype: bool
        >>> df.all(axis=None)
        False


        '''
        ...

    def any(self, axis=0) -> 'Union[Series, bool]':
        r'''

        Return whether any element is True, potentially over an axis.

        Returns False unless there is at least one element along a Dataframe axis that is True.

        Currently, will ignore any columns that are not type bool.
        This is equivalent to the pandas option bool_only=True.

        Parameters
        ----------
        axis: {0 or ‘index’, 1 or ‘columns’, None}, default = 0

            Indicate which axis or axes should be reduced.

            0 / ‘index’ : reduce the index, return a Series whose index is the original column labels.

            1 / ‘columns’ : reduce the columns, return a Series whose index is the original index.

            None : reduce all axes, return a scalar.

        Returns
        -------
        arkouda.pandas.series.Series or bool

        Raises
        ------
        ValueError
            Raised if axis does not have a value in {0 or ‘index’, 1 or ‘columns’, None}.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"A":[True,True,True,False],"B":[True,True,True,False],
        ...          "C":[True,False,True,False],"D":[False,False,False,False]})
        >>> df
               A      B      C      D
        0   True   True   True  False
        1   True   True  False  False
        2   True   True   True  False
        3  False  False  False  False (4 rows x 4 columns)

        >>> df.any(axis=0)
        A     True
        B     True
        C     True
        D    False
        dtype: bool
        >>> df.any(axis=1)
        0     True
        1     True
        2     True
        3    False
        dtype: bool
        >>> df.any(axis=None)
        True


        '''
        ...

    def append(self, other, ordered=True):
        r'''

        Concatenate data from 'other' onto the end of this DataFrame, in place.

        Explicitly, use the arkouda concatenate function to append the data
        from each column in other to the end of self. This operation is done
        in place, in the sense that the underlying pdarrays are updated from
        the result of the arkouda concatenate function, rather than returning
        a new DataFrame object containing the result.

        Parameters
        ----------
        other : DataFrame
            The DataFrame object whose data will be appended to this DataFrame.
        ordered: bool, default=True
            If False, allow rows to be interleaved for better performance (but
            data within a row remains together). By default, append all rows
            to the end, in input order.

        Returns
        -------
        self
            Appending occurs in-place, but result is returned for compatibility.

        Examples
        --------
        >>> import arkouda as ak

        >>> df1 = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df1
            col1  col2
        0     1     3
        1     2     4 (2 rows x 2 columns)

        >>> df2 = ak.DataFrame({'col1': [3], 'col2': [5]})
        >>> df2
            col1  col2
        0     3     5 (1 rows x 2 columns)

        >>> df1.append(df2)
            col1  col2
        0     1     3
        1     2     4
        2     3     5 (3 rows x 2 columns)

        >>> df1
            col1  col2
        0     1     3
        1     2     4
        2     3     5 (3 rows x 2 columns)


        '''
        ...

    def apply_permutation(self, perm):
        r'''

        Apply a permutation to an entire DataFrame.

        The operation is done in place and the original DataFrame will be modified.

        This may be useful if you want to unsort an DataFrame, or even to
        apply an arbitrary permutation such as the inverse of a sorting
        permutation.

        Parameters
        ----------
        perm : pdarray
            A permutation array. Should be the same size as the data
            arrays, and should consist of the integers [0,size-1] in
            some order. Very minimal testing is done to ensure this
            is a permutation.

        See Also
        --------
        sort

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df
           col1  col2
        0     1     4
        1     2     5
        2     3     6 (3 rows x 2 columns)

        >>> perm_arry = ak.array([0, 2, 1])
        >>> df.apply_permutation(perm_arry)
        >>> df
           col1  col2
        0     1     4
        2     3     6
        1     2     5 (3 rows x 2 columns)


        '''
        ...

    def argsort(self, key, ascending=True):
        r'''

        Return the permutation that sorts the dataframe by `key`.

        Parameters
        ----------
        key : str
            The key to sort on.
        ascending : bool, default = True
            If true, sort the key in ascending order.
            Otherwise, sort the key in descending order.

        Returns
        -------
        arkouda.numpy.pdarrayclass.pdarray
            The permutation array that sorts the data on `key`.

        See Also
        --------
        coargsort

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1.1, 3.1, 2.1], 'col2': [6, 5, 4]})
        >>> df
           col1  col2
        0   1.1     6
        1   3.1     5
        2   2.1     4 (3 rows x 2 columns)

        >>> df.argsort('col1')
        array([0 2 1])
        >>> sorted_df1 = df[df.argsort('col1')]
        >>> sorted_df1
           col1  col2
        0   1.1     6
        2   2.1     4
        1   3.1     5 (3 rows x 2 columns)

        >>> df.argsort('col2')
        array([2 1 0])
        >>> sorted_df2 = df[df.argsort('col2')]
        >>> sorted_df2
           col1  col2
        2   2.1     4
        1   3.1     5
        0   1.1     6 (3 rows x 2 columns)


        '''
        ...

    def assign(self, **kwargs) -> 'DataFrame':
        r'''

        Assign new columns to a DataFrame.

        Return a new object with all original columns in addition to new ones.
        Existing columns that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs : dict of {str: callable or Series}
            The column names are keywords. If the values are
            callable, they are computed on the DataFrame and
            assigned to the new columns. The callable must not
            change input DataFrame (though pandas doesn't check it).
            If the values are not callable, (e.g. a Series, scalar, or array),
            they are simply assigned.

        Returns
        -------
        DataFrame
            A new DataFrame with the new columns in addition to
            all the existing columns.

        Notes
        -----
        Assigning multiple columns within the same ``assign`` is possible.
        Later items in '\*\*kwargs' may refer to newly created or modified
        columns in 'df'; items are computed and assigned into 'df' in order.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'temp_c': [17.0, 25.0]},
        ...                   index=['Portland', 'Berkeley'])
        >>> df
                  temp_c
        Portland    17.0
        Berkeley    25.0 (2 rows x 1 columns)

        Where the value is a callable, evaluated on `df`:
        >>> df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0 (2 rows x 2 columns)

        Alternatively, the same behavior can be achieved by directly
        referencing an existing Series or sequence:

        >>> df.assign(temp_f=df['temp_c'] * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0 (2 rows x 2 columns)

        You can create multiple columns within the same assign where one
        of the columns depends on another one defined within the same assign:

        >>> df.assign(temp_f=lambda x: x['temp_c'] * 9 / 5 + 32,
        ...           temp_k=lambda x: (x['temp_f'] + 459.67) * 5 / 9)
                  temp_c  temp_f  temp_k
        Portland    17.0    62.6  290.15
        Berkeley    25.0    77.0  298.15 (2 rows x 3 columns)


        '''
        ...

    def coargsort(self, keys, ascending=True):
        r'''

        Return the permutation that sorts the dataframe by `keys`.

        Note: Sorting using Strings may not yield correct sort order.

        Parameters
        ----------
        keys : list of str
            The keys to sort on.

        Returns
        -------
        arkouda.numpy.pdarrayclass.pdarray
            The permutation array that sorts the data on `keys`.

        Example
        -------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [2, 2, 1], 'col2': [3, 4, 3], 'col3':[5, 6, 7]})
        >>> df
           col1  col2  col3
        0     2     3     5
        1     2     4     6
        2     1     3     7 (3 rows x 3 columns)

        >>> df.coargsort(['col1', 'col2'])
        array([2 0 1])
        >>>


        '''
        ...

    @property
    def columns(self, keys, ascending=True):
        r'''

        An Index where the values are the column names of the dataframe.

        Returns
        -------
        arkouda.index.Index
            The values of the index are the column names of the dataframe.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4 (2 rows x 2 columns)

        >>> df.columns
        Index(['col1', 'col2'], dtype='<U0')


        '''
        ...

    def concat(self, items, ordered=True):
        r'''
        Essentially an append, but different formatting.
        '''
        ...

    def corr(self) -> 'DataFrame':
        r'''

        Return new DataFrame with pairwise correlation of columns.

        Returns
        -------
        DataFrame
            Arkouda DataFrame containing correlation matrix of all columns.

        Raises
        ------
        RuntimeError
            Raised if there's a server-side error thrown.

        See Also
        --------
        pdarray.corr

        Notes
        -----
        Generate the correlation matrix using Pearson R for all columns.

        Attempts to convert to numeric values where possible for inclusion in the matrix.

        Example
        -------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [-1, -2]})
        >>> df
           col1  col2
        0     1    -1
        1     2    -2 (2 rows x 2 columns)

        >>> corr = df.corr()
        >>> corr
              col1  col2
        col1   1.0  -1.0
        col2  -1.0   1.0 (2 rows x 2 columns)


        '''
        ...

    def count(self, axis: 'Union[int, str]' = 0, numeric_only=False) -> 'Series':
        r'''

        Count non-NA cells for each column or row.

        The values np.NaN are considered NA.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            If 0 or ‘index’ counts are generated for each column.
            If 1 or ‘columns’ counts are generated for each row.

        numeric_only: bool = False
            Include only float, int or boolean data.

        Returns
        -------
        Series
            For each column/row the number of non-NA/null entries.

        Raises
        ------
        ValueError
            Raised if axis is not 0, 1, 'index', or 'columns'.

        See Also
        --------
        GroupBy.count()

        Examples
        --------
        >>> import arkouda as ak
        >>> import numpy as np
        >>> df = ak.DataFrame({'col_A': ak.array([7, np.nan]), 'col_B':ak.array([1, 9])})
        >>> df
           col_A  col_B
        0    7.0      1
        1    NaN      9 (2 rows x 2 columns)

        >>> df.count()
        col_A    1
        col_B    2
        dtype: int64

        >>> df = ak.DataFrame({'col_A': ak.array(["a","b","c"]), 'col_B':ak.array([1, np.nan, np.nan])})
        >>> df
          col_A  col_B
        0     a    1.0
        1     b    NaN
        2     c    NaN (3 rows x 2 columns)

        >>> df.count()
        col_A    3
        col_B    1
        dtype: int64

        >>> df.count(numeric_only=True)
        col_B    1
        dtype: int64

        >>> df.count(axis=1)
        0    2
        1    1
        2    1
        dtype: int64


        '''
        ...

    def drop(self, keys: 'Union[str, int, List[Union[str, int]]]', axis: 'Union[str, int]' = 0, inplace: 'bool' = False) -> 'Union[None, DataFrame]':
        r'''

        Drop column/s or row/s from the dataframe.

        Parameters
        ----------
        keys : str, int or list
            The labels to be dropped on the given axis.
        axis : int or str
            The axis on which to drop from. 0/'index' - drop rows, 1/'columns' - drop columns.
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.

        Returns
        -------
        DataFrame or None
            DateFrame when `inplace=False`;
            None when `inplace=True`

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4 (2 rows x 2 columns)

        Drop column
        >>> df.drop('col1', axis = 1)
           col2
        0     3
        1     4 (2 rows x 1 columns)

        Drop row
        >>> df.drop(0, axis = 0)
           col1  col2
        1     2     4 (1 rows x 2 columns)


        '''
        ...

    def drop_duplicates(self, subset=None, keep='first'):
        r'''

        Drop duplcated rows and returns resulting DataFrame.

        If a subset of the columns are provided then only one instance of each
        duplicated row will be returned (keep determines which row).

        Parameters
        ----------
        subset : Iterable
            Iterable of column names to use to dedupe.
        keep : {'first', 'last'}, default='first'
            Determines which duplicates (if any) to keep.

        Returns
        -------
        DataFrame
            DataFrame with duplicates removed.

        Example
        -------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2, 2, 3], 'col2': [4, 5, 5, 6]})
        >>> df
           col1  col2
        0     1     4
        1     2     5
        2     2     5
        3     3     6 (4 rows x 2 columns)

        >>> df.drop_duplicates()
           col1  col2
        0     1     4
        1     2     5
        3     3     6 (3 rows x 2 columns)


        '''
        ...

    def dropna(self, axis: 'Union[int, str]' = 0, how: 'Optional[str]' = None, thresh: 'Optional[int]' = None, ignore_index: 'bool' = False) -> 'DataFrame':
        r'''

        Remove missing values.

        Parameters
        ----------
        axis: {0 or 'index', 1 or 'columns'}, default = 0
            Determine if rows or columns which contain missing values are removed.

            0, or 'index': Drop rows which contain missing values.

            1, or 'columns': Drop columns which contain missing value.

            Only a single axis is allowed.
        how: {'any', 'all'}, default='any'
            Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.

            'any': If any NA values are present, drop that row or column.

            'all': If all values are NA, drop that row or column.
        thresh: int, optional
            Require that many non - NA values.Cannot be combined with how.
        ignore_index: bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame
            DataFrame with NA entries dropped from it.

        Examples
        --------
        >>> import arkouda as ak
        >>> import numpy as np
        >>> df = ak.DataFrame(
        ...    {
        ...        "A": [True, True, True, True],
        ...        "B": [1, np.nan, 2, np.nan],
        ...        "C": [1, 2, 3, np.nan],
        ...        "D": [False, False, False, False],
        ...        "E": [1, 2, 3, 4],
        ...        "F": ["a", "b", "c", "d"],
        ...        "G": [1, 2, 3, 4],
        ...    }
        ...   )

        >>> df
              A    B    C      D  E  F  G
        0  True  1.0  1.0  False  1  a  1
        1  True  NaN  2.0  False  2  b  2
        2  True  2.0  3.0  False  3  c  3
        3  True  NaN  NaN  False  4  d  4 (4 rows x 7 columns)

        >>> df.dropna()
              A    B    C      D  E  F  G
        0  True  1.0  1.0  False  1  a  1
        2  True  2.0  3.0  False  3  c  3 (2 rows x 7 columns)

        >>> df.dropna(axis=1)
              A      D  E  F  G
        0  True  False  1  a  1
        1  True  False  2  b  2
        2  True  False  3  c  3
        3  True  False  4  d  4 (4 rows x 5 columns)

        >>> df.dropna(axis=1, thresh=3)
              A    C      D  E  F  G
        0  True  1.0  False  1  a  1
        1  True  2.0  False  2  b  2
        2  True  3.0  False  3  c  3
        3  True  NaN  False  4  d  4 (4 rows x 6 columns)

        >>> df.dropna(axis=1, how="all")
              A    B    C      D  E  F  G
        0  True  1.0  1.0  False  1  a  1
        1  True  NaN  2.0  False  2  b  2
        2  True  2.0  3.0  False  3  c  3
        3  True  NaN  NaN  False  4  d  4 (4 rows x 7 columns)


        '''
        ...

    @property
    def dtypes(self, axis: 'Union[int, str]' = 0, how: 'Optional[str]' = None, thresh: 'Optional[int]' = None, ignore_index: 'bool' = False) -> 'DataFrame':
        r'''

        The dtypes of the dataframe.

        Returns
        -------
        dtypes :  arkouda.pandas.row.Row
            The dtypes of the dataframe.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': ["a", "b"]})
        >>> df
           col1 col2
        0     1    a
        1     2    b (2 rows x 2 columns)

        >>> df.dtypes
        {'col1': 'int64', 'col2': 'str'}


        '''
        ...

    @property
    def empty(self, axis: 'Union[int, str]' = 0, how: 'Optional[str]' = None, thresh: 'Optional[int]' = None, ignore_index: 'bool' = False) -> 'DataFrame':
        r'''

        Whether the dataframe is empty.

        Returns
        -------
        bool
            True if the dataframe is empty, otherwise False.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({})
        >>> df
        Empty DataFrame
        Columns: []
        Index: [] (None rows x 0 columns)

        >>> df.empty
        True


        '''
        ...

    def filter_by_range(self, keys, low=1, high=None):
        r'''

        Filter rows by the size of groups defined on one or more columns.

        Group the DataFrame by the specified `keys`, compute the count of each group,
        and return a boolean mask indicating which rows belong to groups whose sizes
        fall within the inclusive range [`low`, `high`].

        Parameters
        ----------
        keys : str or list of str
            Column name or list of column names to group by.
        low : int, default=1
            Minimum group size (inclusive). Must be >= 0.
        high : int or None, default=None
            Maximum group size (inclusive). If `None`, no upper bound is applied.

        Returns
        -------
        pdarray of bool
            A boolean mask array of length equal to the number of rows in the DataFrame,
            where `True` indicates the row’s group size is between `low` and `high`.

        Raises
        ------
        ValueError
            If `low` is negative, or if `high` is not `None` and `high < low`.
        TypeError
            If `keys` is not a string or list of strings.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2, 2, 2, 3, 3], 'col2': [4, 5, 6, 7, 8, 9]})
        >>> df
           col1  col2
        0     1     4
        1     2     5
        2     2     6
        3     2     7
        4     3     8
        5     3     9 (6 rows x 2 columns)

        >>> df.filter_by_range("col1", low=1, high=2)
        array([True False False False True True])

        >>> filtered_df = df[df.filter_by_range("col1", low=1, high=2)]
        >>> filtered_df
           col1  col2
        0     1     4
        4     3     8
        5     3     9 (3 rows x 2 columns)


        '''
        ...

    def from_pandas(self, pd_df):
        r'''

        Copy the data from a pandas DataFrame into a new arkouda.pandas.dataframe.DataFrame.

        Parameters
        ----------
        pd_df : pandas.DataFrame
            A pandas DataFrame to convert.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import arkouda as ak
        >>> import pandas as pd
        >>> pd_df = pd.DataFrame({"A":[1,2],"B":[3,4]})
        >>> type(pd_df)
        <class 'pandas.core.frame.DataFrame'>
        >>> pd_df
           A  B
        0  1  3
        1  2  4

        >>> ak_df = DataFrame.from_pandas(pd_df)
        >>> type(ak_df)
        <class 'arkouda.dataframe.DataFrame'>
        >>> ak_df
           A  B
        0  1  3
        1  2  4 (2 rows x 2 columns)


        '''
        ...

    def from_return_msg(self, rep_msg):
        r'''

        Create a DataFrame object from an arkouda server response message.

        Parameters
        ----------
        rep_msg : string
            Server response message used to create a DataFrame.

        Returns
        -------
        DataFrame


        '''
        ...

    def groupby(self, keys, use_series=True, as_index=True, dropna=True):
        r'''

        Group the dataframe by a column or a list of columns.

        Alias for GroupBy.

        Parameters
        ----------
        keys : str or list of str
            An (ordered) list of column names or a single string to group by.
        use_series : bool, default=True
            If True, returns an arkouda.pandas.dataframe.DataFrameGroupBy object.
            Otherwise an arkouda.pandas.groupbyclass.GroupBy object.
        as_index: bool, default=True
            If True, groupby columns will be set as index
            otherwise, the groupby columns will be treated as DataFrame columns.
        dropna : bool, default=True
            If True, and the groupby keys contain NaN values,
            the NaN values together with the corresponding row will be dropped.
            Otherwise, the rows corresponding to NaN values will be kept.

        Returns
        -------
        arkouda.pandas.dataframe.DataFrameGroupBy or arkouda.pandas.groupbyclass.GroupBy
            If use_series = True, returns an arkouda.pandas.dataframe.DataFrameGroupBy object.
            Otherwise returns an arkouda.pandas.groupbyclass.GroupBy object.

        See Also
        --------
        arkouda.GroupBy

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1.0, 1.0, 2.0, np.nan], 'col2': [4, 5, 6, 7]})
        >>> df
           col1  col2
        0   1.0     4
        1   1.0     5
        2   2.0     6
        3   NaN     7 (4 rows x 2 columns)

        >>> df.GroupBy("col1") # doctest: +SKIP
        <arkouda.groupbyclass.GroupBy object at 0x795584773f00>
        >>> df.GroupBy("col1").size()
        (array([1.00000000000000000 2.00000000000000000]), array([2 1]))

        >>> df.GroupBy("col1",use_series=True).size()
        col1
        1.0    2
        2.0    1
        dtype: int64
        >>> df.GroupBy("col1",use_series=True, as_index = False).size()
           col1  size
        0   1.0     2
        1   2.0     1 (2 rows x 2 columns)


        '''
        ...

    def head(self, n=5):
        r'''

        Return the first `n` rows.

        This function returns the first `n` rows of the the dataframe. It is
        useful for quickly verifying data, for example, after sorting or
        appending rows.

        Parameters
        ----------
        n : int, default = 5
            Number of rows to select.

        Returns
        -------
        DataFrame
            The first `n` rows of the DataFrame.

        See Also
        --------
        tail

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': ak.arange(10), 'col2': -1 * ak.arange(10)})
        >>> df
           col1  col2
        0     0     0
        1     1    -1
        2     2    -2
        3     3    -3
        4     4    -4
        5     5    -5
        6     6    -6
        7     7    -7
        8     8    -8
        9     9    -9 (10 rows x 2 columns)

        >>> df.head()
           col1  col2
        0     0     0
        1     1    -1
        2     2    -2
        3     3    -3
        4     4    -4 (5 rows x 2 columns)

        >>> df.head(n=2)
           col1  col2
        0     0     0
        1     1    -1 (2 rows x 2 columns)


        '''
        ...

    @property
    def index(self, n=5):
        r'''

        The index of the dataframe.

        Returns
        -------
        arkouda.index.Index or arkouda.index.MultiIndex
            The index of the dataframe.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4 (2 rows x 2 columns)

        >>> df.index
        Index(array([0 1]), dtype='int64')


        '''
        ...

    @property
    def info(self, n=5):
        r'''

        Return a summary string of this dataframe.

        Returns
        -------
        str
            A summary string of this dataframe.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2], 'col2': ["a", "b"]})
        >>> df
           col1 col2
        0     1    a
        1     2    b (2 rows x 2 columns)

        >>> df.info
        "DataFrame(['col1', 'col2'], 2 rows, 36.00 B)"


        '''
        ...

    def is_registered(self) -> 'bool':
        r'''

        Return True if the object is contained in the registry.

        Returns
        -------
        bool
            Indicates if the object is contained in the registry.

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mismatch of registered components.

        See Also
        --------
        register
        unregister
        unregister_dataframe_by_name

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        Example
        -------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df.register("my_table_name")
           col1  col2
        0     1     4
        1     2     5
        2     3     6 (3 rows x 2 columns)

        >>> df.is_registered()
        True
        >>> df.unregister()
        >>> df.is_registered()
        False


        '''
        ...

    def isin(self, values: 'Union[pdarray, Dict, Series, DataFrame]') -> 'DataFrame':
        r'''

        Determine whether each element in the DataFrame is contained in values.

        Parameters
        ----------
        values : pdarray, dict, Series, or DataFrame
            The values to check for in DataFrame. Series can only have a single index.

        Returns
        -------
        DataFrame
            Arkouda DataFrame of booleans showing whether each element in the DataFrame is
            contained in values.

        See Also
        --------
        ak.Series.isin

        Notes
        -----
        - Pandas supports values being an iterable type. In arkouda, we replace this with pdarray.
        - Pandas supports ~ operations. Currently, ak.DataFrame does not support this.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col_A': ak.array([7, 3]), 'col_B':ak.array([1, 9])})
        >>> df
           col_A  col_B
        0      7      1
        1      3      9 (2 rows x 2 columns)

        When `values` is a pdarray, check every value in the DataFrame to determine if
        it exists in values.
        >>> df.isin(ak.array([0, 1]))
           col_A  col_B
        0  False   True
        1  False  False (2 rows x 2 columns)

        When `values` is a dict, the values in the dict are passed to check the column
        indicated by the key.
        >>> df.isin({'col_A': ak.array([0, 3])})
           col_A  col_B
        0  False  False
        1   True  False (2 rows x 2 columns)

        When `values` is a Series, each column is checked if values is present positionally.
        This means that for `True` to be returned, the indexes must be the same.
        >>> i = ak.Index(ak.arange(2))
        >>> s = ak.Series(data=[3, 9], index=i)
        >>> df.isin(s)
           col_A  col_B
        0  False  False
        1  False   True (2 rows x 2 columns)

        When `values` is a DataFrame, the index and column must match.
        Note that 9 is not found because the column name does not match.
        >>> other_df = ak.DataFrame({'col_A':ak.array([7, 3]), 'col_C':ak.array([0, 9])})
        >>> df.isin(other_df)
           col_A  col_B
        0   True  False
        1   True  False (2 rows x 2 columns)


        '''
        ...

    def isna(self) -> 'DataFrame':
        r'''

        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        numpy.NaN values get mapped to True values.
        Everything else gets mapped to False values.

        Returns
        -------
        DataFrame
            Mask of bool values for each element in DataFrame
            that indicates whether an element is an NA value.

        Examples
        --------
        >>> import arkouda as ak
        >>> import numpy as np
        >>> df = ak.DataFrame({"A": [np.nan, 2, 2, 3], "B": [3, np.nan, 5, 6],
        ...          "C": [1, np.nan, 2, np.nan], "D":["a","b","c","d"]})
        >>> df
             A    B    C  D
        0  NaN  3.0  1.0  a
        1  2.0  NaN  NaN  b
        2  2.0  5.0  2.0  c
        3  3.0  6.0  NaN  d (4 rows x 4 columns)

        >>> df.isna()
               A      B      C      D
        0   True  False  False  False
        1  False   True   True  False
        2  False  False  False  False
        3  False  False   True  False (4 rows x 4 columns)


        '''
        ...

    def load(self, prefix_path, file_format='INFER'):
        r'''

        Load dataframe from file.

        file_format needed for consistency with other load functions.

        Parameters
        ----------
        prefix_path : str
            The prefix path for the data.

        file_format : string, default = "INFER"

        Returns
        -------
        DataFrame
            A dataframe loaded from the prefix_path.

        Examples
        --------
        >>> import arkouda as ak

        To store data in <my_dir>/my_data_LOCALE0000,
        use "<my_dir>/my_data" as the prefix.
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'hdf5_output','my_data')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)
        >>> df = ak.DataFrame({"A": ak.arange(5), "B": -1 * ak.arange(5)})
        >>> df.to_parquet(my_path + "/my_data")
        File written successfully!

        >>> df.load(my_path + "/my_data")
           B  A
        0  0  0
        1 -1  1
        2 -2  2
        3 -3  3
        4 -4  4 (5 rows x 2 columns)


        '''
        ...

    def memory_usage(self, index=True, unit='B') -> 'Series':
        r'''

        Return the memory usage of each column in bytes.

        The memory usage can optionally include the contribution of
        the index.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the DataFrame's
            index in returned Series. If ``index=True``, the memory usage of
            the index is the first item in the output.
        unit : str, default = "B"
            Unit to return. One of {'B', 'KB', 'MB', 'GB'}.

        Returns
        -------
        Series
            A Series whose index is the original column names and whose values
            is the memory usage of each column in bytes.

        See Also
        --------
        arkouda.numpy.pdarrayclass.nbytes
        arkouda.index.Index.memory_usage
        arkouda.index.MultiIndex.memory_usage
        arkouda.pandas.series.Series.memory_usage

        Examples
        --------
        >>> import arkouda as ak
        >>> dtypes = {"int64":ak.int64, "float64":ak.float64,  "bool":ak.bool_}
        >>> data = dict([(t, ak.ones(5000, dtype=dtypes[t])) for t in dtypes.keys()])
        >>> df = ak.DataFrame(data)
        >>> df.head()
           int64  float64  bool
        0      1      1.0  True
        1      1      1.0  True
        2      1      1.0  True
        3      1      1.0  True
        4      1      1.0  True (5 rows x 3 columns)

        >>> df.memory_usage()
        Index      40000
        int64      40000
        float64    40000
        bool        5000
        dtype: int64

        >>> df.memory_usage(index=False)
        int64      40000
        float64    40000
        bool        5000
        dtype: int64

        >>> df.memory_usage(unit="KB")
        Index      39.062500
        int64      39.062500
        float64    39.062500
        bool        4.882812
        dtype: float64

        To get the approximate total memory usage:
        >>> df.memory_usage(index=True).sum()
        np.int64(125000)


        '''
        ...

    def memory_usage_info(self, unit='GB'):
        r'''

        Return a formatted string representation of the size of this DataFrame.

        Parameters
        ----------
        unit : str, default = "GB"
            Unit to return. One of {'KB', 'MB', 'GB'}.

        Returns
        -------
        str
            A string representation of the number of bytes used by this DataFrame in [unit]s.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': ak.arange(1000), 'col2': ak.arange(1000)})
        >>> df.memory_usage_info()
        '0.00 GB'

        >>> df.memory_usage_info(unit="KB")
        '23.44 KB'


        '''
        ...

    def merge(self, right: 'DataFrame', on: 'Optional[Union[str, List[str]]]' = None, how: 'str' = 'inner', left_suffix: 'str' = '_x', right_suffix: 'str' = '_y', convert_ints: 'bool' = True, sort: 'bool' = True) -> 'DataFrame':
        r'''

        Merge Arkouda DataFrames with a database-style join.

        The resulting dataframe contains rows from both DataFrames as specified by
        the merge condition (based on the "how" and "on" parameters).

        Based on pandas merge functionality.
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html

        Parameters
        ----------
        right: DataFrame
            The Right DataFrame to be joined.
        on: Optional[Union[str, List[str]]] = None
            The name or list of names of the DataFrame column(s) to join on.
            If on is None, this defaults to the intersection of the columns in both DataFrames.
        how:  {"inner", "left", "right}, default = "inner"
            The merge condition.
            Must be "inner", "left", or "right".
        left_suffix: str, default = "_x"
            A string indicating the suffix to add to columns from the left dataframe for overlapping
            column names in both left and right. Defaults to "_x". Only used when how is "inner".
        right_suffix: str, default = "_y"
            A string indicating the suffix to add to columns from the right dataframe for overlapping
            column names in both left and right. Defaults to "_y". Only used when how is "inner".
        convert_ints: bool = True
            If True, convert columns with missing int values (due to the join) to float64.
            This is to match pandas.
            If False, do not convert the column dtypes.
            This has no effect when how = "inner".
        sort: bool = True
            If True, DataFrame is returned sorted by "on".
            Otherwise, the DataFrame is not sorted.

        Returns
        -------
        DataFrame
            Joined Arkouda DataFrame.

        Note
        ----
        Multiple column joins are only supported for integer columns.

        Examples
        --------
        >>> import arkouda as ak
        >>> left_df = ak.DataFrame({'col1': ak.arange(5), 'col2': -1 * ak.arange(5)})
        >>> left_df
           col1  col2
        0     0     0
        1     1    -1
        2     2    -2
        3     3    -3
        4     4    -4 (5 rows x 2 columns)

        >>> right_df = ak.DataFrame({'col1': 2 * ak.arange(5), 'col2': 2 * ak.arange(5)})
        >>> right_df
           col1  col2
        0     0     0
        1     2     2
        2     4     4
        3     6     6
        4     8     8 (5 rows x 2 columns)

        >>> left_df.merge(right_df, on = "col1")
           col1  col2_x  col2_y
        0     0       0       0
        1     2      -2       2
        2     4      -4       4 (3 rows x 3 columns)

        >>> left_df.merge(right_df, on = "col1", how = "left")
           col1  col2_x  col2_y
        0     0       0     0.0
        1     1      -1     NaN
        2     2      -2     2.0
        3     3      -3     NaN
        4     4      -4     4.0 (5 rows x 3 columns)

        >>> left_df.merge(right_df, on = "col1", how = "right")
           col1  col2_x  col2_y
        0     0     0.0       0
        1     2    -2.0       2
        2     4    -4.0       4
        3     6     NaN       6
        4     8     NaN       8 (5 rows x 3 columns)

        >>> left_df.merge(right_df, on = "col1", how = "outer")
           col1  col2_x  col2_y
        0     0     0.0     0.0
        1     1    -1.0     NaN
        2     2    -2.0     2.0
        3     3    -3.0     NaN
        4     4    -4.0     4.0
        5     6     NaN     6.0
        6     8     NaN     8.0 (7 rows x 3 columns)


        '''
        ...

    def notna(self) -> 'DataFrame':
        r'''

        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        numpy.NaN values get mapped to False values.

        Returns
        -------
        DataFrame
            Mask of bool values for each element in DataFrame
            that indicates whether an element is not an NA value.

        Examples
        --------
        >>> import arkouda as ak
        >>> import numpy as np
        >>> df = ak.DataFrame({"A": [np.nan, 2, 2, 3], "B": [3, np.nan, 5, 6],
        ...          "C": [1, np.nan, 2, np.nan], "D":["a","b","c","d"]})
        >>> df
             A    B    C  D
        0  NaN  3.0  1.0  a
        1  2.0  NaN  NaN  b
        2  2.0  5.0  2.0  c
        3  3.0  6.0  NaN  d (4 rows x 4 columns)

        >>> df.notna()
               A      B      C     D
        0  False   True   True  True
        1   True  False  False  True
        2   True   True   True  True
        3   True   True  False  True (4 rows x 4 columns)


        '''
        ...

    def objType(self, *args, **kwargs):
        r'''
        str(object='') -> str
        str(bytes_or_buffer[, encoding[, errors]]) -> str

        Create a new string object from the given object. If encoding or
        errors is specified, then the object must expose a data buffer
        that will be decoded using the given encoding and error handler.
        Otherwise, returns the result of object.__str__() (if defined)
        or repr(object).
        encoding defaults to 'utf-8'.
        errors defaults to 'strict'.
        '''
        ...

    def read_csv(self, filename: 'str', col_delim: 'str' = ','):
        r'''

        Read the columns of a CSV file into an Arkouda DataFrame.

        If the file contains the appropriately formatted header, typed data will be returned.
        Otherwise, all data will be returned as a Strings objects.

        Parameters
        ----------
        filename: str
            Filename to read data from.
        col_delim: str, default=","
            The delimiter for columns within the data.

        Returns
        -------
        DataFrame
            Arkouda DataFrame containing the columns from the CSV file.

        Raises
        ------
        ValueError
            Raised if all datasets are not present in all parquet files or if one or
            more of the specified files do not exist.
        RuntimeError
            Raised if one or more of the specified files cannot be opened.
            If `allow_errors` is true this may be raised if no values are returned
            from the server.
        TypeError
            Raised if we receive an unknown arkouda_type returned from the server.

        See Also
        --------
        to_csv

        Notes
        -----
        - CSV format is not currently supported by load/load_all operations.
        - The column delimiter is expected to be the same for column names and data.
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline ("\\n") at this time.
        - Unlike other file formats, CSV files store Strings as their UTF-8 format instead of storing
        bytes as uint(8).

        Examples
        --------
        >>> import arkouda as ak
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'csv_output','my_data')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_csv(my_path)
        >>> df2 = ak.DataFrame.read_csv(my_path + "_LOCALE0000")
        >>> df2
           A  B
        0  1  3
        1  2  4 (2 rows x 2 columns)


        '''
        ...

    def register(self, user_defined_name: 'str') -> 'DataFrame':
        r'''

        Register this DataFrame object and underlying components with the Arkouda server.

        Parameters
        ----------
        user_defined_name : str
            User defined name the DataFrame is to be registered under.
            This will be the root name for underlying components.

        Returns
        -------
        DataFrame
            The same DataFrame which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support a
            fluid programming style.
            Please note you cannot register two different DataFrames with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str.
        RegistrationError
            If the server was unable to register the DataFrame with the user_defined_name.

        See Also
        --------
        unregister
        unregister_dataframe_by_name
        is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        Any changes made to a DataFrame object after registering with the server may not be reflected
        in attached copies.

        Example
        -------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df.register("my_table_name")
            col1  col2
        0     1     4
        1     2     5
        2     3     6 (3 rows x 2 columns)
        >>> df.is_registered()
        True
        >>> df.unregister()
        >>> df.is_registered()
        False


        '''
        ...

    def rename(self, mapper: 'Optional[Union[Callable, Dict]]' = None, index: 'Optional[Union[Callable, Dict]]' = None, column: 'Optional[Union[Callable, Dict]]' = None, axis: 'Union[str, int]' = 0, inplace: 'bool' = False) -> 'Optional[DataFrame]':
        r'''

        Rename indexes or columns according to a mapping.

        Parameters
        ----------
        mapper : callable or dict-like, Optional
            Function or dictionary mapping existing values to new values.
            Nonexistent names will not raise an error.
            Uses the value of axis to determine if renaming column or index
        index : callable or dict-like, Optional
            Function or dictionary mapping existing index names to
            new index names. Nonexistent names will not raise an
            error.
            When this is set, axis is ignored.
        column : callable or dict-like, Optional
            Function or dictionary mapping existing column names to
            new column names. Nonexistent names will not raise an
            error.
            When this is set, axis is ignored.
        axis: int or str, default=0
            Indicates which axis to perform the rename.
            0/"index" - Indexes
            1/"column" - Columns
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.

        Returns
        -------
        DataFrame or None
            DateFrame when `inplace=False`;
            None when `inplace=True`.

        Examples
        --------
        >>> import arkouda as ak

        >>> df = ak.DataFrame({"A": ak.array([1, 2, 3]), "B": ak.array([4, 5, 6])})
        >>> df
           A  B
        0  1  4
        1  2  5
        2  3  6 (3 rows x 2 columns)

        Rename columns using a mapping:
        >>> df.rename(column={'A':'a', 'B':'c'})
           a  c
        0  1  4
        1  2  5
        2  3  6 (3 rows x 2 columns)

        Rename indexes using a mapping:
        >>> df.rename(index={0:99, 2:11})
            A  B
        99  1  4
        1   2  5
        11  3  6 (3 rows x 2 columns)

        Rename using an axis style parameter:
        >>> df.rename(str.lower, axis='column')
           a  b
        0  1  4
        1  2  5
        2  3  6 (3 rows x 2 columns)


        '''
        ...

    def reset_index(self, size: 'Optional[int]' = None, inplace: 'bool' = False) -> 'Union[None, DataFrame]':
        r'''

        Set the index to an integer range.

        Useful if this dataframe is the result of a slice operation from
        another dataframe, or if you have permuted the rows and no longer need
        to keep that ordering on the rows.

        Parameters
        ----------
        size : int, optional
            If size is passed, do not attempt to determine size based on
            existing column sizes. Assume caller handles consistency correctly.
        inplace: bool, default=False
            When True, perform the operation on the calling object.
            When False, return a new object.

        Returns
        -------
        DataFrame or None
            DateFrame when `inplace=False`;
            None when `inplace=True`.

        NOTE
        ----------
        Pandas adds a column 'index' to indicate the original index. Arkouda does not currently
        support this behavior.

        Example
        -------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"A": ak.array([1, 2, 3]), "B": ak.array([4, 5, 6])})
        >>> df
           A  B
        0  1  4
        1  2  5
        2  3  6 (3 rows x 2 columns)

        >>> perm_df = df[ak.array([0,2,1])]
        >>> perm_df
           A  B
        0  1  4
        2  3  6
        1  2  5 (3 rows x 2 columns)

        >>> perm_df.reset_index()
           A  B
        0  1  4
        1  3  6
        2  2  5 (3 rows x 2 columns)


        '''
        ...

    def sample(self, n=5) -> 'DataFrame':
        r'''

        Return a random sample of `n` rows.

        Parameters
        ----------
        n : int, default=5
            Number of rows to return.

        Returns
        -------
        DataFrame
            The sampled `n` rows of the DataFrame.

        Example
        -------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"A": ak.arange(5), "B": -1 * ak.arange(5)})
        >>> df
           A  B
        0  0  0
        1  1 -1
        2  2 -2
        3  3 -3
        4  4 -4 (5 rows x 2 columns)

        Random output of size 3:
        >>> df.sample(n=3)  # doctest: +SKIP
           A  B
        4  4 -4
        3  3 -3
        1  1 -1 (3 rows x 2 columns)


        '''
        ...

    @property
    def shape(self, n=5) -> 'DataFrame':
        r'''

        The shape of the dataframe.

        Returns
        -------
        tuple of int
            Tuple of array dimensions.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df
           col1  col2
        0     1     4
        1     2     5
        2     3     6 (3 rows x 2 columns)

        >>> df.shape
        (3, 2)


        '''
        ...

    @property
    def size(self, n=5) -> 'DataFrame':
        r'''

        Return the number of bytes on the arkouda server.

        Returns
        -------
        int
            The number of bytes on the arkouda server.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df
           col1  col2
        0     1     4
        1     2     5
        2     3     6 (3 rows x 2 columns)

        >>> df.size
        6


        '''
        ...

    def sort_index(self, ascending=True):
        r'''

        Sort the DataFrame by indexed columns.

        Note: Fails on sort order of arkouda.numpy.strings.Strings columns when
            multiple columns being sorted.

        Parameters
        ----------
        ascending : bool, default = True
            Sort values in ascending (default) or descending order.

        Example
        -------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1.1, 3.1, 2.1], 'col2': [6, 5, 4]},
        ...          index = Index(ak.array([2,0,1]), name="idx"))

        >>> df
             col1  col2
        idx
        2     1.1     6
        0     3.1     5
        1     2.1     4 (3 rows x 2 columns)

        >>> df.sort_index()
             col1  col2
        idx
        0     3.1     5
        1     2.1     4
        2     1.1     6 (3 rows x 2 columns)


        '''
        ...

    def sort_values(self, by=None, ascending=True):
        r'''

        Sort the DataFrame by one or more columns.

        If no column is specified, all columns are used.

        Note: Fails on order of arkouda.numpy.strings.Strings columns when multiple columns being sorted.

        Parameters
        ----------
        by : str or list/tuple of str, default = None
            The name(s) of the column(s) to sort by.
        ascending : bool, default = True
            Sort values in ascending (default) or descending order.

        See Also
        --------
        apply_permutation

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [2, 2, 1], 'col2': [3, 4, 3], 'col3':[5, 6, 7]})
        >>> df
           col1  col2  col3
        0     2     3     5
        1     2     4     6
        2     1     3     7 (3 rows x 3 columns)

        >>> df.sort_values()
           col1  col2  col3
        2     1     3     7
        0     2     3     5
        1     2     4     6 (3 rows x 3 columns)

        >>> df.sort_values("col3")
           col1  col2  col3
        0     2     3     5
        1     2     4     6
        2     1     3     7 (3 rows x 3 columns)


        '''
        ...

    def tail(self, n=5):
        r'''

        Return the last `n` rows.

        This function returns the last `n` rows for the dataframe. It is
        useful for quickly testing if your object has the right type of data in
        it.

        Parameters
        ----------
        n : int, default=5
            Number of rows to select.

        Returns
        -------
        DataFrame
            The last `n` rows of the DataFrame.

        See Also
        --------
        arkouda.pandas.dataframe.head

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': ak.arange(10), 'col2': -1 * ak.arange(10)})
        >>> df
           col1  col2
        0     0     0
        1     1    -1
        2     2    -2
        3     3    -3
        4     4    -4
        5     5    -5
        6     6    -6
        7     7    -7
        8     8    -8
        9     9    -9 (10 rows x 2 columns)

        >>> df.tail()
           col1  col2
        5     5    -5
        6     6    -6
        7     7    -7
        8     8    -8
        9     9    -9 (5 rows x 2 columns)

        >>> df.tail(n=2)
           col1  col2
        8     8    -8
        9     9    -9 (2 rows x 2 columns)


        '''
        ...

    def to_csv(self, path: 'str', index: 'bool' = False, columns: 'Optional[List[str]]' = None, col_delim: 'str' = ',', overwrite: 'bool' = False):
        r'''

        Write DataFrame to CSV file(s).

        File will contain a column for each column in the DataFrame.
        All CSV Files written by Arkouda include a header denoting data types of the columns.
        Unlike other file formats, CSV files store Strings as their UTF-8 format instead of storing
        bytes as uint(8).

        Parameters
        ----------
        path: str
            The filename prefix to be used for saving files. Files will have _LOCALE#### appended
            when they are written to disk.
        index: bool, default=False
            If True, the index of the DataFrame will be written to the file
            as a column.
        columns: list of str (Optional)
            Column names to assign when writing data.
        col_delim: str, default=","
            Value to be used to separate columns within the file.
            Please be sure that the value used DOES NOT appear in your dataset.
        overwrite: bool, default=False
            If True, any existing files matching your provided prefix_path will
            be overwritten. If False, an error will be returned if existing files are found.

        Raises
        ------
        ValueError
            Raised if all datasets are not present in all parquet files or if one or
            more of the specified files do not exist.
        RuntimeError
            Raised if one or more of the specified files cannot be opened.
            If `allow_errors` is true this may be raised if no values are returned
            from the server.
        TypeError
            Raised if we receive an unknown arkouda_type returned from the server.

        Notes
        -----
        - CSV format is not currently supported by load/load_all operations.
        - The column delimiter is expected to be the same for column names and data.
        - Be sure that column delimiters are not found within your data.
        - All CSV files must delimit rows using newline ("\\n") at this time.

        Examples
        --------
        >>> import arkouda as ak
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'csv_output')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_csv(my_path)
        >>> df2 = ak.DataFrame.read_csv(my_path + "_LOCALE0000")
        >>> df2
           A  B
        0  1  3
        1  2  4 (2 rows x 2 columns)


        '''
        ...

    def to_hdf(self, path, index=False, columns=None, file_type='distribute'):
        r'''

        Save DataFrame to disk as hdf5, preserving column names.

        Parameters
        ----------
        path : str
            File path to save data.
        index : bool, default=False
            If True, save the index column. By default, do not save the index.
        columns: List, default = None
            List of columns to include in the file. If None, writes out all columns.
        file_type: str (single | distribute), default=distribute
            Whether to save to a single file or distribute across Locales.

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray.

        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.

        See Also
        --------
        to_parquet
        load

        Examples
        --------
        >>> import arkouda as ak
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'hdf_output')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_hdf(my_path + "/my_data")
        >>> df.load(my_path + "/my_data")
           A  B
        0  1  3
        1  2  4 (2 rows x 2 columns)


        '''
        ...

    def to_markdown(self, mode='wt', index=True, tablefmt='grid', storage_options=None, **kwargs):
        r'''

        Print DataFrame in Markdown-friendly format.

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
        This function should only be called on small DataFrames as it calls pandas.DataFrame.to_markdown:
        https://pandas.pydata.org/pandas-docs/version/1.2.4/reference/api/pandas.DataFrame.to_markdown.html

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]})
        >>> print(df.to_markdown())
        +----+------------+------------+
        |    | animal_1   | animal_2   |
        +====+============+============+
        |  0 | elk        | dog        |
        +----+------------+------------+
        |  1 | pig        | quetzal    |
        +----+------------+------------+

        Suppress the index:
        >>> print(df.to_markdown(index = False))
        +------------+------------+
        | animal_1   | animal_2   |
        +============+============+
        | elk        | dog        |
        +------------+------------+
        | pig        | quetzal    |
        +------------+------------+


        '''
        ...

    def to_pandas(self, datalimit=1073741824, retain_index=False):
        r'''

        Send this DataFrame to a pandas DataFrame.

        Parameters
        ----------
        datalimit : int, default=arkouda.client.maxTransferBytes
            The maximum number size, in megabytes to transfer. The requested
            DataFrame will be converted to a pandas DataFrame only if the
            estimated size of the DataFrame does not exceed this value.

        retain_index : bool, default=False
            Normally, to_pandas() creates a new range index object. If you want
            to keep the index column, set this to True.

        Returns
        -------
        pandas.DataFrame
            The result of converting this DataFrame to a pandas DataFrame.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak_df = ak.DataFrame({"A": ak.arange(2), "B": -1 * ak.arange(2)})
        >>> type(ak_df)
        <class 'arkouda.dataframe.DataFrame'>
        >>> ak_df
           A  B
        0  0  0
        1  1 -1 (2 rows x 2 columns)

        >>> import pandas as pd
        >>> pd_df = ak_df.to_pandas()
        >>> type(pd_df)
        <class 'pandas.core.frame.DataFrame'>
        >>> pd_df
           A  B
        0  0  0
        1  1 -1


        '''
        ...

    def to_parquet(self, path, index=False, columns=None, compression: 'Optional[str]' = None, convert_categoricals: 'bool' = False):
        r'''

        Save DataFrame to disk as parquet, preserving column names.

        Parameters
        ----------
        path : str
            File path to save data.
        index : bool, default=False
            If True, save the index column. By default, do not save the index.
        columns: list
            List of columns to include in the file. If None, writes out all columns.
        compression : str (Optional), default=None
            Provide the compression type to use when writing the file.
            Supported values: snappy, gzip, brotli, zstd, lz4
        convert_categoricals: bool, default=False
            Parquet requires all columns to be the same size and Categoricals
            don't satisfy that requirement.
            If set, write the equivalent Strings in place of any Categorical columns.

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray

        Notes
        -----
        This method saves one file per locale of the arkouda server. All
        files are prefixed by the path argument and suffixed by their
        locale number.

        See Also
        --------
        to_hdf
        load

        Examples
        --------
        >>> import arkouda as ak
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'parquet_output')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_parquet(my_path + "/my_data")
        File written successfully!

        >>> df.load(my_path + "/my_data")
           B  A
        0  3  1
        1  4  2 (2 rows x 2 columns)


        '''
        ...

    def transfer(self, hostname, port):
        r'''

        Send a DataFrame to a different Arkouda server.

        Parameters
        ----------
        hostname : str
            The hostname where the Arkouda server intended to
            receive the DataFrame is running.
        port : int_scalars
            The port to send the array over. This needs to be an
            open port (i.e., not one that the Arkouda server is
            running on). This will open up `numLocales` ports,
            each of which in succession, so will use ports of the
            range {port..(port+numLocales)} (e.g., running an
            Arkouda server of 4 nodes, port 1234 is passed as
            `port`, Arkouda will use ports 1234, 1235, 1236,
            and 1237 to send the array data).
            This port much match the port passed to the call to
            `ak.receive_array()`.


        Returns
        -------
        str
            A message indicating a complete transfer.

        Raises
        ------
        ValueError
            Raised if the op is not within the pdarray.BinOps set
        TypeError
            Raised if other is not a pdarray or the pdarray.dtype is not
            a supported dtype


        '''
        ...

    def unregister(self):
        r'''

        Unregister this DataFrame object in the arkouda server.

        Unregister this DataFrame object in the arkouda server which was previously
        registered using register() and/or attached to using attach().

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister.

        See Also
        --------
        register
        unregister_dataframe_by_name
        is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        Example
        -------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        >>> df.register("my_table_name")
                   col1  col2
        0     1     4
        1     2     5
        2     3     6 (3 rows x 2 columns)

        >>> df.is_registered()
        True
        >>> df.unregister()
        >>> df.is_registered()
        False


        '''
        ...

    def update_hdf(self, prefix_path: 'str', index=False, columns=None, repack: 'bool' = True):
        r'''

        Overwrite the dataset with the name provided with this dataframe.

        If the dataset does not exist it is added.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share.
        index : bool, default=False
            If True, save the index column. By default, do not save the index.
        columns: List, default=None
            List of columns to include in the file. If None, writes out all columns.
        repack: bool, default=True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Returns
        -------
        str
            Success message if successful.

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the pdarray.

        Notes
        -----
        If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        If the dataset provided does not exist, it will be added.

        Examples
        --------
        >>> import arkouda as ak
        >>> import os.path
        >>> from pathlib import Path
        >>> my_path = os.path.join(os.getcwd(), 'hdf_output')
        >>> Path(my_path).mkdir(parents=True, exist_ok=True)

        >>> df = ak.DataFrame({"A":[1,2],"B":[3,4]})
        >>> df.to_hdf(my_path + "/my_data")
        >>> df.load(my_path + "/my_data")
           A  B
        0  1  3
        1  2  4 (2 rows x 2 columns)

        >>> df2 = ak.DataFrame({"A":[5,6],"B":[7,8]})
        >>> df2.update_hdf(my_path + "/my_data")
        >>> df.load(my_path + "/my_data")
           A  B
        0  5  7
        1  6  8 (2 rows x 2 columns)


        '''
        ...

    def update_nrows(self):
        r'''
        Compute the number of rows on the arkouda server and updates the size parameter.
        '''
        ...


class DataFrameGroupBy:
    r'''

    A DataFrame that has been grouped by a subset of columns.

    Parameters
    ----------
    gb_key_names : str or list(str), default=None
        The column name(s) associated with the aggregated columns.
    as_index : bool, default=True
        If True, interpret aggregated column as index
        (only implemented for single dimensional aggregates).
        Otherwise, treat aggregated column as a dataframe column.

    Attributes
    ----------
    gb : GroupBy
        GroupBy object, where the aggregation keys are values of column(s) of a dataframe,
        usually in preparation for aggregating with respect to the other columns.
    df : DataFrame
        The dataframe containing the original data.
    gb_key_names : Union[str, List[str]]
        The column name(s) associated with the aggregated columns.
    as_index : bool
        If True the grouped values of the aggregation keys will be treated as an index.
        Defaults to True.


    '''
    ...

    def _get_df_col(self, c):

        ...

    def _make_aggop(self, opname):

        ...

    def _return_agg_dataframe(self, values, name, sort_index=True):

        ...

    def _return_agg_series(self, values, sort_index=True):

        ...

    def all(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def __and__(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def any(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def argmax(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def argmin(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def broadcast(self, x, permute=True):
        r'''

        Fill each group’s segment with a constant value.

        Parameters
        ----------
        x :  Series or pdarray
            The values to put in each group’s segment.
        permute : bool, default=True
            If True (default), permute broadcast values back to the
            ordering of the original array on which GroupBy was called.
            If False, the broadcast values are grouped by value.

        Returns
        -------
        arkouda.pandas.series.Series
            A Series with the Index of the original frame and the values of the broadcast.

        Examples
        --------
        >>> import arkouda as ak
        >>> from arkouda.pandas.dataframe import DataFrameGroupBy
        >>> df = ak.DataFrame({"A":[1,2,2,3],"B":[3,4,5,6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  2  5
        3  3  6 (4 rows x 2 columns)

        >>> gb = df.groupby("A")
        >>> x = ak.array([10,11,12])
        >>> s = DataFrameGroupBy.broadcast(gb, x)
        >>> df["C"] = s.values
        >>> df
           A  B   C
        0  1  3  10
        1  2  4  11
        2  2  5  11
        3  3  6  12 (4 rows x 3 columns)


        '''
        ...

    def count(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def diff(self, colname):
        r'''

        Create a difference aggregate for the given column.

        For each group, the difference between successive values is calculated.
        Aggregate operations (mean,min,max,std,var) can be done on the results.

        Parameters
        ----------
        colname:  str
            Name of the column to compute the difference on.

        Returns
        -------
        DiffAggregate
            Object containing the differences, which can be aggregated.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"A":[1,2,2,2,3,3],"B":[3,9,11,27,86,100]})
        >>> df
           A    B
        0  1    3
        1  2    9
        2  2   11
        3  2   27
        4  3   86
        5  3  100 (6 rows x 2 columns)

        >>> gb = df.groupby("A")
        >>> gb.diff("B").values
        array([nan nan 2.00000000000000000 16.00000000000000000 nan 14.00000000000000000])


        '''
        ...

    def first(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def head(self, n: 'int' = 5, sort_index: 'bool' = True) -> 'DataFrame':
        r'''

        Return the first n rows from each group.

        Parameters
        ----------
        n: int, optional, default = 5
            Maximum number of rows to return for each group.
            If the number of rows in a group is less than n,
            all the values from that group will be returned.
        sort_index: bool, default = True
            If true, return the DataFrame with indices sorted.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"a":ak.arange(10) %3 , "b":ak.arange(10)})
        >>> df
           a  b
        0  0  0
        1  1  1
        2  2  2
        3  0  3
        4  1  4
        5  2  5
        6  0  6
        7  1  7
        8  2  8
        9  0  9 (10 rows x 2 columns)

        >>> df.groupby("a").head(2)
           a  b
        0  0  0
        1  1  1
        2  2  2
        3  0  3
        4  1  4
        5  2  5 (6 rows x 2 columns)


        '''
        ...

    def max(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def mean(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def median(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def min(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def mode(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def nunique(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def __or__(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def prod(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None):
        r'''

        Return a random sample from each group.

        You can either specify the number of elements
        or the fraction of elements to be sampled. random_state can be used for reproducibility

        Parameters
        ----------
        n: int, optional
            Number of items to return for each group.
            Cannot be used with frac and must be no larger than
            the smallest group unless replace is True.
            Default is one if frac is None.

        frac: float, optional
            Fraction of items to return. Cannot be used with n.

        replace: bool, default False
            Allow or disallow sampling of the same row more than once.

        weights: pdarray, optional
            Default None results in equal probability weighting.
            If passed a pdarray, then values must have the same length as the underlying DataFrame
            and will be used as sampling probabilities after normalization within each group.
            Weights must be non-negative with at least one positive element within each group.

        random_state: int or ak.random.Generator, optional
            If int, seed for random number generator.
            If ak.random.Generator, use as given.

        Returns
        -------
        DataFrame
            A new DataFrame containing items randomly sampled from each group
            sorted according to the grouped columns.

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"A":[3,1,2,1,2,3],"B":[3,4,5,6,7,8]})
        >>> df
           A  B
        0  3  3
        1  1  4
        2  2  5
        3  1  6
        4  2  7
        5  3  8 (6 rows x 2 columns)

        >>> df.groupby("A").sample(random_state=6)
           A  B
        3  1  6
        4  2  7
        5  3  8 (3 rows x 2 columns)

        >>> df.groupby("A").sample(frac=0.5, random_state=3, weights=ak.array([1,1,1,0,0,0]))
           A  B
        1  1  4
        2  2  5
        0  3  3 (3 rows x 2 columns)

        >>> df.groupby("A").sample(n=3, replace=True, random_state=ak.random.default_rng(7))
           A  B
        1  1  4
        3  1  6
        1  1  4
        4  2  7
        4  2  7
        4  2  7
        0  3  3
        5  3  8
        5  3  8 (9 rows x 2 columns)


        '''
        ...

    def size(self, as_series=None, sort_index=True):
        r'''

        Compute the size of each value as the total number of rows, including NaN values.

        Parameters
        ----------
        as_series : bool, default=None
            Indicates whether to return arkouda.pandas.dataframe.DataFrame (if as_series = False) or
            arkouda.pandas.series.Series (if as_series = True)
        sort_index : bool, default=True
            If True, results will be returned with index values sorted in ascending order.

        Returns
        -------
        arkouda.pandas.dataframe.DataFrame or arkouda.pandas.series.Series

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"A":[1,2,2,3],"B":[3,4,5,6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  2  5
        3  3  6 (4 rows x 2 columns)

        >>> df.groupby("A").size(as_series = False)
           size
        A
        1     1
        2     2
        3     1 (3 rows x 1 columns)


        '''
        ...

    def std(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def sum(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def tail(self, n: 'int' = 5, sort_index: 'bool' = True) -> 'DataFrame':
        r'''

        Return the last n rows from each group.

        Parameters
        ----------
        n: int, optional, default = 5
            Maximum number of rows to return for each group.
            If the number of rows in a group is less than n,
            all the rows from that group will be returned.
        sort_index: bool, default = True
            If true, return the DataFrame with indices sorted.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import arkouda as ak
        >>> df = ak.DataFrame({"a":ak.arange(10) %3 , "b":ak.arange(10)})
        >>> df
           a  b
        0  0  0
        1  1  1
        2  2  2
        3  0  3
        4  1  4
        5  2  5
        6  0  6
        7  1  7
        8  2  8
        9  0  9 (10 rows x 2 columns)

        >>> df.groupby("a").tail(2)
           a  b
        4  1  4
        5  2  5
        6  0  6
        7  1  7
        8  2  8
        9  0  9 (6 rows x 2 columns)


        '''
        ...

    def unique(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def var(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...

    def xor(self, colnames=None):
        r'''

        Aggregate the operation, with the grouped column(s) values as keys.

        Parameters
        ----------
        colnames : (list of) str, default=None
            Column name or list of column names to compute the aggregation over.

        Returns
        -------
        DataFrame


        '''
        ...


class DiffAggregate:
    r'''

    A column in a GroupBy that has been differenced.

    Aggregation operations can be done on the result.

    Attributes
    ----------
    gb : GroupBy
        GroupBy object, where the aggregation keys are values of column(s) of a dataframe.
    values : Series
        A column to compute the difference on.


    '''
    ...

    def _make_aggop(self, opname):

        ...

    def all(self):

        ...

    def __and__(self):

        ...

    def any(self):

        ...

    def argmax(self):

        ...

    def argmin(self):

        ...

    def count(self):

        ...

    def first(self):

        ...

    def max(self):

        ...

    def mean(self):

        ...

    def median(self):

        ...

    def min(self):

        ...

    def mode(self):

        ...

    def nunique(self):

        ...

    def __or__(self):

        ...

    def prod(self):

        ...

    def std(self):

        ...

    def sum(self):

        ...

    def unique(self):

        ...

    def var(self):

        ...

    def xor(self):

        ...


def intersect(a, b, positions=True, unique=False):
    r'''

    Find the intersection of two arkouda arrays.

    This function can be especially useful when `positions=True` so
    that the caller gets the indices of values present in both arrays.

    Parameters
    ----------
    a : Strings or pdarray
        An array of strings.

    b : Strings or pdarray
        An array of strings.

    positions : bool, default=True
        Return tuple of boolean pdarrays that indicate positions in `a` and `b`
        of the intersection values.

    unique : bool, default=False
        If the number of distinct values in `a` (and `b`) is equal to the size of
        `a` (and `b`), there is a more efficient method to compute the intersection.

    Returns
    -------
    (arkouda.numpy.pdarrayclass.pdarray, arkouda.numpy.pdarrayclass.pdarray) or
    arkouda.numpy.pdarrayclass.pdarray
        The indices of `a` and `b` where any element occurs at least once in both
        arrays.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.arange(10)
    >>> print(a)
    [0 1 2 3 4 5 6 7 8 9]

    >>> b = 2 * ak.arange(10)
    >>> print(b)
    [0 2 4 6 8 10 12 14 16 18]

    >>> intersect(a,b, positions=True)
    (array([True False True False True False True False True False]),
    array([True True True True True False False False False False]))

    >>> intersect(a,b, positions=False)
    array([0 2 4 6 8])


    '''
    ...


def intx(a, b):
    r'''

    Find all the rows that are in both dataframes.

    Columns should be in identical order.

    Note: does not work for columns of floating point values, but does work for
    Strings, pdarrays of int64 type, and Categorical *should* work.

    Examples
    --------
    >>> import arkouda as ak
    >>> a = ak.DataFrame({'a':ak.arange(5),'b': 2* ak.arange(5)})
    >>> a
       a  b
    0  0  0
    1  1  2
    2  2  4
    3  3  6
    4  4  8 (5 rows x 2 columns)

    >>> b = ak.DataFrame({'a':ak.arange(5),'b':ak.array([0,3,4,7,8])})
    >>> b
       a  b
    0  0  0
    1  1  3
    2  2  4
    3  3  7
    4  4  8 (5 rows x 2 columns)

    >>> intx(a,b)
    array([True False True False True])
    >>> intersect_df = a[intx(a,b)]
    >>> intersect_df
       a  b
    0  0  0
    2  2  4
    4  4  8 (3 rows x 2 columns)


    '''
    ...


def invert_permutation(perm):
    r'''

    Find the inverse of a permutation array.

    Parameters
    ----------
    perm : pdarray
        The permutation array.

    Returns
    -------
    arkouda.numpy.pdarrayclass.pdarray
        The inverse of the permutation array.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda.index import Index
    >>> i = Index(ak.array([1,2,0,5,4]))
    >>> perm = i.argsort()
    >>> print(perm)
    [2 0 1 4 3]
    >>> invert_permutation(perm)
    array([1 2 0 4 3])


    '''
    ...


def merge(left: 'DataFrame', right: 'DataFrame', on: 'Optional[Union[str, List[str]]]' = None, left_on: 'Optional[Union[str, List[str]]]' = None, right_on: 'Optional[Union[str, List[str]]]' = None, how: 'str' = 'inner', left_suffix: 'str' = '_x', right_suffix: 'str' = '_y', convert_ints: 'bool' = True, sort: 'bool' = True) -> 'DataFrame':
    r'''

    Merge Arkouda DataFrames with a database-style join.

    The resulting dataframe contains rows from both DataFrames as specified by
    the merge condition (based on the "how" and "on" parameters).

    Based on pandas merge functionality.
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html

    Parameters
    ----------
    left: DataFrame
        The Left DataFrame to be joined.
    right: DataFrame
        The Right DataFrame to be joined.
    on: Optional[Union[str, List[str]]] = None
        The name or list of names of the DataFrame column(s) to join on.
        If on is None, this defaults to the intersection of the columns in both DataFrames.
    left_on: str or List of str, optional
        Column name or names to join on in the left DataFrame. If this is not None, then right_on
        must also not be None, and this will override `on`.
    right_on: str or List of str, optional
        Column name or names to join on in the right DataFrame. If this is not None, then left_on
        must also not be None, and this will override `on`.
    how: str, default = "inner"
        The merge condition.
        Must be one of "inner", "left", "right", or "outer".
    left_suffix: str, default = "_x"
        A string indicating the suffix to add to columns from the left dataframe for overlapping
        column names in both left and right. Defaults to "_x". Only used when how is "inner".
    right_suffix: str, default = "_y"
        A string indicating the suffix to add to columns from the right dataframe for overlapping
        column names in both left and right. Defaults to "_y". Only used when how is "inner".
    convert_ints: bool = True
        If True, convert columns with missing int values (due to the join) to float64.
        This is to match pandas.
        If False, do not convert the column dtypes.
        This has no effect when how = "inner".
    sort: bool = True
        If True, DataFrame is returned sorted by "on".
        Otherwise, the DataFrame is not sorted.

    Returns
    -------
    DataFrame
        Joined Arkouda DataFrame.

    Note
    ----
    Multiple column joins are only supported for integer columns.

    Examples
    --------
    >>> import arkouda as ak
    >>> from arkouda import merge
    >>> left_df = ak.DataFrame({'col1': ak.arange(5), 'col2': -1 * ak.arange(5)})
    >>> left_df
       col1  col2
    0     0     0
    1     1    -1
    2     2    -2
    3     3    -3
    4     4    -4 (5 rows x 2 columns)

    >>> right_df = ak.DataFrame({'col1': 2 * ak.arange(5), 'col2': 2 * ak.arange(5)})
    >>> right_df
       col1  col2
    0     0     0
    1     2     2
    2     4     4
    3     6     6
    4     8     8 (5 rows x 2 columns)

    >>> merge(left_df, right_df, on = "col1")
       col1  col2_x  col2_y
    0     0       0       0
    1     2      -2       2
    2     4      -4       4 (3 rows x 3 columns)

    >>> merge(left_df, right_df, on = "col1", how = "left")
       col1  col2_x  col2_y
    0     0       0     0.0
    1     1      -1     NaN
    2     2      -2     2.0
    3     3      -3     NaN
    4     4      -4     4.0 (5 rows x 3 columns)

    >>> merge(left_df, right_df, on = "col1", how = "right")
       col1  col2_x  col2_y
    0     0     0.0       0
    1     2    -2.0       2
    2     4    -4.0       4
    3     6     NaN       6
    4     8     NaN       8 (5 rows x 3 columns)

    >>> merge(left_df, right_df, on = "col1", how = "outer")
       col1  col2_x  col2_y
    0     0     0.0     0.0
    1     1    -1.0     NaN
    2     2    -2.0     2.0
    3     3    -3.0     NaN
    4     4    -4.0     4.0
    5     6     NaN     6.0
    6     8     NaN     8.0 (7 rows x 3 columns)


    '''
    ...
