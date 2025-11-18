# flake8: noqa
# mypy: ignore-errors
from _typeshed import Incomplete


class GROUPBY_REDUCTION_TYPES:
    r'''
    Build an immutable unordered collection of unique elements.
    '''
    ...

    def copy(self, ):
        r'''
        Return a shallow copy of a set.
        '''
        ...

    def difference(self, *others):
        r'''
        Return a new set with elements in the set that are not in the others.
        '''
        ...

    def intersection(self, *others):
        r'''
        Return a new set with elements common to the set and all others.
        '''
        ...

    def isdisjoint(self, other, /):
        r'''
        Return True if two sets have a null intersection.
        '''
        ...

    def issubset(self, other, /):
        r'''
        Report whether another set contains this set.
        '''
        ...

    def issuperset(self, other, /):
        r'''
        Report whether this set contains another set.
        '''
        ...

    def symmetric_difference(self, other, /):
        r'''
        Return a new set with elements in either the set or other but not both.
        '''
        ...

    def union(self, *others):
        r'''
        Return a new set with elements from the set and all others.
        '''
        ...


class GroupBy:
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

    def AND(self, values: 'pdarray') -> 'Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]':
        r'''

        Bitwise AND of values in each segment.

        Group another array of values and perform a bitwise AND reduction on each group.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray, int64
            The values to group and reduce with AND

        Returns
        -------
        Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            result : pdarray, int64
                Bitwise AND of values in segments corresponding to keys

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray or if the pdarray
            dtype is not int64
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if all is not supported for the values dtype


        '''
        ...

    def OR(self, values: 'pdarray') -> 'Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]':
        r'''

        Bitwise OR of values in each segment.

        Group another array of values and perform a bitwise OR reduction on each group.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray, int64
            The values to group and reduce with OR

        Returns
        -------
        Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            result : pdarray, int64
                Bitwise OR of values in segments corresponding to keys

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray or if the pdarray
            dtype is not int64
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if all is not supported for the values dtype


        '''
        ...

    def Reductions(self, *args, **kwargs):
        r'''
        Build an immutable unordered collection of unique elements.
        '''
        ...

    def XOR(self, values: 'pdarray') -> 'Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]':
        r'''

        Bitwise XOR of values in each segment.

        Group another array of values and perform a bitwise XOR reduction on each group.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray, int64
            The values to group and reduce with XOR

        Returns
        -------
        Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            result : pdarray, int64
                Bitwise XOR of values in segments corresponding to keys

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray or if the pdarray
            dtype is not int64
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if all is not supported for the values dtype


        '''
        ...

    def _get_groupby_required_pieces(self) -> 'Dict':
        r'''

        Return a dictionary with all required components of self.

        Returns
        -------
        Dict
            Dictionary of all required components of self
                Components (keys, permutation)


        '''
        ...

    def _nested_grouping_helper(self, values: 'groupable') -> 'groupable':

        ...

    def aggregate(self, values: 'groupable', operator: 'str', skipna: 'bool' = True, ddof: 'int_scalars' = 1) -> 'Tuple[groupable, groupable]':
        r'''

        Group another array of values and apply a reduction to each group's values.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and reduce
        operator: str
            The name of the reduction operator to use
        skipna: bool
            boolean which determines if NANs should be skipped
        ddof : int_scalars
            "Delta Degrees of Freedom" used in calculating std

        Returns
        -------
        Tuple[groupable, groupable]
            unique_keys : groupable
                The unique keys, in grouped order
            aggregates : groupable
                One aggregate value per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if the requested operator is not supported for the
            values dtype

        Examples
        --------
        >>> import arkouda as ak
        >>> keys = ak.arange(0, 5)
        >>> vals = ak.linspace(-1, 1, 5)
        >>> g = ak.GroupBy(keys)
        >>> g.aggregate(vals, 'sum')
        (array([0 1 2 3 4]),
         array([-1.00000000000000000 -0.5 0.00000000000000000 0.5 1.00000000000000000]))
        >>> g.aggregate(vals, 'min')
        (array([0 1 2 3 4]),
         array([-1.00000000000000000 -0.5 0.00000000000000000 0.5 1.00000000000000000]))


        '''
        ...

    def all(self, values: 'pdarray') -> 'Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]':
        r'''

        Group another array of values and perform an "and" reduction on each group.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray, bool
            The values to group and reduce with "and"

        Returns
        -------
        Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_any : pdarray, bool
                One bool per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray or if the pdarray
            dtype is not bool
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if all is not supported for the values dtype


        '''
        ...

    def any(self, values: 'pdarray') -> 'Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]':
        r'''

        Group another array of values and perform an "or" reduction on each group.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray, bool
            The values to group and reduce with "or"

        Returns
        -------
        Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_any : pdarray, bool
                One bool per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray or if the pdarray
            dtype is not bool
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array


        '''
        ...

    def argmax(self, values: 'pdarray') -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and return the location of the first maximum of each group.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and find argmax

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_argmaxima : pdarray, int64
                One index per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object or if argmax
            is not supported for the values dtype
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array

        Notes
        -----
        The returned indices refer to the original values array as passed in,
        not the permutation applied by the GroupBy instance.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4 3])
        >>> b = ak.randint(1, 5, 10, seed=1)
        >>> b
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g.argmax(b)
        (array([1 2 3 4]), array([4 0 9 1]))


        '''
        ...

    def argmin(self, values: 'pdarray') -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and return the location of the first minimum of each group.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and find argmin

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_argminima : pdarray, int64
                One index per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object or if argmax
            is not supported for the values dtype
        ValueError
            Raised if the key array size does not match the values
            size or if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if argmin is not supported for the values dtype

        Notes
        -----
        The returned indices refer to the original values array as
        passed in, not the permutation applied by the GroupBy instance.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4 3])
        >>> b = ak.randint(1, 5, 10, seed=1)
        >>> b
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g.argmin(b)
        (array([1 2 3 4]), array([4 0 9 1]))


        '''
        ...

    def broadcast(self, values: 'Union[pdarray, Strings]', permute: 'bool' = True) -> 'Union[pdarray, Strings]':
        r'''

        Fill each group's segment with a constant value.

        Parameters
        ----------
        values : pdarray, Strings
            The values to put in each group's segment
        permute : bool
            If True (default), permute broadcast values back to the ordering
            of the original array on which GroupBy was called. If False, the
            broadcast values are grouped by value.

        Returns
        -------
        pdarray, Strings
            The broadcasted values

        Raises
        ------
        TypeError
            Raised if value is not a pdarray object
        ValueError
            Raised if the values array does not have one
            value per segment

        Notes
        -----
        This function is a sparse analog of ``np.broadcast``. If a
        GroupBy object represents a sparse matrix (tensor), then
        this function takes a (dense) column vector and replicates
        each value to the non-zero elements in the corresponding row.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.array([0, 1, 0, 1, 0])
        >>> values = ak.array([3, 5])
        >>> g = ak.GroupBy(a)

        By default, result is in original order
        >>> g.broadcast(values)
        array([3 5 3 5 3])

        With permute=False, result is in grouped order
        >>> g.broadcast(values, permute=False)
        array([3 3 3 5 5])
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> keys,counts = g.size()
        >>> g.broadcast(counts > 2)
        array([True True True True False True False True True False])
        >>> g.broadcast(counts == 3)
        array([True False False True False False False True False False])
        >>> g.broadcast(counts < 4)
        array([True False False True True False True True False True])


        '''
        ...

    def build_from_components(self, user_defined_name: 'Optional[str]' = None, **kwargs) -> 'GroupBy':
        r'''

        Build a new GroupBy object from component keys and permutation.

        Parameters
        ----------
        user_defined_name : str (Optional) Passing a name will init the new GroupBy
                and assign it the given name
        kwargs : dict Dictionary of components required for rebuilding the GroupBy.
                Expected keys are "orig_keys", "permutation", "unique_keys", and "segments"

        Returns
        -------
        GroupBy
            The GroupBy object created by using the given components


        '''
        ...

    def count(self, values: 'pdarray') -> 'Tuple[groupable, pdarray]':
        r'''

        Count the number of elements in each group.

        NaN values will be excluded from the total.

        Parameters
        ----------
        values: pdarray
            The values to be count by group (excluding NaN values).

        Returns
        -------
        List[pdarray|Strings], pdarray|int64
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            counts : pdarray, int64
                The number of times each unique key appears (excluding NaN values).

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.array([1, 0, -1, 1, 0, -1])
        >>> a
        array([1 0 -1 1 0 -1])
        >>> b = ak.array([1, np.nan, -1, np.nan, np.nan, -1], dtype = "float64")
        >>> b
        array([1.00000000000000000 nan -1.00000000000000000 nan nan -1.00000000000000000])
        >>> g = ak.GroupBy(a)
        >>> keys,counts = g.count(b)
        >>> keys
        array([-1 0 1])
        >>> counts
        array([2 0 1])


        '''
        ...

    def first(self, values: 'groupable_element_type') -> 'Tuple[groupable, groupable_element_type]':
        r'''

        First value in each group.

        Parameters
        ----------
        values : pdarray-like
            The values from which to take the first of each group

        Returns
        -------
        Tuple[groupable, groupable_element_type]
            unique_keys : (list of) pdarray-like
                The unique keys, in grouped order
            result : pdarray-like
                The first value of each group


        '''
        ...

    def from_return_msg(self, rep_msg):
        r'''

        Reconstruct a GroupBy object from a server return message.

        This is used to deserialize a GroupBy object that was previously saved or transferred.

        Parameters
        ----------
        rep_msg : str
            A JSON-formatted string containing GroupBy components returned by the Arkouda server.

        Returns
        -------
        GroupBy
            A reconstructed GroupBy object.

        Raises
        ------
        ValueError
            If an unsupported or unknown data type is encountered during reconstruction.


        '''
        ...

    def head(self, values: 'groupable_element_type', n: 'int' = 5, return_indices: 'bool' = True) -> 'Tuple[groupable, groupable_element_type]':
        r'''

        Return the first n values from each group.

        Parameters
        ----------
        values : (list of) pdarray-like
            The values from which to select, according to their group membership.
        n: int, optional, default = 5
            Maximum number of items to return for each group.
            If the number of values in a group is less than n,
            all the values from that group will be returned.
        return_indices: bool, default False
            If True, return the indices of the sampled values.
            Otherwise, return the selected values.

        Returns
        -------
        Tuple[groupable, groupable_element_type]
            unique_keys : (list of) pdarray-like
                The unique keys, in grouped order
            result : pdarray-like
                The first n items of each group.
                If return_indices is True, the result are indices.
                O.W. the result are values.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.arange(10) %3
        >>> a
        array([0 1 2 0 1 2 0 1 2 0])
        >>> v = ak.arange(10)
        >>> v
        array([0 1 2 3 4 5 6 7 8 9])
        >>> g = GroupBy(a)
        >>> unique_keys, idx = g.head(v, 2, return_indices=True)
        >>> _, values = g.head(v, 2, return_indices=False)
        >>> unique_keys
        array([0 1 2])
        >>> idx
        array([0 3 1 4 2 5])
        >>> values
        array([0 3 1 4 2 5])

        >>> v2 =  -2 * ak.arange(10)
        >>> v2
        array([0 -2 -4 -6 -8 -10 -12 -14 -16 -18])
        >>> _, idx2 = g.head(v2, 2, return_indices=True)
        >>> _, values2 = g.head(v2, 2, return_indices=False)
        >>> idx2
        array([0 3 1 4 2 5])
        >>> values2
        array([0 -6 -2 -8 -4 -10])


        '''
        ...

    def is_registered(self) -> 'bool':
        r'''

        Return True if the object is contained in the registry.

        Returns
        -------
        bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mismatch of registered components

        See Also
        --------
        register, attach, unregister, unregister_groupby_by_name

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.


        '''
        ...

    def max(self, values: 'pdarray', skipna: 'bool' = True) -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and return the maximum of each group's values.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and find maxima
        skipna: bool
            boolean which determines if NANs should be skipped

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_maxima : pdarray
                One maximum per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object or if max is
            not supported for the values dtype
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if max is not supported for the values dtype

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4 3])
        >>> b = ak.randint(1, 5, 10, seed=1)
        >>> b
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g.max(b)
        (array([1 2 3 4]), array([1 2 3 4]))


        '''
        ...

    def mean(self, values: 'pdarray', skipna: 'bool' = True) -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and compute the mean of each group's values.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and average
        skipna: bool
            boolean which determines if NANs should be skipped

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_means : pdarray, float64
                One mean value per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size
            or if the operator is not in the GroupBy.Reductions array

        Notes
        -----
        The return dtype is always float64.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4 3])
        >>> b = ak.randint(1, 5, 10, seed=1)
        >>> b
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g.mean(b)
        (array([1 2 3 4]),
        array([1.00000000000000000 2.00000000000000000 3.00000000000000000 4.00000000000000000]))


        '''
        ...

    def median(self, values: 'pdarray', skipna: 'bool' = True) -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and compute the median of each group's values.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and find median
        skipna: bool
            boolean which determines if NANs should be skipped

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_medians : pdarray, float64
                One median value per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size
            or if the operator is not in the GroupBy.Reductions array

        Notes
        -----
        The return dtype is always float64.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 9, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4])
        >>> b = ak.linspace(-5, 5, 9)
        >>> b
        array([-5.00000000000000000 -3.75 -2.5 -1.25 0.00000000000000000
            1.25 2.5 3.75 5.00000000000000000])
        >>> g.median(b)
        (array([1 2 4]), array([1.25 -1.25 -0.625]))


        '''
        ...

    def min(self, values: 'pdarray', skipna: 'bool' = True) -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and return the minimum of each group's values.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and find minima
        skipna: bool
            boolean which determines if NANs should be skipped

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_minima : pdarray
                One minimum per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object or if min is
            not supported for the values dtype
        ValueError
            Raised if the key array size does not match the values size
            or if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if min is not supported for the values dtype

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4 3])
        >>> b = ak.randint(1, 5, 10, seed=1)
        >>> b
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g.min(b)
        (array([1 2 3 4]), array([1 2 3 4]))


        '''
        ...

    def mode(self, values: 'groupable') -> 'Tuple[groupable, groupable]':
        r'''

        Return the most common value in each group.

        If a group is multi-modal, return the
        modal value that occurs first.

        Parameters
        ----------
        values : (list of) pdarray-like
            The values from which to take the mode of each group

        Returns
        -------
        Tuple[groupable, groupable]
            unique_keys : (list of) pdarray-like
                The unique keys, in grouped order
            result : (list of) pdarray-like
                The most common value of each group


        '''
        ...

    def nunique(self, values: 'groupable') -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and return the number of unique values in each group.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray, int64
            The values to group and find unique values

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : groupable
                The unique keys, in grouped order
            group_nunique : groupable
                Number of unique values per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the dtype(s) of values array(s) does/do not support
            the nunique method
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if nunique is not supported for the values dtype

        Examples
        --------
        >>> import arkouda as ak
        >>> data = ak.array([3, 4, 3, 1, 1, 4, 3, 4, 1, 4])
        >>> data
        array([3 4 3 1 1 4 3 4 1 4])
        >>> labels = ak.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        >>> labels
        array([1 1 1 2 2 2 3 3 3 4])
        >>> g = ak.GroupBy(labels)
        >>> g.keys
        array([1 1 1 2 2 2 3 3 3 4])
        >>> g.nunique(data)
        (array([1 2 3 4]), array([2 2 3 1]))

        Group (1,1,1) has values [3,4,3] -> there are 2 unique values 3&4
        Group (2,2,2) has values [1,1,4] -> 2 unique values 1&4
        Group (3,3,3) has values [3,4,1] -> 3 unique values
        Group (4) has values [4] -> 1 unique value


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

    def prod(self, values: 'pdarray', skipna: 'bool' = True) -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and compute the product of each group's values.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and multiply
        skipna: bool
            boolean which determines if NANs should be skipped

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_products : pdarray, float64
                One product per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size
            or if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if prod is not supported for the values dtype

        Notes
        -----
        The return dtype is always float64.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4 3])
        >>> b = ak.randint(1, 5, 10, seed=1)
        >>> b
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g.prod(b)
        (array([1 2 3 4]),
        array([1.00000000000000000 7.9999999999999982 3.0000000000000004 255.99999999999994]))


        '''
        ...

    def register(self, user_defined_name: 'str') -> 'GroupBy':
        r'''

        Register this GroupBy object and underlying components with the Arkouda server.

        Parameters
        ----------
        user_defined_name : str
            user defined name the GroupBy is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        GroupBy
            The same GroupBy which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support a
            fluid programming style.
            Please note you cannot register two different GroupBys with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the GroupBy with the user_defined_name

        See Also
        --------
        unregister, attach, unregister_groupby_by_name, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.


        '''
        ...

    def sample(self, values: 'groupable', n=None, frac=None, replace=False, weights=None, random_state=None, return_indices=False, permute_samples=False):
        r'''

        Return a random sample from each group.

        You can either specify the number of elements
        or the fraction of elements to be sampled. random_state can be used for reproducibility

        Parameters
        ----------
        values : (list of) pdarray-like
            The values from which to sample, according to their group membership.

        n: int, optional
            Number of items to return for each group.
            Cannot be used with frac and must be no larger than
            the smallest group unless replace is True.
            Default is one if frac is None.

        frac: float, optional
            Fraction of items to return. Cannot be used with n.

        replace: bool, default False
            Allow or disallow sampling of the value more than once.

        weights: pdarray, optional
            Default None results in equal probability weighting.
            If passed a pdarray, then values must have the same length as the groupby keys
            and will be used as sampling probabilities after normalization within each group.
            Weights must be non-negative with at least one positive element within each group.

        random_state: int or ak.random.Generator, optional
            If int, seed for random number generator.
            If ak.random.Generator, use as given.

        return_indices: bool, default False
            if True, return the indices of the sampled values.
            Otherwise, return the sample values.

        permute_samples: bool, default False
            if True, return permute the samples according to group
            Otherwise, keep samples in original order.

        Returns
        -------
        pdarray
            if return_indices is True, return the indices of the sampled values.
            Otherwise, return the sample values.


        '''
        ...

    def size(self) -> 'Tuple[groupable, pdarray]':
        r'''

        Count the number of elements in each group, i.e. the number of times each key appears.

        This counts the total number of rows (including NaN values).

        Returns
        -------
        List[pdarray|Strings], pdarray|int64
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            counts : pdarray, int64
                The number of times each unique key appears

        See Also
        --------
        count

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> keys,counts = g.size()
        >>> keys
        array([1 2 3 4])
        >>> counts
        array([2 3 1 4])


        '''
        ...

    def std(self, values: 'pdarray', skipna: 'bool' = True, ddof: 'int_scalars' = 1) -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and compute the standard deviation of each group's values.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and find standard deviation
        skipna: bool
            boolean which determines if NANs should be skipped
        ddof : int_scalars
            "Delta Degrees of Freedom" used in calculating std

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_stds : pdarray, float64
                One std value per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size
            or if the operator is not in the GroupBy.Reductions array

        Notes
        -----
        The return dtype is always float64.

        The standard deviation is the square root of the average of the squared
        deviations from the mean, i.e., ``std = sqrt(mean((x - x.mean())**2))``.

        The average squared deviation is normally calculated as
        ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
        the divisor ``N - ddof`` is used instead. In standard statistical
        practice, ``ddof=1`` provides an unbiased estimator of the variance
        of the infinite population. ``ddof=0`` provides a maximum likelihood
        estimate of the variance for normally distributed variables. The
        standard deviation computed in this function is the square root of
        the estimated variance, so even with ``ddof=1``, it will not be an
        unbiased estimate of the standard deviation per se.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4 3])
        >>> b = ak.randint(1, 5, 10, seed=1)
        >>> b
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g.std(b)
        (array([1 2 3 4]), array([0.00000000000000000 0.00000000000000000 nan 0.00000000000000000]))


        '''
        ...

    def sum(self, values: 'pdarray', skipna: 'bool' = True) -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and sum each group's values.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and sum
        skipna: bool
            boolean which determines if NANs should be skipped


        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_sums : pdarray
                One sum per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array

        Notes
        -----
        The grouped sum of a boolean ``pdarray`` returns integers.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4 3])
        >>> b = ak.randint(1, 5, 10, seed=1)
        >>> b
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g.sum(b)
        (array([1 2 3 4]), array([2 6 3 16]))


        '''
        ...

    def tail(self, values: 'groupable_element_type', n: 'int' = 5, return_indices: 'bool' = True) -> 'Tuple[groupable, groupable_element_type]':
        r'''

        Return the last n values from each group.

        Parameters
        ----------
        values : (list of) pdarray-like
            The values from which to select, according to their group membership.
        n: int, optional, default = 5
            Maximum number of items to return for each group.
            If the number of values in a group is less than n,
            all the values from that group will be returned.
        return_indices: bool, default False
            If True, return the indices of the sampled values.
            Otherwise, return the selected values.

        Returns
        -------
        Tuple[groupable, groupable_element_type]
            unique_keys : (list of) pdarray-like
                The unique keys, in grouped order
            result : pdarray-like
                The last n items of each group.
                If return_indices is True, the result are indices.
                O.W. the result are values.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.arange(10) %3
        >>> a
        array([0 1 2 0 1 2 0 1 2 0])
        >>> v = ak.arange(10)
        >>> v
        array([0 1 2 3 4 5 6 7 8 9])
        >>> g = GroupBy(a)
        >>> unique_keys, idx = g.tail(v, 2, return_indices=True)
        >>> _, values = g.tail(v, 2, return_indices=False)
        >>> unique_keys
        array([0 1 2])
        >>> idx
        array([6 9 4 7 5 8])
        >>> values
        array([6 9 4 7 5 8])

        >>> v2 =  -2 * ak.arange(10)
        >>> v2
        array([0 -2 -4 -6 -8 -10 -12 -14 -16 -18])
        >>> _, idx2 = g.tail(v2, 2, return_indices=True)
        >>> _, values2 = g.tail(v2, 2, return_indices=False)
        >>> idx2
        array([6 9 4 7 5 8])
        >>> values2
        array([-12 -18 -8 -14 -10 -16])


        '''
        ...

    def to_hdf(self, prefix_path, dataset='groupby', mode='truncate', file_type='distribute'):
        r'''

        Save the GroupBy to HDF5.

        The result is a collection of HDF5 files, one file
        per locale of the arkouda server, where each filename starts with prefix_path.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files will share
        dataset : str
            Name prefix for saved data within the HDF5 file
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', add data as a new column to existing files.
        file_type: str ("single" | "distribute")
            Default: "distribute"
            When set to single, dataset is written to a single file.
            When distribute, dataset is written on a file per locale.
            This is only supported by HDF5 files and will have no impact of Parquet Files.

        Notes
        -----
        GroupBy is not currently supported by Parquet


        '''
        ...

    def unique(self, values: 'groupable'):
        r'''

        Return the set of unique values in each group, as a SegArray.

        Parameters
        ----------
        values : (list of) pdarray-like
            The values to unique

        Returns
        -------
        (list of) pdarray-like, (list of) SegArray
            unique_keys : (list of) pdarray-like
                The unique keys, in grouped order
            result : (list of) SegArray
                The unique values of each group

        Raises
        ------
        TypeError
            Raised if values is or contains Strings or Categorical


        '''
        ...

    def unregister(self):
        r'''

        Unregister this GroupBy object.

        Unregister this GroupBy object in the arkouda server which was previously
        registered using register() and/or attached to using attach().

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See Also
        --------
        register, attach, unregister_groupby_by_name, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.


        '''
        ...

    def update_hdf(self, prefix_path: 'str', dataset: 'str' = 'groupby', repack: 'bool' = True):
        r'''

        Update an existing HDF5 dataset with new GroupBy data.

        This overwrites the previous GroupBy data stored in HDF5 files and optionally repacks the file.

        Parameters
        ----------
        prefix_path : str
            Path prefix for the HDF5 files to update.
        dataset : str, default="groupby"
            Name of the dataset within the HDF5 file.
        repack : bool, default=True
            Whether to repack the HDF5 file after updating.

        Raises
        ------
        ValueError
            If the file type cannot be determined or required metadata is missing.


        '''
        ...

    def var(self, values: 'pdarray', skipna: 'bool' = True, ddof: 'int_scalars' = 1) -> 'Tuple[groupable, pdarray]':
        r'''

        Group another array of values and compute the variance of each group's values.

        Group using the permutation stored in the GroupBy instance.

        Parameters
        ----------
        values : pdarray
            The values to group and find variance
        skipna: bool
            boolean which determines if NANs should be skipped
        ddof : int_scalars
            "Delta Degrees of Freedom" used in calculating var

        Returns
        -------
        Tuple[groupable, pdarray]
            unique_keys : (list of) pdarray or Strings
                The unique keys, in grouped order
            group_vars : pdarray, float64
                One var value per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size
            or if the operator is not in the GroupBy.Reductions array

        Notes
        -----
        The return dtype is always float64.

        The variance is the average of the squared deviations from the mean,
        i.e.,  ``var = mean((x - x.mean())**2)``.

        The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
        If, however, `ddof` is specified, the divisor ``N - ddof`` is used
        instead.  In standard statistical practice, ``ddof=1`` provides an
        unbiased estimator of the variance of a hypothetical infinite population.
        ``ddof=0`` provides a maximum likelihood estimate of the variance for
        normally distributed variables.

        Examples
        --------
        >>> import arkouda as ak
        >>> a = ak.randint(1, 5, 10, seed=1)
        >>> a
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([2 4 4 2 1 4 1 2 4 3])
        >>> b = ak.randint(1, 5, 10, seed=1)
        >>> b
        array([2 4 4 2 1 4 1 2 4 3])
        >>> g.var(b)
        (array([1 2 3 4]), array([0.00000000000000000 0.00000000000000000 nan 0.00000000000000000]))


        '''
        ...


def broadcast(segments: 'pdarray', values: 'Union[pdarray, Strings]', size: 'Union[int, np.int64, np.uint64]' = -1, permutation: 'Union[pdarray, None]' = None):
    r'''

    Broadcast a dense column vector to the rows of a sparse matrix or grouped array.

    Parameters
    ----------
    segments : pdarray, int64
        Offsets of the start of each row in the sparse matrix or grouped array.
        Must be sorted in ascending order.
    values : pdarray, Strings
        The values to broadcast, one per row (or group)
    size : int
        The total number of nonzeros in the matrix. If permutation is given, this
        argument is ignored and the size is inferred from the permutation array.
    permutation : pdarray, int64
        The permutation to go from the original ordering of nonzeros to the ordering
        grouped by row. To broadcast values back to the original ordering, this
        permutation will be inverted. If no permutation is supplied, it is assumed
        that the original nonzeros were already grouped by row. In this case, the
        size argument must be given.

    Returns
    -------
    pdarray, Strings
        The broadcast values, one per nonzero

    Raises
    ------
    ValueError
        - If segments and values are different sizes
        - If segments are empty
        - If number of nonzeros (either user-specified or inferred from permutation)
          is less than one

    Examples
    --------
    >>> import arkouda as ak
    >>>
    # Define a sparse matrix with 3 rows and 7 nonzeros
    >>> row_starts = ak.array([0, 2, 5])
    >>> nnz = 7

    Broadcast the row number to each nonzero element
    >>> row_number = ak.arange(3)
    >>> ak.broadcast(row_starts, row_number, nnz)
    array([0 0 1 1 1 2 2])

    If the original nonzeros were in reverse order...
    >>> permutation = ak.arange(6, -1, -1)
    >>> ak.broadcast(row_starts, row_number, permutation=permutation)
    array([2 2 1 1 1 0 0])


    '''
    ...


from typing import _NotIterable


class groupable(_NotIterable):

    ...

    def _determine_new_args(self, args):

        ...

    def _getitem(self, parameters):
        r'''
        Union type; Union[X, Y] means either X or Y.

        On Python 3.10 and higher, the | operator
        can also be used to denote unions;
        X | Y means the same thing to the type checker as Union[X, Y].

        To define a union, use e.g. Union[int, str]. Details:
        - The arguments must be types and there must be at least one.
        - None as an argument is a special case and is replaced by
          type(None).
        - Unions of unions are flattened, e.g.::

            assert Union[Union[int, str], float] == Union[int, str, float]

        - Unions of a single argument vanish, e.g.::

            assert Union[int] == int  # The constructor actually returns int

        - Redundant arguments are skipped, e.g.::

            assert Union[int, str, int] == Union[int, str]

        - When comparing unions, the argument order is ignored, e.g.::

            assert Union[int, str] == Union[str, int]

        - You cannot subclass or instantiate a union.
        - You can use Optional[X] as a shorthand for Union[X, None].

        '''
        ...

    def _inst(self, *args, **kwargs):
        r'''
        Returns True when the argument is true, False otherwise.
        The builtins True and False are the only two instances of the class bool.
        The class bool is a subclass of the class int, and cannot be subclassed.
        '''
        ...

    def _make_substitution(self, args, new_arg_by_param):
        r'''
        Create a list of new type arguments.
        '''
        ...

    def _name(self, *args, **kwargs):
        r'''
        The type of the None singleton.
        '''
        ...

    def copy_with(self, params):

        ...


def unique(pda: 'groupable', return_groups: 'bool' = False, assume_sorted: 'bool' = False, return_indices: 'bool' = False) -> 'Union[groupable, Tuple[groupable, pdarray, pdarray, int]]':
    r'''

    Find the unique elements of an array.

    Returns the unique elements of an array, sorted if the values are integers.
    There is an optional output in addition to the unique elements: the number
    of times each unique value comes up in the input array.

    Parameters
    ----------
    pda : (list of) pdarray, Strings, or Categorical
        Input array.
    return_groups : bool, optional
        If True, also return grouping information for the array.
    assume_sorted : bool, optional
        If True, assume pda is sorted and skip sorting step
    return_indices: bool, optional
        Only applicable if return_groups is True.
        If True, return unique key indices along with other groups

    Returns
    -------
    Union[groupable, Tuple[groupable, pdarray, pdarray, int]]
        unique : (list of) pdarray, Strings, or Categorical
            The unique values. If input dtype is int64, return values will be sorted.
        permutation : pdarray, optional
            Permutation that groups equivalent values together (only when return_groups=True)
        segments : pdarray, optional
            The offset of each group in the permuted array (only when return_groups=True)

    Raises
    ------
    TypeError
        Raised if pda is not a pdarray or Strings object
    RuntimeError
        Raised if the pdarray or Strings dtype is unsupported

    Notes
    -----
    For integer arrays, this function checks to see whether `pda` is sorted
    and, if so, whether it is already unique. This step can save considerable
    computation. Otherwise, this function will sort `pda`.

    Examples
    --------
    >>> import arkouda as ak
    >>> A = ak.array([3, 2, 1, 1, 2, 3])
    >>> ak.unique(A)
    array([1 2 3])


    '''
    ...
