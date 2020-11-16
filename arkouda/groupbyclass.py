from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np
from typeguard import typechecked
from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.sorting import argsort, coargsort
from arkouda.strings import Strings
from arkouda.pdarraycreation import array, zeros, arange
from arkouda.pdarraysetops import concatenate
from arkouda.numeric import cumsum
from arkouda.logger import getArkoudaLogger

__all__ = ["GroupBy"]

class GroupBy:
    """
    Group an array or list of arrays by value, usually in preparation 
    for aggregating the within-group values of another array.

    Parameters
    ----------
    keys : (list of) pdarray, int64 or Strings
        The array to group by value, or if list, the column arrays to group by row
    assume_sorted : bool
        If True, assume keys is already sorted (Default: False)
    logger : ArkoudaLogger
        Used for all logging operations

    Attributes
    ----------
    nkeys : int
        The number of key arrays (columns)
    size : int
        The length of the array(s), i.e. number of rows
    permutation : pdarray
        The permutation that sorts the keys array(s) by value (row)
    unique_keys : (list of) pdarray or Strings
        The unique values of the keys array(s), in grouped order
    segments : pdarray
        The start index of each group in the grouped array(s)
    unique_key_indices : pdarray
        The first index in the raw (ungrouped) keys array(s) where each 
        unique value (row) occurs

    Notes
    -----
    Only accepts pdarrays of int64 dtype or Strings.

    """
    Reductions = frozenset(['sum', 'prod', 'mean',
                            'min', 'max', 'argmin', 'argmax',
                            'nunique', 'any', 'all'])
    def __init__(self, keys : List[Union[pdarray,np.int64,Strings]], 
                assume_sorted : bool=False, hash_strings : bool=True) -> None:
        self.logger = getArkoudaLogger(name=self.__class__.__name__)
        self.assume_sorted = assume_sorted
        self.hash_strings = hash_strings
        self.keys = keys
        if isinstance(keys, pdarray):
            self.nkeys = 1
            self.size = keys.size
            if assume_sorted:
                self.permutation = arange(self.size)
            else:
                self.permutation = argsort(keys)
        # for Strings or Categorical
        elif hasattr(keys, "group"):
            self.nkeys = 1
            self.size = keys.size
            if assume_sorted:
                self.permutation = arange(self.size)
            else:
                self.permutation = keys.group()
        else:
            self.nkeys = len(keys)
            self.size = keys[0].size
            for k in keys:
                if k.size != self.size:
                    raise ValueError("Key arrays must all be same size")
            if assume_sorted:
                self.permutation = arange(self.size)
            else:
                self.permutation = coargsort(keys)
            
        # self.permuted_keys = self.keys[self.permutation]
        self.find_segments()       
            
    def find_segments(self) -> None:
        cmd = "findSegments"
        if self.nkeys == 1:
            # for Categorical
            if hasattr(self.keys, 'segments') and self.keys.segments is not None:
                self.unique_keys = self.keys.categories
                self.segments = self.keys.segments
                return
            else:
                mykeys = [self.keys]            
        else:
            mykeys = self.keys
        keyobjs = [] # needed to maintain obj refs esp for h1 and h2 in the strings case
        keynames = []
        keytypes = []
        effectiveKeys = self.nkeys
        for k in mykeys:
            if isinstance(k, Strings):
                if self.hash_strings:
                    h1, h2 = k.hash()
                    keyobjs.extend([h1,h2])
                    keynames.extend([h1.name, h2.name])
                    keytypes.extend([h1.objtype, h2.objtype])
                    effectiveKeys += 1
                else:
                    keyobjs.append(k)
                    keynames.append('{}+{}'.format(k.offsets.name, 
                                                   k.bytes.name))
                    keytypes.append(k.objtype)
            # for Categorical
            elif hasattr(k, 'codes'):
                keyobjs.append(k)
                keynames.append(k.codes.name)
                keytypes.append(k.codes.objtype)
            elif isinstance(k, pdarray):
                keyobjs.append(k)
                keynames.append(k.name)
                keytypes.append(k.objtype)
        reqMsg = "{} {} {:n} {} {}".format(cmd,
                                             self.permutation.name,
                                             effectiveKeys,
                                             ' '.join(keynames),
                                             ' '.join(keytypes))
        repMsg = generic_msg(reqMsg)
        segAttr, uniqAttr = repMsg.split("+")
        self.logger.debug('{},{}'.format(segAttr, uniqAttr))
        self.segments = create_pdarray(segAttr)
        unique_key_indices = create_pdarray(uniqAttr)
        if self.nkeys == 1:
            self.unique_keys = self.keys[unique_key_indices]
        else:
            self.unique_keys = [k[unique_key_indices] for k in self.keys]


    def count(self) -> Tuple[List[Union[pdarray,Strings]],pdarray]:
        '''
        Count the number of elements in each group, i.e. the number of times
        each key appears.

        Parameters
        ----------
        none

        Returns
        -------
        unique_keys : (list of) pdarray or Strings
            The unique keys, in grouped order
        counts : pdarray, int64
            The number of times each unique key appears
        
        '''
        cmd = "countReduction"
        reqMsg = "{} {} {}".format(cmd, self.segments.name, self.size)
        repMsg = generic_msg(reqMsg)
        self.logger.debug(repMsg)
        return self.unique_keys, create_pdarray(repMsg)
    
    @typechecked
    def aggregate(self, values : pdarray, operator : str, skipna : bool=True) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        '''
        Using the permutation stored in the GroupBy instance, group another 
        array of values and apply a reduction to each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and reduce
        operator: str
            The name of the reduction operator to use

        Returns
        -------
        unique_keys : [Union[pdarray,List[Union[pdarray,Strings]]]
            The unique keys, in grouped order
        aggregates : pdarray
            One aggregate value per unique key in the GroupBy instance
            
        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array

        '''
        if values.size != self.size:
            raise ValueError(("Attempt to group array using key array of " +
                             "different length"))
        if operator not in self.Reductions:
            raise ValueError(("Unsupported reduction: {}\nMust be one of {}")\
                                  .format(operator, self.Reductions))
        if self.assume_sorted:
            permuted_values = values
        else:
            permuted_values = values[self.permutation]
        cmd = "segmentedReduction"
        reqMsg = "{} {} {} {} {}".format(cmd,
                                         permuted_values.name,
                                         self.segments.name,
                                         operator,
                                         skipna)
        repMsg = generic_msg(reqMsg)
        self.logger.debug(repMsg)
        if operator.startswith('arg'):
            return (self.unique_keys, 
                              self.permutation[create_pdarray(repMsg)])
        else:
            return self.unique_keys, create_pdarray(repMsg)

    def sum(self, values : pdarray, skipna : bool=True) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group 
        another array of values and sum each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and sum

        Returns
        -------
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
        """
        return self.aggregate(values, "sum", skipna)
    
    def prod(self, values : pdarray, skipna : bool=True) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group
        another array of values and compute the product of each group's 
        values. 

        Parameters
        ----------
        values : pdarray
            The values to group and multiply

        Returns
        -------
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

        Notes
        -----
        The return dtype is always float64.
        """
        return self.aggregate(values, "prod", skipna)
    
    def mean(self, values : pdarray, skipna : bool=True) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group 
        another array of values and compute the mean of each group's 
        values. 

        Parameters
        ----------
        values : pdarray
            The values to group and average

        Returns
        -------
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
        """
        return self.aggregate(values, "mean", skipna)
    
    def min(self, values : pdarray, skipna : bool=True) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group 
        another array of values and return the minimum of each group's 
        values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find minima

        Returns
        -------
        unique_keys : (list of) pdarray or Strings
            The unique keys, in grouped order
        group_minima : pdarray
            One minimum per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size
            or if the operator is not in the GroupBy.Reductions array

        """
        return self.aggregate(values, "min", skipna)
    
    def max(self, values : pdarray, skipna : bool=True) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group
        another array of values and return the maximum of each 
        group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find maxima

        Returns
        -------
        unique_keys : (list of) pdarray or Strings
            The unique keys, in grouped order
        group_maxima : pdarray
            One maximum per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array

        """
        return self.aggregate(values, "max", skipna)
    
    def argmin(self, values : pdarray) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group   
        another array of values and return the location of the first 
        minimum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find argmin

        Returns
        -------
        unique_keys : (list of) pdarray or Strings
            The unique keys, in grouped order
        group_argminima : pdarray, int64
            One index per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values
            size or if the operator is not in the GroupBy.Reductions 
            array

        Notes
        -----
        The returned indices refer to the original values array as
        passed in, not the permutation applied by the GroupBy instance.

        Examples
        --------
        >>> A = ak.array([0, 1, 0, 1, 0, 1])
        >>> B = ak.array([0, 1, 1, 0, 0, 1])
        >>> byA = ak.GroupBy(A)
        >>> byA.argmin(B)
        (array([0, 1]), array([0, 3]))
        """
        return self.aggregate(values, "argmin")
    
    def argmax(self, values : pdarray)\
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group   
        another array of values and return the location of the first 
        maximum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find argmax

        Returns
        -------
        unique_keys : (list of) pdarray or Strings
            The unique keys, in grouped order
        group_argmaxima : pdarray, int64
            One index per unique key in the GroupBy instance

        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array

        Notes
        -----
        The returned indices refer to the original values array as passed in,
        not the permutation applied by the GroupBy instance.

        Examples
        --------
        >>> A = ak.array([0, 1, 0, 1, 0, 1])
        >>> B = ak.array([0, 1, 1, 0, 0, 1])
        >>> byA = ak.GroupBy(A)
        >>> byA.argmax(B)
        (array([0, 1]), array([2, 1]))
        """
        return self.aggregate(values, "argmax")
    
    def nunique(self, values : pdarray) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group another
        array of values and return the number of unique values in each group. 

        Parameters
        ----------
        values : pdarray, int64
            The values to group and find unique values

        Returns
        -------
        unique_keys : (list of) pdarray or Strings
            The unique keys, in grouped order
        group_nunique : pdarray, int64
            Number of unique values per unique key in the GroupBy instance
            
        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        """
        return self.aggregate(values, "nunique")
    
    def any(self, values : pdarray) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group another 
        array of values and perform an "or" reduction on each group. 

        Parameters
        ----------
        values : pdarray, bool
            The values to group and reduce with "or"

        Returns
        -------
        unique_keys : (list of) pdarray or Strings
            The unique keys, in grouped order
        group_any : pdarray, bool
            One bool per unique key in the GroupBy instance
            
        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        """
        return self.aggregate(values, "any")

    def all(self, values : pdarray) \
                    -> Tuple[Union[pdarray,List[Union[pdarray,Strings]]],pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group  
        another array of values and perform an "and" reduction on 
        each group. 

        Parameters
        ----------
        values : pdarray, bool
            The values to group and reduce with "and"

        Returns
        -------
        unique_keys : (list of) pdarray or Strings
            The unique keys, in grouped order
        group_any : pdarray, bool
            One bool per unique key in the GroupBy instance
            
        Raises
        ------
        TypeError
            Raised if the values array is not a pdarray object
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        """
        return self.aggregate(values, "all")

    @typechecked
    def broadcast(self, values : pdarray) -> pdarray:
        """
        Fill each group's segment with a constant value.

        Parameters
        ----------
        values : pdarray
            The values to put in each group's segment

        Returns
        -------
        pdarray
            The broadcast values
            
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

        The returned array is in permuted (grouped) order. To get
        back to the order of the array on which GroupBy was called,
        the user must invert the permutation (see below).

        Examples
        --------
        >>> a = ak.array([0, 1, 0, 1, 0])
        >>> values = ak.array([3, 5])
        >>> g = ak.GroupBy(a)
        # Result is in grouped order
        >>> g.broadcast(values)
        array([3, 3, 3, 5, 5]

        >>> b = ak.zeros_like(a)
        # Result is in original order
        >>> b[g.permutation] = g.broadcast(values)
        >>> b
        array([3, 5, 3, 5, 3])
        """
        if values.size != self.segments.size:
            raise ValueError("Must have one value per segment")
        temp = zeros(self.size, values.dtype)
        if values.size == 0:
            return temp
        diffs = concatenate((array([values[0]]), values[1:] - values[:-1]))
        temp[self.segments] = diffs
        return cumsum(temp)

   
