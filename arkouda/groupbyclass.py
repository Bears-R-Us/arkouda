from arkouda.client import generic_msg, verbose
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.sorting import argsort, coargsort, local_argsort

__all__ = ["GroupBy"]

class GroupBy:
    """
    Group an array by value, usually in preparation for aggregating the 
    within-group values of another array.

    Parameters
    ----------
    keys : (list of) pdarray, int64
        The array to group by value, or if list, the column arrays to group by row

    Attributes
    ----------
    nkeys : int
        The number of key arrays (columns)
    size : int
        The length of the array(s), i.e. number of rows
    permutation : pdarray
        The permutation that sorts the keys array(s) by value (row)
    unique_keys : pdarray
        The unique values of the keys array(s), in cosorted order
    segments : pdarray
        The start index of each group in the sorted array(s)
    unique_key_indices : pdarray
        The first index in the unsorted keys array(s) where each unique value (row) occurs

    Notes
    -----
    Only accepts pdarrays of int64 dtype.

    """
    Reductions = frozenset(['sum', 'prod', 'mean',
                            'min', 'max', 'argmin', 'argmax',
                            'nunique', 'any', 'all'])
    def __init__(self, keys):    
        self.per_locale = False
        self.keys = keys
        if isinstance(keys, pdarray):
            self.nkeys = 1
            self.size = keys.size
            if self.per_locale:
                self.permutation = local_argsort(keys)
            else:
                self.permutation = argsort(keys)
        else:
            self.nkeys = len(keys)
            self.size = keys[0].size
            for k in keys:
                if k.size != self.size:
                    raise ValueError("Key arrays must all be same size")
            self.permutation = coargsort(keys)
            
        # self.permuted_keys = self.keys[self.permutation]
        self.find_segments()
            
    def find_segments(self):
        if self.per_locale:
            cmd = "findLocalSegments"
        else:
            cmd = "findSegments"
        if self.nkeys == 1:
            keynames = self.keys.name
        else:
            keynames = ' '.join([k.name for k in self.keys])
        reqMsg = "{} {} {:n} {:n} {}".format(cmd,
                                             self.permutation.name,
                                             self.nkeys,
                                             self.size,
                                             keynames)
        repMsg = generic_msg(reqMsg)
        segAttr, uniqAttr = repMsg.split("+")
        if verbose: print(segAttr, uniqAttr)
        self.segments = create_pdarray(segAttr)
        self.unique_key_indices = create_pdarray(uniqAttr)
        if self.nkeys == 1:
            self.unique_keys = self.keys[self.unique_key_indices]
        else:
            self.unique_keys = [k[self.unique_key_indices] for k in self.keys]


    def count(self):
        '''
        Count the number of elements in each group, i.e. the number of times
        each key appears.

        Parameters
        ----------
        none

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        counts : pdarray, int64
            The number of times each unique key appears
        
        '''
        if self.per_locale:
            cmd = "countLocalRdx"
        else:
            cmd = "countReduction"
        reqMsg = "{} {} {}".format(cmd, self.segments.name, self.size)
        repMsg = generic_msg(reqMsg)
        if verbose: print(repMsg)
        return self.unique_keys, create_pdarray(repMsg)
        
    def aggregate(self, values, operator):
        '''
        Using the permutation stored in the GroupBy instance, group another array 
        of values and apply a reduction to each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and reduce
        operator: str
            The name of the reduction operator to use

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        aggregates : pdarray
            One aggregate value per unique key in the GroupBy instance

        '''
        if not isinstance(values, pdarray):
            raise TypeError("<values> must be a pdarray")
        if values.size != self.size:
            raise ValueError("Attempt to group array using key array of different length")
        if operator not in self.Reductions:
            raise ValueError("Unsupported reduction: {}\nMust be one of {}".format(operator, self.Reductions))
        permuted_values = values[self.permutation]
        if self.per_locale:
            cmd = "segmentedLocalRdx"
        else:
            cmd = "segmentedReduction"
        reqMsg = "{} {} {} {}".format(cmd,
                                         permuted_values.name,
                                         self.segments.name,
                                         operator)
        repMsg = generic_msg(reqMsg)
        if verbose: print(repMsg)
        if operator.startswith('arg'):
            return self.unique_keys, self.permutation[create_pdarray(repMsg)]
        else:
            return self.unique_keys, create_pdarray(repMsg)

    def sum(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and sum each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and sum

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_sums : pdarray
            One sum per unique key in the GroupBy instance

        Notes
        -----
        The grouped sum of a boolean ``pdarray`` returns integers.
        """
        return self.aggregate(values, "sum")
    
    def prod(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and compute the product of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and multiply

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_products : pdarray, float64
            One product per unique key in the GroupBy instance

        Notes
        -----
        The return dtype is always float64.
        """
        return self.aggregate(values, "prod")
    
    def mean(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and compute the mean of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and average

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_means : pdarray, float64
            One mean value per unique key in the GroupBy instance

        Notes
        -----
        The return dtype is always float64.
        """
        return self.aggregate(values, "mean")
    
    def min(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the minimum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find minima

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_minima : pdarray
            One minimum per unique key in the GroupBy instance

        """
        return self.aggregate(values, "min")
    
    def max(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the maximum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find maxima

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_maxima : pdarray
            One maximum per unique key in the GroupBy instance

        """
        return self.aggregate(values, "max")
    
    def argmin(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the location of the first minimum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find argmin

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_argminima : pdarray, int64
            One index per unique key in the GroupBy instance

        Notes
        -----
        The returned indices refer to the original values array as passed in, not
        the permutation applied by the GroupBy instance.

        Examples
        --------
        >>> A = ak.array([0, 1, 0, 1, 0, 1])
        >>> B = ak.array([0, 1, 1, 0, 0, 1])
        >>> byA = ak.GroupBy(A)
        >>> byA.argmin(B)
        (array([0, 1]), array([0, 3]))
        """
        return self.aggregate(values, "argmin")
    
    def argmax(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the location of the first maximum of each group's values. 

        Parameters
        ----------
        values : pdarray
            The values to group and find argmax

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_argmaxima : pdarray, int64
            One index per unique key in the GroupBy instance

        Notes
        -----
        The returned indices refer to the original values array as passed in, not
        the permutation applied by the GroupBy instance.

        Examples
        --------
        >>> A = ak.array([0, 1, 0, 1, 0, 1])
        >>> B = ak.array([0, 1, 1, 0, 0, 1])
        >>> byA = ak.GroupBy(A)
        >>> byA.argmax(B)
        (array([0, 1]), array([2, 1]))
        """
        return self.aggregate(values, "argmax")
    
    def nunique(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and return the number of unique values in each group. 

        Parameters
        ----------
        values : pdarray
            The values to group and find unique values

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_nunique : pdarray, int64
            Number of unique values per unique key in the GroupBy instance
        """
        return self.aggregate(values, "nunique")
    
    def any(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and perform an "or" reduction on each group. 

        Parameters
        ----------
        values : pdarray
            The values to group and reduce with "or"

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_any : pdarray, bool
            One bool per unique key in the GroupBy instance
        """
        return self.aggregate(values, "any")
    
    def all(self, values):
        """
        Using the permutation stored in the GroupBy instance, group another array 
        of values and perform an "and" reduction on each group. 

        Parameters
        ----------
        values : pdarray
            The values to group and reduce with "and"

        Returns
        -------
        unique_keys : pdarray, int64
            The unique keys, in sorted order
        group_any : pdarray, bool
            One bool per unique key in the GroupBy instance
        """
        return self.aggregate(values, "all")
