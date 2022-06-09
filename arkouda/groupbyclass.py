from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union, cast

if TYPE_CHECKING:
    from arkouda.categorical import Categorical

import numpy as np  # type: ignore
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import uint64 as akuint64
from arkouda.infoclass import list_registry
from arkouda.logger import getArkoudaLogger
from arkouda.pdarrayclass import (
    RegistrationError,
    create_pdarray,
    pdarray,
    unregister_pdarray_by_name,
)
from arkouda.pdarraycreation import arange
from arkouda.strings import Strings

__all__ = ["unique", "GroupBy", "broadcast", "GROUPBY_REDUCTION_TYPES"]

groupable_element_type = Union[pdarray, Strings, "Categorical"]
groupable = Union[groupable_element_type, Sequence[groupable_element_type]]


def unique(
    pda: groupable, return_groups: bool = False, assume_sorted: bool = False  # type: ignore
) -> Union[
    groupable, Tuple[groupable, pdarray, pdarray, int]  # type: ignore
]:  # type: ignore
    """
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

    Returns
    -------
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
    >>> A = ak.array([3, 2, 1, 1, 2, 3])
    >>> ak.unique(A)
    array([1, 2, 3])
    """
    from arkouda.categorical import Categorical as Categorical_

    if not return_groups and hasattr(pda, "unique"):
        return cast(Categorical_, pda).unique()

    # Get all grouping keys
    if hasattr(pda, "_get_grouping_keys"):
        # Single groupable array
        nkeys = 1
        grouping_keys = cast(
            list, cast(groupable_element_type, pda)._get_grouping_keys()
        )
    else:
        # Sequence of groupable arrays
        nkeys = len(pda)
        grouping_keys = []
        first = True
        for k in pda:
            if first:
                size = k.size
                first = False
            elif k.size != size:
                raise ValueError("Key arrays must all be same size")
            if not hasattr(k, "_get_grouping_keys"):
                raise TypeError(f"{type(k)} does not support grouping")
            grouping_keys.extend(cast(list, k._get_grouping_keys()))
    keynames = [k.name for k in grouping_keys]
    keytypes = [k.objtype for k in grouping_keys]
    effectiveKeys = len(grouping_keys)
    repMsg = generic_msg(
        cmd="unique",
        args="{} {} {:n} {} {}".format(
            return_groups,
            assume_sorted,
            effectiveKeys,
            " ".join(keynames),
            " ".join(keytypes),
        ),
    )
    if return_groups:
        parts = cast(str, repMsg).split("+")
        permutation = create_pdarray(cast(str, parts[0]))
        segments = create_pdarray(cast(str, parts[1]))
        unique_key_indices = create_pdarray(cast(str, parts[2]))
    else:
        unique_key_indices = create_pdarray(cast(str, repMsg))

    if nkeys == 1:
        unique_keys = pda[unique_key_indices]
    else:
        unique_keys = tuple([a[unique_key_indices] for a in pda])
    if return_groups:
        return (unique_keys, permutation, segments, nkeys)
    else:
        return unique_keys


class GroupByReductionType(enum.Enum):
    SUM = "sum"
    PROD = "prod"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    ARGMIN = "argmin"
    ARGMAX = "argmax"
    NUNUNIQUE = "nunique"
    ANY = "any"
    ALL = "all"
    OR = "or"
    AND = "and"
    XOR = "xor"

    def __str__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a GroupByReductionType as a request parameter
        """
        return self.value

    def __repr__(self) -> str:
        """
        Overridden method returns value, which is useful in outputting
        a GroupByReductionType as a request parameter
        """
        return self.value


GROUPBY_REDUCTION_TYPES = frozenset(
    [member.value for _, member in GroupByReductionType.__members__.items()]
)


class GroupBy:
    """
    Group an array or list of arrays by value, usually in preparation
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
    size : int
        The length of the input array(s), i.e. number of rows
    permutation : pdarray
        The permutation that sorts the keys array(s) by value (row)
    unique_keys : (list of) pdarray, Strings, or Categorical
        The unique values of the keys array(s), in grouped order
    ngroups : int
        The length of the unique_keys array(s), i.e. number of groups
    segments : pdarray
        The start index of each group in the grouped array(s)
    logger : ArkoudaLogger
        Used for all logging operations

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

    """

    Reductions = GROUPBY_REDUCTION_TYPES

    def __init__(
        self,
        keys: Optional[groupable],
        assume_sorted: bool = False,
        hash_strings: bool = True,
        **kwargs,
    ) -> None:
        # Type Checks required because @typechecked was removed for causing other issues
        # This prevents non-bool values that can be evaluated to true (ie non-empty arrays)
        # from causing unexpected results. Experienced when forgetting to wrap multiple key arrays in [].
        # See Issue #1267
        if not isinstance(assume_sorted, bool):
            raise TypeError("assume_sorted must be of type bool.")
        if not isinstance(hash_strings, bool):
            raise TypeError("hash_strings must be of type bool.")

        self.logger = getArkoudaLogger(name=self.__class__.__name__)
        self.assume_sorted = assume_sorted
        self.hash_strings = hash_strings
        self.permutation: pdarray
        self.name: Optional[str] = None

        if (
            "orig_keys" in kwargs
            and "permutation" in kwargs
            and "unique_keys" in kwargs
            and "segments" in kwargs
        ):
            self.keys = cast(groupable, kwargs.get("orig_keys", None))
            self.unique_keys = kwargs.get("unique_keys", None)
            self.permutation = kwargs.get("permutation", None)
            self.segments = kwargs.get("segments", None)
            self.nkeys = len(self.keys)
        elif keys is None:
            raise ValueError("No keys passed to GroupBy.")
        else:
            self.keys = cast(groupable, keys)
            self.unique_keys, self.permutation, self.segments, self.nkeys = unique(  # type: ignore
                self.keys, return_groups=True, assume_sorted=self.assume_sorted
            )

        self.size = self.permutation.size
        self.ngroups = self.segments.size

    def count(self) -> Tuple[groupable, pdarray]:
        """
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

        Examples
        --------
        >>> a = ak.randint(1,5,10)
        >>> a
        array([3, 2, 3, 1, 2, 4, 3, 4, 3, 4])
        >>> g = ak.GroupBy(a)
        >>> keys,counts = g.count()
        >>> keys
        array([1, 2, 3, 4])
        >>> counts
        array([1, 2, 4, 3])
        """
        cmd = "countReduction"
        args = "{} {}".format(cast(pdarray, self.segments).name, self.size)
        repMsg = generic_msg(cmd=cmd, args=args)
        self.logger.debug(repMsg)
        return self.unique_keys, create_pdarray(repMsg)

    def aggregate(
        self, values: groupable, operator: str, skipna: bool = True
    ) -> Tuple[groupable, pdarray]:
        """
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
        >>> keys = ak.arange(0, 10)
        >>> vals = ak.linspace(-1, 1, 10)
        >>> g = ak.GroupBy(keys)
        >>> g.aggregate(vals, 'sum')
        (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([-1, -0.77777777777777768,
        -0.55555555555555536, -0.33333333333333348, -0.11111111111111116,
        0.11111111111111116, 0.33333333333333348, 0.55555555555555536, 0.77777777777777768,
        1]))
        >>> g.aggregate(vals, 'min')
        (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([-1, -0.77777777777777779,
        -0.55555555555555558, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116,
        0.33333333333333326, 0.55555555555555536, 0.77777777777777768, 1]))
        """

        operator = operator.lower()
        if operator not in self.Reductions:
            raise ValueError(
                f"Unsupported reduction: {operator}\nMust be one of {self.Reductions}"
            )

        # TO DO: remove once logic is ported over to Chapel
        if operator == "nunique":
            return self.nunique(values)

        # All other aggregations operate on pdarray
        if cast(pdarray, values).size != self.size:
            raise ValueError(
                "Attempt to group array using key array of different length"
            )

        if self.assume_sorted:
            permuted_values = cast(pdarray, values)
        else:
            permuted_values = cast(pdarray, values)[cast(pdarray, self.permutation)]

        cmd = "segmentedReduction"
        args = "{} {} {} {}".format(
            permuted_values.name, self.segments.name, operator, skipna
        )
        repMsg = generic_msg(cmd=cmd, args=args)
        self.logger.debug(repMsg)
        if operator.startswith("arg"):
            return (
                self.unique_keys,
                cast(pdarray, self.permutation[create_pdarray(repMsg)]),
            )
        else:
            return self.unique_keys, create_pdarray(repMsg)

    def sum(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
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

        Examples
        --------
        >>> a = ak.randint(1,5,10)
        >>> a
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> b = ak.randint(1,5,10)
        >>> b
        array([3, 3, 3, 4, 1, 1, 3, 3, 3, 4])
        >>> g.sum(b)
        (array([2, 3, 4]), array([8, 14, 6]))
        """
        return self.aggregate(values, "sum", skipna)

    def prod(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
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
        RuntimeError
            Raised if prod is not supported for the values dtype

        Notes
        -----
        The return dtype is always float64.

        Examples
        --------
        >>> a = ak.randint(1,5,10)
        >>> a
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> b = ak.randint(1,5,10)
        >>> b
        array([3, 3, 3, 4, 1, 1, 3, 3, 3, 4])
        >>> g.prod(b)
        (array([2, 3, 4]), array([12, 108.00000000000003, 8.9999999999999982]))
        """
        return self.aggregate(values, "prod", skipna)

    def mean(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
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

        Examples
        --------
        >>> a = ak.randint(1,5,10)
        >>> a
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> b = ak.randint(1,5,10)
        >>> b
        array([3, 3, 3, 4, 1, 1, 3, 3, 3, 4])
        >>> g.mean(b)
        (array([2, 3, 4]), array([2.6666666666666665, 2.7999999999999998, 3]))
        """
        return self.aggregate(values, "mean", skipna)

    def min(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
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
            Raised if the values array is not a pdarray object or if min is
            not supported for the values dtype
        ValueError
            Raised if the key array size does not match the values size
            or if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if min is not supported for the values dtype

        Examples
        --------
        >>> a = ak.randint(1,5,10)
        >>> a
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> b = ak.randint(1,5,10)
        >>> b
        array([3, 3, 3, 4, 1, 1, 3, 3, 3, 4])
        >>> g.min(b)
        (array([2, 3, 4]), array([1, 1, 3]))
        """
        if values.dtype == bool:
            raise TypeError(
                "min is only supported for pdarrays of dtype float64, uint64, and int64"
            )
        return self.aggregate(values, "min", skipna)

    def max(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
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
            Raised if the values array is not a pdarray object or if max is
            not supported for the values dtype
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if max is not supported for the values dtype

        Examples
        --------
        >>> a = ak.randint(1,5,10)
        >>> a
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> b = ak.randint(1,5,10)
        >>> b
        array([3, 3, 3, 4, 1, 1, 3, 3, 3, 4])
        >>> g.max(b)
        (array([2, 3, 4]), array([4, 4, 3]))
        """
        if values.dtype == bool:
            raise TypeError(
                "max is only supported for pdarrays of dtype float64, uint64, and int64"
            )
        return self.aggregate(values, "max", skipna)

    def argmin(self, values: pdarray) -> Tuple[groupable, pdarray]:
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
        >>> a = ak.randint(1,5,10)
        >>> a
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> b = ak.randint(1,5,10)
        >>> b
        array([3, 3, 3, 4, 1, 1, 3, 3, 3, 4])
        >>> g.argmin(b)
        (array([2, 3, 4]), array([5, 4, 2]))
        """

        return self.aggregate(values, "argmin")

    def argmax(self, values: pdarray) -> Tuple[groupable, pdarray]:
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
        >>> a = ak.randint(1,5,10)
        >>> a
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> g = ak.GroupBy(a)
        >>> g.keys
        array([3, 3, 4, 3, 3, 2, 3, 2, 4, 2])
        >>> b = ak.randint(1,5,10)
        >>> b
        array([3, 3, 3, 4, 1, 1, 3, 3, 3, 4])
        >>> g.argmax(b)
        (array([2, 3, 4]), array([9, 3, 2]))
        """

        return self.aggregate(values, "argmax")

    def nunique(self, values: groupable) -> Tuple[groupable, pdarray]:
        """
        Using the permutation stored in the GroupBy instance, group another
        array of values and return the number of unique values in each group.

        Parameters
        ----------
        values : pdarray, int64
            The values to group and find unique values

        Returns
        -------
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
        >>> data = ak.array([3, 4, 3, 1, 1, 4, 3, 4, 1, 4])
        >>> data
        array([3, 4, 3, 1, 1, 4, 3, 4, 1, 4])
        >>> labels = ak.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        >>> labels
        ak.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        >>> g = ak.GroupBy(labels)
        >>> g.keys
        ak.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        >>> g.nunique(data)
        array([1,2,3,4]), array([2, 2, 3, 1])
        #    Group (1,1,1) has values [3,4,3] -> there are 2 unique values 3&4
        #    Group (2,2,2) has values [1,1,4] -> 2 unique values 1&4
        #    Group (3,3,3) has values [3,4,1] -> 3 unique values
        #    Group (4) has values [4] -> 1 unique value
        """
        # TO DO: defer to self.aggregate once logic is ported over to Chapel
        # return self.aggregate(values, "nunique")

        ukidx = self.broadcast(arange(self.ngroups), permute=True)
        # Test if values is single array, i.e. either pdarray, Strings,
        # or Categorical (the last two have a .group() method).
        # Can't directly test Categorical due to circular import.
        if isinstance(values, pdarray):
            if (
                cast(pdarray, values).dtype != akint64
                and cast(pdarray, values).dtype != akuint64
            ):
                raise TypeError("nunique unsupported for this dtype")
            togroup = [ukidx, values]
        elif hasattr(values, "group"):
            togroup = [ukidx, values]
        else:
            for v in values:
                if (
                    isinstance(values, pdarray)
                    and cast(pdarray, values).dtype != akint64
                    and cast(pdarray, values).dtype != akuint64
                ):
                    raise TypeError("nunique unsupported for this dtype")
            togroup = [ukidx] + list(values)
        # Find unique pairs of (key, val)
        g = GroupBy(togroup)
        # Group unique pairs again by original key
        g2 = GroupBy(g.unique_keys[0], assume_sorted=True)
        # Count number of unique values per key
        _, nuniq = g2.count()
        # Re-join unique counts with original keys (sorting guarantees same order)
        return self.unique_keys, nuniq

    def any(
        self, values: pdarray
    ) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
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
            Raised if the values array is not a pdarray or if the pdarray
            dtype is not bool
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        """
        if values.dtype != bool:
            raise TypeError("any is only supported for pdarrays of dtype bool")
        return self.aggregate(values, "any")  # type: ignore

    def all(
        self, values: pdarray
    ) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
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
            Raised if the values array is not a pdarray or if the pdarray
            dtype is not bool
        ValueError
            Raised if the key array size does not match the values size or
            if the operator is not in the GroupBy.Reductions array
        RuntimeError
            Raised if all is not supported for the values dtype
        """
        if values.dtype != bool:
            raise TypeError("all is only supported for pdarrays of dtype bool")

        return self.aggregate(values, "all")  # type: ignore

    def OR(
        self, values: pdarray
    ) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
        """
        Bitwise OR of values in each segment.

        Using the permutation stored in the GroupBy instance, group
        another array of values and perform a bitwise OR reduction on
        each group.

        Parameters
        ----------
        values : pdarray, int64
            The values to group and reduce with OR

        Returns
        -------
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
        """
        if values.dtype != akint64 and values.dtype != akuint64:
            raise TypeError(
                "OR is only supported for pdarrays of dtype int64 or uint64"
            )

        return self.aggregate(values, "or")  # type: ignore

    def AND(
        self, values: pdarray
    ) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
        """
        Bitwise AND of values in each segment.

        Using the permutation stored in the GroupBy instance, group
        another array of values and perform a bitwise AND reduction on
        each group.

        Parameters
        ----------
        values : pdarray, int64
            The values to group and reduce with AND

        Returns
        -------
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
        """
        if values.dtype != akint64 and values.dtype != akuint64:
            raise TypeError(
                "AND is only supported for pdarrays of dtype int64 or uint64"
            )

        return self.aggregate(values, "and")  # type: ignore

    def XOR(
        self, values: pdarray
    ) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
        """
        Bitwise XOR of values in each segment.

        Using the permutation stored in the GroupBy instance, group
        another array of values and perform a bitwise XOR reduction on
        each group.

        Parameters
        ----------
        values : pdarray, int64
            The values to group and reduce with XOR

        Returns
        -------
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
        """
        if values.dtype != akint64 and values.dtype != akuint64:
            raise TypeError(
                "XOR is only supported for pdarrays of dtype int64 or uint64"
            )

        return self.aggregate(values, "xor")  # type: ignore

    @typechecked
    def broadcast(self, values: pdarray, permute: bool = True) -> pdarray:
        """
        Fill each group's segment with a constant value.

        Parameters
        ----------
        values : pdarray
            The values to put in each group's segment
        permute : bool
            If True (default), permute broadcast values back to the ordering
            of the original array on which GroupBy was called. If False, the
            broadcast values are grouped by value.

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

        Examples
        --------
        >>> a = ak.array([0, 1, 0, 1, 0])
        >>> values = ak.array([3, 5])
        >>> g = ak.GroupBy(a)
        # By default, result is in original order
        >>> g.broadcast(values)
        array([3, 5, 3, 5, 3])

        # With permute=False, result is in grouped order
        >>> g.broadcast(values, permute=False)
        array([3, 3, 3, 5, 5]

        >>> a = ak.randint(1,5,10)
        >>> a
        array([3, 1, 4, 4, 4, 1, 3, 3, 2, 2])
        >>> g = ak.GroupBy(a)
        >>> keys,counts = g.count()
        >>> g.broadcast(counts > 2)
        array([True False True True True False True True False False])
        >>> g.broadcast(counts == 3)
        array([True False True True True False True True False False])
        >>> g.broadcast(counts < 4)
        array([True True True True True True True True True True])
        """
        if values.size != self.segments.size:
            raise ValueError("Must have one value per segment")
        cmd = "broadcast"
        args = "{} {} {} {} {}".format(
            self.permutation.name, self.segments.name, values.name, permute, self.size
        )
        repMsg = generic_msg(cmd=cmd, args=args)
        return create_pdarray(repMsg)

    @staticmethod
    def build_from_components(user_defined_name: str = None, **kwargs) -> GroupBy:
        """
        function to build a new GroupBy object from component keys and permutation.

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

        """
        if (
            "orig_keys" in kwargs
            and "permutation" in kwargs
            and "unique_keys" in kwargs
            and "segments" in kwargs
        ):
            g = GroupBy(None, **kwargs)
            g.name = user_defined_name

            return g
        else:
            missingKeys = []
            if "orig_keys" not in kwargs:
                missingKeys.append("orig_keys")
            if "permutation" not in kwargs:
                missingKeys.append("permutation")
            if "unique_keys" not in kwargs:
                missingKeys.append("unique_keys")
            if "segments" not in kwargs:
                missingKeys.append("segments")

            raise ValueError(
                f"Can't build GroupBy. kwargs is missing required keys: {missingKeys}."
            )

    def _get_groupby_required_pieces(self) -> Dict:
        """
        Internal function that returns a dictionary with all required components of self

        Returns
        -------
        Dict
            Dictionary of all required components of self
                Components (keys, permutation)
        """
        requiredPieces = frozenset(["keys", "permutation", "unique_keys", "segments"])

        return {piece_name: getattr(self, piece_name) for piece_name in requiredPieces}

    @typechecked
    def register(self, user_defined_name: str) -> GroupBy:
        """
        Register this GroupBy object and underlying components with the Arkouda server

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

        See also
        --------
        unregister, attach, unregister_groupby_by_name, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        from arkouda import Categorical

        # By registering unique properties first, we can ensure no overlap in naming between
        #   two registered GroupBy's since this will throw a RegistrationError before any of
        #   the dynamically created names are registered
        self.permutation.register(f"{user_defined_name}.permutation")
        self.segments.register(f"{user_defined_name}.segments")

        if isinstance(self.keys, (Strings, pdarray, Categorical)):
            self.keys.register(f"{user_defined_name}_{self.keys.objtype}.keys")
            self.unique_keys.register(
                f"{user_defined_name}_{self.keys.objtype}.unique_keys"
            )
        elif isinstance(self.keys, Sequence):
            for x in range(len(self.keys)):
                # Possible for multiple types in a sequence, so we have to check each key's
                # type individually
                if isinstance(self.keys[x], (Strings, pdarray, Categorical)):
                    self.keys[x].register(
                        f"{x}_{user_defined_name}_{self.keys[x].objtype}.keys"
                    )
                    self.unique_keys[x].register(
                        f"{x}_{user_defined_name}_{self.keys[x].objtype}.unique_keys"
                    )

        else:
            raise RegistrationError(f"Unsupported key type found: {type(self.keys)}")

        self.name = user_defined_name
        return self

    def unregister(self):
        """
        Unregister this GroupBy object in the arkouda server which was previously
        registered using register() and/or attached to using attach()

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See also
        --------
        register, attach, unregister_groupby_by_name, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """

        if not self.name:
            raise RegistrationError(
                "This item does not have a name and does not appear to be registered."
            )

        # Unregister all components in keys in the case of a Sequence
        if isinstance(self.keys, Sequence):
            for x in range(len(self.keys)):
                self.keys[x].unregister()
                self.unique_keys[x].unregister()
        else:
            self.keys.unregister()
            self.unique_keys.unregister()

        self.permutation.unregister()
        self.segments.unregister()

        self.name = None  # Clear our internal GroupBy object name

    def is_registered(self) -> bool:
        """
         Return True if the object is contained in the registry

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
        """
        import warnings

        from arkouda import Categorical

        if self.name is None:
            return False  # unnamed GroupBy cannot be registered

        if isinstance(self.keys, Sequence):  # Sequence - Check for all components
            from re import compile

            registry = list_registry()

            # Only check for a single component of Categorical to ensure correct count.
            regEx = compile(
                f"^\\d+_{self.name}_.+\\.keys$|^\\d+_{self.name}_.+\\.unique_keys$|"
                f"^\\d+_{self.name}_.+\\.unique_keys(?=\\.categories$)"
            )
            cat_regEx = compile(
                f"^\\d+_{self.name}_{Categorical.objtype}\\.keys(?=\\.codes$)"
            )

            simple_registered = list(filter(regEx.match, registry))
            cat_registered = list(filter(cat_regEx.match, registry))
            if f"{self.name}.permutation" in registry:
                simple_registered.append(f"{self.name}.permutation")
            if f"{self.name}.segments" in registry:
                simple_registered.append(f"{self.name}.segments")

            # In the case of Categorical, unique keys is registered with only categories and codes and
            # overwrites keys.categories
            total = (len(self.keys) * 2) + 2

            registered = len(simple_registered) + len(cat_registered)
            if 0 < registered < total:
                warnings.warn(
                    f"WARNING: GroupBy {self.name} expected {total} components to be registered,"
                    f" but only located {registered}."
                )
                return False
            else:
                return registered == total
        else:
            parts_registered: List[bool] = []
            for k, v in GroupBy._get_groupby_required_pieces(self).items():
                if k != "unique_keys" or not isinstance(self.unique_keys, Categorical):
                    reg = v.is_registered()
                    parts_registered.append(reg)

            if any(parts_registered) and not all(parts_registered):  # test for error
                warnings.warn(
                    f"WARNING: GroupBy {self.name} expected {len(parts_registered)} "
                    f"components to be registered, but only located {sum(parts_registered)}."
                )
                return False
            else:
                return any(parts_registered)

    @staticmethod
    def attach(user_defined_name: str) -> GroupBy:
        """
        Function to return a GroupBy object attached to the registered name in the
        arkouda server which was registered using register()

        Parameters
        ----------
        user_defined_name : str
            user defined name which GroupBy object was registered under

        Returns
        -------
        GroupBy
               The GroupBy object created by re-attaching to the corresponding server components

        Raises
        ------
        RegistrationError
            if user_defined_name is not registered

        See Also
        --------
        register, is_registered, unregister, unregister_groupby_by_name
        """

        from re import compile, match

        from arkouda.categorical import Categorical

        keys: List[groupable] = []
        unique_keys = []
        matches = []
        regEx = compile(
            f"^{user_defined_name}_.+\\.keys$|^\\d+_{user_defined_name}_.+\\.keys$|"
            f"^{user_defined_name}_.+\\.unique_keys$|^\\d+_{user_defined_name}_.+\\.unique_keys$|"
            f"^(?:\\d+_)?{user_defined_name}_{Categorical.objtype}\\.unique_keys(?=\\.categories$)"
        )
        # Using the regex, cycle through the registered items and find all the pieces of
        # the GroupBy's keys
        for name in list_registry():
            x = match(regEx, name)
            if x is not None:
                matches.append(x.group())
        matches.sort()

        if len(matches) == 0:
            raise RegistrationError(
                f"No registered elements with name '{user_defined_name}'"
            )

        for name in matches:
            # Parse the name for the dtype and use the proper create method to create the element
            if f"_{Strings.objtype}." in name or f"_{pdarray.objtype}." in name:
                keys_resp = cast(str, generic_msg(cmd="attach", args=name))
                dtype = keys_resp.split()[2]
                if ".unique_keys" in name:
                    if dtype == Strings.objtype:
                        unique_keys.append(Strings.from_return_msg(keys_resp))
                    else:  # pdarray
                        unique_keys.append(create_pdarray(keys_resp))
                else:
                    if dtype == Strings.objtype:
                        keys.append(Strings.from_return_msg(keys_resp))
                    else:  # pdarray
                        keys.append(create_pdarray(keys_resp))

            elif f"_{Categorical.objtype}.unique_keys" in name:
                # Due to unique_keys overwriting keys.categories, we have to use unique_keys.categories
                # to create the keys Categorical
                unique_key = Categorical.attach(name)
                key_name = name.replace(".unique_keys", ".keys")

                catParts = {
                    "categories": unique_key.categories,
                    "codes": pdarray.attach(f"{key_name}.codes"),
                    "_akNAcode": pdarray.attach(f"{key_name}._akNAcode"),
                }

                # Grab optional components if they exist
                if f"{user_defined_name}.permutation" in matches:
                    catParts["permutation"] = pdarray.attach(f"{key_name}.permutation")
                if f"{user_defined_name}.segments" in matches:
                    catParts["segments"] = pdarray.attach(f"{key_name}.segments")

                unique_keys.append(unique_key)
                keys.append(Categorical(None, **catParts))

            else:
                raise RegistrationError(
                    f"Unknown type associated with registered item: {user_defined_name}."
                    f" Supported types are: {groupable}"
                )

        if len(keys) == 0:
            raise RegistrationError(
                f"Unable to attach to '{user_defined_name}' or '{user_defined_name}'"
                f" is not registered"
            )

        perm_resp = generic_msg(cmd="attach", args=f"{user_defined_name}.permutation")
        segments_resp = generic_msg(cmd="attach", args=f"{user_defined_name}.segments")

        parts = {
            "orig_keys": keys if len(keys) > 1 else keys[0],
            "unique_keys": unique_keys if len(unique_keys) > 1 else unique_keys[0],
            "permutation": create_pdarray(perm_resp),
            "segments": create_pdarray(segments_resp),
        }

        g = GroupBy.build_from_components(
            user_defined_name, **parts
        )  # Call build_from_components method
        return g

    @staticmethod
    @typechecked
    def unregister_groupby_by_name(user_defined_name: str) -> None:
        """
        Function to unregister GroupBy object by name which was registered
        with the arkouda server via register()

        Parameters
        ----------
        user_defined_name : str
            Name under which the GroupBy object was registered

        Raises
        -------
        TypeError
            if user_defined_name is not a string
        RegistrationError
            if there is an issue attempting to unregister any underlying components

        See Also
        --------
        register, unregister, attach, is_registered
        """
        # We have 2 components, unregister both of them
        from re import compile, match

        from arkouda.categorical import Categorical

        registry = list_registry()

        # keys, unique_keys, or categorical components
        regEx = compile(
            f"^{user_defined_name}_.+\\.keys$|^\\d+_{user_defined_name}_.+\\.keys$|"
            f"^{user_defined_name}_.+\\.unique_keys$|^\\d+_{user_defined_name}_.+\\.unique_keys$|"
            f"^(?:\\d+_)?{user_defined_name}_{Categorical.objtype}\\.unique_keys(?=\\.categories$)|"
            f"^(\\d+_)?{user_defined_name}_{Categorical.objtype}\\.keys\\.(_)?([A-Z,a-z])+$"
        )

        for name in registry:
            # Search through registered items and find matches to the given name
            x = match(regEx, name)
            if x is not None:
                print(x.group())
                # Only categorical requires a separate unregister case
                if f"_{Categorical.objtype}.unique_keys" in x.group():
                    Categorical.unregister_categorical_by_name(x.group())
                else:
                    unregister_pdarray_by_name(x.group())

        if f"{user_defined_name}.permutation" in registry:
            unregister_pdarray_by_name(f"{user_defined_name}.permutation")

        if f"{user_defined_name}.segments" in registry:
            unregister_pdarray_by_name(f"{user_defined_name}.segments")

    def most_common(self, values):
        """
        Find the most common value for each segment of a GroupBy object. This method only supports
        array-like GroupBy key types.

        Parameters
        ----------
        values : array-like
            Values in which to find most common based on the GroupBy's segment indexes.
            values.size must equal GroupBy.keys[0].size

        Returns
        -------
        most_common_values : array-like
            The most common value for each segment of the GroupBy
        """
        # Give each key an integer index
        keyidx = self.broadcast(arange(self.unique_keys[0].size), permute=True)
        # Annex values and group by (key, val)
        bykeyval = GroupBy([keyidx, values])
        # Count number of records for each (key, val)
        (ki, uval), count = bykeyval.count()
        # Group out value
        bykey = GroupBy(ki, assume_sorted=True)
        # Find the index of the most frequent value for each key
        _, topidx = bykey.argmax(count)
        # Gather the most frequent values
        return uval[topidx]


def broadcast(
    segments: pdarray,
    values: pdarray,
    size: Union[int, np.int64, np.uint64] = -1,
    permutation: Union[pdarray, None] = None,
):
    """
    Broadcast a dense column vector to the rows of a sparse matrix or grouped array.

    Parameters
    ----------
    segments : pdarray, int64
        Offsets of the start of each row in the sparse matrix or grouped array.
        Must be sorted in ascending order.
    values : pdarray
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
    pdarray
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
    # Define a sparse matrix with 3 rows and 7 nonzeros
    >>> row_starts = ak.array([0, 2, 5])
    >>> nnz = 7
    # Broadcast the row number to each nonzero element
    >>> row_number = ak.arange(3)
    >>> ak.broadcast(row_starts, row_number, nnz)
    array([0 0 1 1 1 2 2])

    # If the original nonzeros were in reverse order...
    >>> permutation = ak.arange(6, -1, -1)
    >>> ak.broadcast(row_starts, row_number, permutation=permutation)
    array([2 2 1 1 1 0 0])
    """
    if segments.size != values.size:
        raise ValueError("segments and values arrays must be same size")
    if segments.size == 0:
        raise ValueError("cannot broadcast empty array")
    if permutation is None:
        if size == -1:
            raise ValueError("must either supply permutation or size")
        pname = "none"
        permute = False
    else:
        pname = permutation.name
        permute = True
        size = cast(Union[int, np.int64, np.uint64], permutation.size)
    if size < 1:
        raise ValueError("result size must be greater than zero")
    cmd = "broadcast"
    args = "{} {} {} {} {}".format(pname, segments.name, values.name, permute, size)
    repMsg = generic_msg(cmd=cmd, args=args)
    return create_pdarray(repMsg)
