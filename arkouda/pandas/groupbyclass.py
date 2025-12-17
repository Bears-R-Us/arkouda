"""
GroupBy functionality for Arkouda.

This module implements group-by operations for Arkouda arrays, enabling efficient
grouping, aggregation, and broadcasting of data using distributed `pdarray` structures.

It provides the `GroupBy` class and supporting utilities for working with groupable
data, following a pandas-like interface while scaling to large datasets via Arkouda's
parallel backend.

Components
----------
GroupBy : class
    Main class for performing group-by operations such as sum, mean, min, max, etc.

unique : function
    Returns the unique values in an array and the permutation that groups them.

broadcast : function
    Broadcasts a per-group aggregate back to the original shape of the input array.

GROUPBY_REDUCTION_TYPES : set
    Set of supported reduction types for `GroupBy.aggregate` (e.g., "sum", "count", "mean").

groupable : type alias
    A type alias representing valid input(s) to the `GroupBy` constructor.
    Acceptable types include:
        - a single `pdarray`, `Strings`, or `Categorical` object
        - a sequence (e.g., list or tuple) of any combination of those types

Examples
--------
>>> import arkouda as ak
>>> from arkouda.groupbyclass import GroupBy

>>> keys = ak.array([0, 1, 0, 1, 0])
>>> values = ak.array([10, 20, 30, 40, 50])
>>> gb = GroupBy(keys)
>>> gb.aggregate(values, 'sum')
(array([0 1]), array([90 60]))

"""

from __future__ import annotations

import enum
import json

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    no_type_check,
)

import numpy as np

from typeguard import typechecked

from arkouda.logger import ArkoudaLogger, getArkoudaLogger
from arkouda.numpy.dtypes import _val_isinstance_of_union, bigint, float_scalars, int_scalars
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import float64 as akfloat64
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import uint64 as akuint64
from arkouda.numpy.pdarrayclass import RegistrationError, create_pdarray, is_sorted, pdarray
from arkouda.numpy.random import default_rng
from arkouda.numpy.sorting import argsort, sort


if TYPE_CHECKING:
    from arkouda.numpy.strings import Strings
    from arkouda.pandas.categorical import Categorical
else:
    Categorical = TypeVar("Categorical")
    Strings = TypeVar("Strings")

__all__ = ["unique", "GroupBy", "broadcast", "GROUPBY_REDUCTION_TYPES", "groupable"]

groupable_element_type = Union[pdarray, Strings, "Categorical"]
groupable = Union[groupable_element_type, Sequence[groupable_element_type]]
# Note: we won't be typechecking GroupBy until we can figure out a way to handle
# the circular import with Categorical


def _get_grouping_keys(pda: groupable):
    nkeys = 1
    if hasattr(pda, "_get_grouping_keys"):
        # Single groupable array
        grouping_keys = cast(list, cast(groupable_element_type, pda)._get_grouping_keys())
    else:
        # Sequence of groupable arrays
        nkeys = len(pda)
        grouping_keys = []
        for k in pda:
            if k.size != pda[0].size:
                raise ValueError("Key arrays must all be same size")
            if not hasattr(k, "_get_grouping_keys"):
                raise TypeError(f"{type(k)} does not support grouping")
            grouping_keys.extend(cast(list, k._get_grouping_keys()))

    return grouping_keys, nkeys


def unique(
    pda: groupable,
    return_groups: bool = False,
    assume_sorted: bool = False,
    return_indices: bool = False,
) -> Union[groupable, Tuple[groupable, pdarray, pdarray, int]]:
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

    """
    from arkouda.client import generic_msg
    from arkouda.pandas.categorical import Categorical as Categorical_

    if not return_groups and hasattr(pda, "unique"):
        return cast(Categorical_, pda).unique()

    # Get all grouping keys
    grouping_keys, nkeys = _get_grouping_keys(pda)
    keynames = [k.name for k in grouping_keys]
    keytypes = [k.objType for k in grouping_keys]
    effectiveKeys = len(grouping_keys)
    repMsg = generic_msg(
        cmd="unique",
        args={
            "returnGroupStr": return_groups,
            "assumeSortedStr": assume_sorted,
            "nstr": effectiveKeys,
            "keynames": keynames,
            "keytypes": keytypes,
        },
    )
    if return_groups:
        parts = cast(str, repMsg).split("+")
        permutation = create_pdarray(cast(str, parts[0]))
        segments = create_pdarray(cast(str, parts[1]))
        unique_key_indices = create_pdarray(cast(str, parts[2]))
    else:
        unique_key_indices = create_pdarray(cast(str, repMsg))

    if nkeys == 1 and not isinstance(pda, Sequence):
        unique_keys = pda[unique_key_indices]
    else:
        unique_keys = tuple(a[unique_key_indices] for a in pda)
    if return_groups:
        if return_indices:
            return unique_keys, permutation, segments, nkeys, unique_key_indices
        else:
            return unique_keys, permutation, segments, nkeys
    else:
        return unique_keys


class GroupByReductionType(enum.Enum):
    SUM = "sum"
    COUNT = "count"
    PROD = "prod"
    VAR = "var"
    STD = "std"
    MEAN = "mean"
    MEDIAN = "median"
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
    FIRST = "first"
    MODE = "mode"
    UNIQUE = "unique"

    def __str__(self) -> str:
        """
        Return the string representation of this GroupByReductionType.

        This overrides the default `str()` to return the internal `value` attribute,
        which is used when specifying reduction types in Arkouda server requests.

        Returns
        -------
        str
            The underlying string value of the reduction type.

        """
        return self.value

    def __repr__(self) -> str:
        """
        Return the canonical string representation of this GroupByReductionType.

        This overrides the default `repr()` to return the internal `value` attribute,
        which is used when encoding reduction types in Arkouda server requests.

        Returns
        -------
        str
            The underlying string value of the reduction type.

        """
        return self.value


GROUPBY_REDUCTION_TYPES = frozenset(
    [member.value for _, member in GroupByReductionType.__members__.items()]
)


class GroupBy:
    """
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

    """

    nkeys: int
    permutation: pdarray
    unique_keys: Union[pdarray, Strings, Categorical, Tuple[Union[pdarray, Strings, Categorical], ...]]
    ngroups: int_scalars
    segments: pdarray
    logger: ArkoudaLogger
    dropna: bool

    Reductions = GROUPBY_REDUCTION_TYPES

    objType = "GroupBy"

    def __init__(
        self,
        keys: Optional[groupable] = None,
        assume_sorted: bool = False,
        dropna: bool = True,
        **kwargs,
    ):
        from arkouda.numpy import isnan

        def drop_na_keys():
            if self.dropna is True:
                if isinstance(self.keys, pdarray) and self.keys.dtype == akfloat64:
                    self.keys = self.keys[~isnan(self.keys)]
                elif isinstance(self.keys, list):
                    is_not_nan = [
                        ~isnan(key)
                        for key in self.keys
                        if isinstance(key, pdarray) and key.dtype == akfloat64
                    ]

                    if len(is_not_nan) > 0:
                        use_value = is_not_nan[0]
                        for bool_arry in is_not_nan:
                            use_value = use_value & bool_arry
                        self.keys = [key[use_value] for key in keys]

        # Type Checks required because @typechecked was removed for causing other issues
        # This prevents non-bool values that can be evaluated to true (ie non-empty arrays)
        # from causing unexpected results. Experienced when forgetting to wrap multiple key arrays in [].
        # See Issue #1267
        self.registered_name: Optional[str] = None
        if not isinstance(assume_sorted, bool):
            raise TypeError("assume_sorted must be of type bool.")

        self.logger = getArkoudaLogger(name=self.__class__.__name__)
        self.assume_sorted = assume_sorted
        self.dropna = dropna
        if (
            "orig_keys" in kwargs
            and "permutation" in kwargs
            and "unique_keys" in kwargs
            and "segments" in kwargs
        ):
            self.keys = cast(groupable, kwargs.get("orig_keys", None))
            drop_na_keys()
            self.unique_keys = cast(
                Union[pdarray, Strings, Categorical, Tuple[Union[pdarray, Strings, Categorical], ...]],
                kwargs["unique_keys"],
            )
            self.permutation = cast(pdarray, kwargs["permutation"])
            self.segments = cast(pdarray, kwargs["segments"])
            self.nkeys = len(self.keys)
            self._uki = self.permutation[self.segments]
        elif (
            "orig_keys" in kwargs
            and "permutation" in kwargs
            and "uki" in kwargs
            and "segments" in kwargs
        ):
            self.keys = cast(groupable, kwargs.get("orig_keys", None))
            drop_na_keys()
            self._uki = kwargs.get("uki", None)
            self.permutation = cast(pdarray, kwargs["permutation"])
            self.segments = cast(pdarray, kwargs["segments"])
            self.nkeys = len(self.keys) if isinstance(self.keys, Sequence) else 1
            if not isinstance(self.keys, Sequence):
                self.unique_keys = self.keys[self._uki]
            else:
                self.unique_keys = tuple(a[self._uki] for a in self.keys)
        elif keys is None:
            raise ValueError("No keys passed to GroupBy.")
        else:
            self.keys = cast(groupable, keys)
            drop_na_keys()
            (
                self.unique_keys,
                self.permutation,
                self.segments,
                self.nkeys,
                self._uki,
            ) = unique(  # type: ignore
                self.keys,
                return_groups=True,
                return_indices=True,
                assume_sorted=self.assume_sorted,
            )
        self.length = self.permutation.size
        self.ngroups = self.segments.size

    @staticmethod
    def from_return_msg(rep_msg):
        """
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

        """
        from arkouda.numpy.strings import Strings
        from arkouda.pandas.categorical import Categorical as Categorical_

        data = json.loads(rep_msg)
        perm = create_pdarray(data["permutation"])
        segs = create_pdarray(data["segments"])
        uki = create_pdarray(data["uki"])
        keys = []
        for k in sorted(data.keys()):  # sort keys to ensure order
            create_data = data[k]
            if k == "permutation" or k == "segments" or k == "uki":
                continue
            comps = create_data.split("+|+")
            if comps[0] == pdarray.objType.upper():
                keys.append(create_pdarray(comps[1]))
            elif comps[0] == Strings.objType.upper():
                keys.append(Strings.from_return_msg(comps[1]))
            elif comps[0] == Categorical_.objType.upper():
                keys.append(Categorical_.from_return_msg(comps[1]))
        if len(keys) == 1:
            keys = keys[0]
        return GroupBy(orig_keys=keys, permutation=perm, segments=segs, uki=uki)

    def to_hdf(
        self,
        prefix_path,
        dataset="groupby",
        mode: Literal["truncate", "append"] = "truncate",
        file_type: Literal["single", "distribute"] = "distribute",
    ):
        """
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

        """
        from arkouda.client import generic_msg
        from arkouda.pandas.categorical import Categorical as Categorical_
        from arkouda.pandas.io import _file_type_to_int, _mode_str_to_int

        keys = self.keys if isinstance(self.keys, Sequence) else [self.keys]

        objTypes = [k.objType for k in keys]  # pdarray, Strings, and Categorical all have objType prop
        dtypes = [k.categories.dtype if isinstance(k, Categorical_) else k.dtype for k in keys]

        # access the names of the key or names of properties for categorical
        gb_keys = [
            (
                k.name
                if not isinstance(k, Categorical_)
                else json.dumps(
                    {
                        "codes": k.codes.name,
                        "categories": k.categories.name,
                        "NA_codes": k._akNAcode.name,
                        **({"permutation": k.permutation.name} if k.permutation is not None else {}),
                        **({"segments": k.segments.name} if k.segments is not None else {}),
                    }
                )
            )
            for k in keys
        ]

        generic_msg(
            cmd="tohdf",
            args={
                "num_keys": len(gb_keys),
                "key_names": gb_keys,
                "key_dtypes": dtypes,
                "key_objTypes": objTypes,
                "unique_key_idx": self._uki,
                "permutation": self.permutation,
                "segments": self.segments,
                "dset": dataset,
                "write_mode": _mode_str_to_int(mode),
                "filename": prefix_path,
                "objType": self.objType,
                "file_format": _file_type_to_int(file_type),
            },
        )

    def update_hdf(
        self,
        prefix_path: str,
        dataset: str = "groupby",
        repack: bool = True,
    ):
        """
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

        """
        from arkouda.client import generic_msg
        from arkouda.pandas.io import (
            _file_type_to_int,
            _get_hdf_filetype,
            _mode_str_to_int,
            _repack_hdf,
        )

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        from arkouda.pandas.categorical import Categorical as Categorical_

        keys = self.keys
        if not isinstance(self.keys, Sequence):
            keys = [self.keys]

        objTypes = [k.objType for k in keys]  # pdarray, Strings, and Categorical all have objType prop
        dtypes = [k.categories.dtype if isinstance(k, Categorical_) else k.dtype for k in keys]

        # access the names of the key or names of properties for categorical
        gb_keys = [
            (
                k.name
                if not isinstance(k, Categorical_)
                else json.dumps(
                    {
                        "codes": k.codes.name,
                        "categories": k.categories.name,
                        "NA_codes": k._akNAcode.name,
                        **({"permutation": k.permutation.name} if k.permutation is not None else {}),
                        **({"segments": k.segments.name} if k.segments is not None else {}),
                    }
                )
            )
            for k in keys
        ]

        generic_msg(
            cmd="tohdf",
            args={
                "num_keys": len(gb_keys),
                "key_names": gb_keys,
                "key_dtypes": dtypes,
                "key_objTypes": objTypes,
                "unique_key_idx": self._uki,
                "permutation": self.permutation,
                "segments": self.segments,
                "dset": dataset,
                "write_mode": _mode_str_to_int("truncate"),
                "filename": prefix_path,
                "objType": self.objType,
                "file_format": _file_type_to_int(file_type),
                "overwrite": True,
            },
        )

        if repack:
            _repack_hdf(prefix_path)

    def size(self) -> Tuple[groupable, pdarray]:
        """
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

        """
        from arkouda.client import generic_msg

        repMsg = generic_msg(
            cmd="sizeReduction",
            args={"segments": cast(pdarray, self.segments), "size": self.length},
        )
        self.logger.debug(repMsg)
        return self.unique_keys, create_pdarray(repMsg)

    def count(self, values: pdarray) -> Tuple[groupable, pdarray]:
        """
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

        """
        k, v = self.aggregate(values, "count")
        return k, cast(pdarray, v)

    def aggregate(
        self,
        values: groupable,
        operator: str,
        skipna: bool = True,
        ddof: int_scalars = 1,
    ) -> Tuple[groupable, groupable]:
        """
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

        """
        from arkouda.client import generic_msg

        operator = operator.lower()
        if operator not in self.Reductions:
            raise ValueError(f"Unsupported reduction: {operator}\nMust be one of {self.Reductions}")

        # TO DO: remove once logic is ported over to Chapel
        if operator == "nunique":
            return self.nunique(values)
        if operator == "first":
            return self.first(cast(groupable_element_type, values))
        if operator == "mode":
            return self.mode(values)
        if operator == "unique":
            return self.unique(values)

        # All other aggregations operate on pdarray
        if cast(pdarray, values).size != self.length:
            raise ValueError("Attempt to group array using key array of different length")

        if self.assume_sorted:
            permuted_values = cast(pdarray, values)
        else:
            permuted_values = cast(pdarray, values)[cast(pdarray, self.permutation)]

        repMsg = generic_msg(
            cmd="segmentedReduction",
            args={
                "values": permuted_values,
                "segments": self.segments,
                "op": operator,
                "skip_nan": skipna,
                "ddof": ddof,
            },
        )
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

        """
        k, v = self.aggregate(values, "sum", skipna)
        return k, cast(pdarray, v)

    def prod(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
        """
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

        """
        k, v = self.aggregate(values, "prod", skipna)
        return k, cast(pdarray, v)

    def var(
        self, values: pdarray, skipna: bool = True, ddof: int_scalars = 1
    ) -> Tuple[groupable, pdarray]:
        """
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

        """
        k, v = self.aggregate(values, "var", skipna, ddof)
        return k, cast(pdarray, v)

    def std(
        self, values: pdarray, skipna: bool = True, ddof: int_scalars = 1
    ) -> Tuple[groupable, pdarray]:
        """
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

        """
        k, v = self.aggregate(values, "std", skipna, ddof)
        return k, cast(pdarray, v)

    def mean(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
        """
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

        """
        k, v = self.aggregate(values, "mean", skipna)
        return k, cast(pdarray, v)

    def median(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
        """
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

        """
        k, v = self.aggregate(values, "median", skipna)
        return k, cast(pdarray, v)

    def min(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
        """
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

        """
        if values.dtype == bool:
            raise TypeError("min is only supported for pdarrays of dtype float64, uint64, and int64")
        k, v = self.aggregate(values, "min", skipna)
        return k, cast(pdarray, v)

    def max(self, values: pdarray, skipna: bool = True) -> Tuple[groupable, pdarray]:
        """
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

        """
        if values.dtype == bool:
            raise TypeError("max is only supported for pdarrays of dtype float64, uint64, and int64")
        k, v = self.aggregate(values, "max", skipna)
        return k, cast(pdarray, v)

    def argmin(self, values: pdarray) -> Tuple[groupable, pdarray]:
        """
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

        """
        k, v = self.aggregate(values, "argmin")
        return k, cast(pdarray, v)

    def argmax(self, values: pdarray) -> Tuple[groupable, pdarray]:
        """
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

        """
        k, v = self.aggregate(values, "argmax")
        return k, cast(pdarray, v)

    def _nested_grouping_helper(self, values: groupable) -> groupable:
        from arkouda.numpy.pdarraycreation import arange

        unique_key_idx = self.broadcast(arange(self.ngroups), permute=True)
        if hasattr(values, "_get_grouping_keys"):
            # All single-array groupable types must have a _get_grouping_keys method
            if isinstance(values, pdarray) and values.dtype == akfloat64:
                raise TypeError("grouping/uniquing unsupported for float64 arrays")
            togroup = [unique_key_idx, values]
        else:
            # Treat as a sequence of groupable arrays
            for v in values:
                if isinstance(v, pdarray) and v.dtype not in [
                    akint64,
                    akuint64,
                    bigint,
                ]:
                    raise TypeError("grouping/uniquing unsupported for this dtype")
            togroup = [unique_key_idx] + list(values)
        return togroup

    def nunique(self, values: groupable) -> Tuple[groupable, pdarray]:
        """
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

        """
        # TO DO: defer to self.aggregate once logic is ported over to Chapel
        # return self.aggregate(values, "nunique")
        togroup = self._nested_grouping_helper(values)
        # Find unique pairs of (key, val)
        g = GroupBy(togroup)
        # Group unique pairs again by original key
        g2 = GroupBy(g.unique_keys[0], assume_sorted=False)
        # Count number of values per key
        keyorder, nuniq = g2.size()

        from arkouda.categorical import Categorical

        if not isinstance(keyorder, (pdarray, Strings, Categorical)):
            raise TypeError(
                f"groupby failed to produce keyorder of the correct type: "
                f"{type(keyorder)} is not pdarray, Strings, or Categorical."
            )

        # The last GroupBy *should* result in sorted key indices, but in case it
        # doesn't, we need to permute the answer to match the original key order
        if isinstance(keyorder, pdarray) and not is_sorted(keyorder):
            perm = argsort(keyorder)
            nuniq = nuniq[perm]
        # Re-join unique counts with original keys (sorting guarantees same order)
        return self.unique_keys, nuniq

    def any(self, values: pdarray) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
        """
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

        """
        if values.dtype != bool:
            raise TypeError("any is only supported for pdarrays of dtype bool")
        return self.aggregate(values, "any")  # type: ignore

    def all(self, values: pdarray) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
        """
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

        """
        if values.dtype != bool:
            raise TypeError("all is only supported for pdarrays of dtype bool")

        return self.aggregate(values, "all")  # type: ignore

    def OR(self, values: pdarray) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
        """
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

        """
        if values.dtype not in [akint64, akuint64, bigint]:
            raise TypeError("OR is only supported for pdarrays of dtype int64, uint64, or bigint")

        return self.aggregate(values, "or")  # type: ignore

    def AND(self, values: pdarray) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
        """
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

        """
        if values.dtype not in [akint64, akuint64, bigint]:
            raise TypeError("AND is only supported for pdarrays of dtype int64, uint64, or bigint")

        return self.aggregate(values, "and")  # type: ignore

    def XOR(self, values: pdarray) -> Tuple[Union[pdarray, List[Union[pdarray, Strings]]], pdarray]:
        """
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

        """
        if values.dtype not in [akint64, akuint64]:
            raise TypeError("XOR is only supported for pdarrays of dtype int64 or uint64")

        return self.aggregate(values, "xor")  # type: ignore

    def head(
        self,
        values: groupable_element_type,
        n: int = 5,
        return_indices: bool = True,
    ) -> Tuple[groupable, groupable_element_type]:
        """
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

        """
        from arkouda.client import generic_msg

        repMsg = generic_msg(
            cmd="segmentedReduction",
            args={
                "values": self.permutation,
                "segments": self.segments,
                "op": "head",
                "n": n,
                "skip_nan": False,
                "ddof": 1,
            },
        )
        self.logger.debug(repMsg)

        ret_indices = create_pdarray(repMsg)
        if return_indices is True:
            return self.unique_keys, ret_indices
        else:
            return self.unique_keys, values[ret_indices]

    def tail(
        self,
        values: groupable_element_type,
        n: int = 5,
        return_indices: bool = True,
    ) -> Tuple[groupable, groupable_element_type]:
        """
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

        """
        from arkouda.client import generic_msg

        repMsg = generic_msg(
            cmd="segmentedReduction",
            args={
                "values": self.permutation,
                "segments": self.segments,
                "op": "tail",
                "n": n,
                "skip_nan": False,
                "ddof": 1,
            },
        )
        self.logger.debug(repMsg)

        ret_indices = create_pdarray(repMsg)
        if return_indices is True:
            return self.unique_keys, ret_indices
        else:
            return self.unique_keys, values[ret_indices]

    def first(self, values: groupable_element_type) -> Tuple[groupable, groupable_element_type]:
        """
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

        """
        # Index of first value in each segment, in input domain
        first_idx = self.permutation[self.segments]
        return self.unique_keys, values[first_idx]

    def mode(self, values: groupable) -> Tuple[groupable, groupable]:
        """
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

        """
        from arkouda.numpy.pdarraycreation import arange

        togroup = self._nested_grouping_helper(values)
        # Get value counts for each key group
        g = GroupBy(togroup)
        keys_values, value_count = g.size()
        # Descending rank of first instance of each (key, value) pair
        first_rank = g.length - g.permutation[g.segments]
        ki, unique_values = keys_values[0], keys_values[1:]
        # Find the index of the modal value for each unique key
        # If more than one modal value, this will get the first one
        g2 = GroupBy(ki)
        uki, mode_count = g2.max(value_count)
        # First value for which value count is maximized
        _, mode_idx = g2.argmax(first_rank * (value_count == g2.broadcast(mode_count)))
        # GroupBy should be stable with a single key array, but
        # if for some reason these unique keys are not in original
        # order, then permute them accordingly
        if not cast(pdarray, (uki == arange(self.ngroups))).all():
            mode_idx = mode_idx[argsort(cast(pdarray, uki))]
        # Gather values at mode indices
        if len(unique_values) == 1:
            # squeeze singletons
            mode = unique_values[0][mode_idx]
        else:
            mode = [uv[mode_idx] for uv in unique_values]
        return self.unique_keys, mode

    def sample(
        self,
        values: groupable,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        return_indices=False,
        permute_samples=False,
    ):
        """
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

        """
        from arkouda.client import generic_msg
        from arkouda.numpy import cast as akcast
        from arkouda.numpy import round as akround
        from arkouda.numpy.pdarraycreation import full

        if frac is not None and n is not None:
            raise ValueError("Please enter a value for `frac` OR `n`, not both")
        if frac is None and n is None:
            n = 1

        if not isinstance(values, Sequence):
            if values.size != self.length:
                raise ValueError("Attempt to group values using key array of different length")
        else:
            if any(val.size != self.length for val in values):
                raise ValueError("Attempt to group values using key array of different length")

        _, seg_lens = self.size()

        if n is not None:
            if not _val_isinstance_of_union(n, int_scalars):
                raise TypeError("n must be an int scalar.")
            num_samples = full(seg_lens.size, n, akint64)

        if frac is not None:
            if not _val_isinstance_of_union(frac, float_scalars):
                raise TypeError("frac must be a float scalar.")
            num_samples = akcast(akround(frac * seg_lens), dt=akint64)

        if not replace and (num_samples > seg_lens).any():
            raise ValueError("Cannot take a larger sample than population when replace is False")

        if (num_samples <= 0).any():
            raise ValueError("Cannot take a negative number of samples")

        has_weights = weights is not None
        if has_weights:
            if not isinstance(weights, pdarray):
                raise TypeError("weights must be a pdarray")

            if weights.size != self.length:
                raise ValueError("Weights and values to be sampled must be of same length")

            if (weights < 0).any():
                raise ValueError("Weights may not include negative values")

            permuted_weights = weights[self.permutation]

            # the below is equivalent to doing `_, weight_sum = self.sum(weights)`, but
            # by calling segmentedReduction directly, we avoid permuting the weights twice.
            # it's unclear if this is worth the additional code complexity
            repMsg = generic_msg(
                cmd="segmentedReduction",
                args={
                    "values": permuted_weights,
                    "segments": self.segments,
                    "op": "sum",
                    "skip_nan": True,
                    "ddof": 1,
                },
            )
            weight_sum = create_pdarray(repMsg)

            if (weight_sum == 0).any():
                raise ValueError("All segments must have at least one value of non-zero weight")

            if permuted_weights.dtype != akfloat64:
                permuted_weights = akcast(permuted_weights, akfloat64)
        else:
            permuted_weights = ""

        random_state = default_rng(random_state)
        gen_name = random_state._name_dict[akdtype(akfloat64 if has_weights else akint64)]

        has_seed = random_state._seed is not None

        repMsg = generic_msg(
            cmd="segmentedSample",
            args={
                "genName": gen_name,
                "perm": self.permutation,
                "segs": self.segments,
                "segLens": seg_lens,
                "weights": permuted_weights,
                "numSamples": num_samples,
                "replace": replace,
                "hasWeights": has_weights,
                "hasSeed": has_seed,
                "seed": random_state._seed if has_seed else "",
                "state": random_state._state,
            },
        )
        random_state._state += self.length

        self.logger.debug(repMsg)
        # sorting the sample permutation gives the sampled indices
        sampled_idx = create_pdarray(repMsg) if permute_samples else sort(create_pdarray(repMsg))
        if return_indices:
            return sampled_idx
        elif not isinstance(values, Sequence):
            return values[sampled_idx]
        else:
            return [val[sampled_idx] for val in values]

    def unique(self, values: groupable):
        """
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

        """
        from arkouda import Categorical
        from arkouda.numpy.pdarraycreation import arange
        from arkouda.numpy.segarray import SegArray
        from arkouda.numpy.strings import Strings

        if isinstance(values, (Strings, Categorical)) or (
            isinstance(values, Sequence) and any([isinstance(v, (Strings, Categorical)) for v in values])
        ):
            raise TypeError("Groupby.unique not supported on Strings or Categorical")

        togroup = self._nested_grouping_helper(values)
        # Group to unique (key, value) pairs
        g = GroupBy(togroup)
        ki, unique_values = g.unique_keys[0], g.unique_keys[1:]
        # Group pairs by key
        g2 = GroupBy(ki)
        # GroupBy should be stable with a single key array, but
        # if for some reason these unique keys are not in original
        # order, then permute them accordingly
        if not (isinstance(g2.unique_keys, pdarray) and (g2.unique_keys == arange(self.ngroups)).all()):
            perm = argsort(cast(pdarray, g2.unique_keys))
            reorder = True
        else:
            reorder = False
        # Form a SegArray for each value array
        # Segments are from grouping by key indices
        # Values are the unique elements of the values arg
        if len(unique_values) == 1:
            # Squeeze singleton results
            ret = SegArray(g2.segments, unique_values[0])
            if reorder:
                ret = ret[perm]
        else:
            ret = [SegArray(g2.segments, uv) for uv in unique_values]  # type: ignore
            if reorder:
                ret = [r[perm] for r in ret]  # type: ignore
        return self.unique_keys, ret

    @typechecked
    def broadcast(
        self, values: Union[pdarray, Strings], permute: bool = True
    ) -> Union[pdarray, Strings]:
        """
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

        """
        from arkouda.client import generic_msg
        from arkouda.numpy.pdarraycreation import arange
        from arkouda.numpy.strings import Strings

        if not isinstance(values, (pdarray, Strings)):
            raise TypeError("values must be a pdarray or Strings")
        if values.size != self.segments.size:
            raise ValueError("Must have one value per segment")
        is_str = isinstance(values, Strings)
        if is_str:
            str_vals = values
            values = arange(str_vals.size)
        cmd = "gbbroadcast"
        repMsg = cast(
            str,
            generic_msg(
                cmd=cmd,
                args={
                    "permName": self.permutation.name,
                    "segName": self.segments.name,
                    "valName": values.name,
                    "permute": permute,
                    "size": self.length,
                },
            ),
        )
        broadcasted = create_pdarray(repMsg)
        return str_vals[broadcasted] if is_str else broadcasted

    @staticmethod
    def build_from_components(user_defined_name: Optional[str] = None, **kwargs) -> GroupBy:
        """
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

        """
        if (
            "orig_keys" in kwargs
            and "permutation" in kwargs
            and "unique_keys" in kwargs
            and "segments" in kwargs
        ):
            g = GroupBy(None, **kwargs)
            g.registered_name = user_defined_name

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

            raise ValueError(f"Can't build GroupBy. kwargs is missing required keys: {missingKeys}.")

    def _get_groupby_required_pieces(self) -> Dict:
        """
        Return a dictionary with all required components of self.

        Returns
        -------
        Dict
            Dictionary of all required components of self
                Components (keys, permutation)

        """
        requiredPieces = frozenset(["keys", "permutation", "unique_keys", "segments"])

        return {piece_name: getattr(self, piece_name) for piece_name in requiredPieces}

    @no_type_check
    def register(self, user_defined_name: str) -> GroupBy:
        """
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

        """
        from arkouda import Categorical
        from arkouda.client import generic_msg
        from arkouda.numpy.strings import Strings

        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")

        if isinstance(self.keys, (pdarray, Strings, Categorical)):
            key_data = [
                (
                    self.keys
                    if not isinstance(self.keys, Categorical)
                    else json.dumps(
                        {
                            "codes": self.keys.codes.name,
                            "categories": self.keys.categories.name,
                            "NA_codes": self.keys._akNAcode.name,
                            **(
                                {"permutation": self.keys.permutation.name}
                                if self.keys.permutation is not None
                                else {}
                            ),
                            **(
                                {"segments": self.keys.segments.name}
                                if self.keys.segments is not None
                                else {}
                            ),
                        }
                    )
                )
            ]
        else:
            key_data = [
                (
                    k.name
                    if not isinstance(k, Categorical)
                    else json.dumps(
                        {
                            "codes": k.codes.name,
                            "categories": k.categories.name,
                            "NA_codes": k._akNAcode.name,
                            **({"permutation": k.permutation.name} if k.permutation is not None else {}),
                            **({"segments": k.segments.name} if k.segments is not None else {}),
                        }
                    )
                )
                for k in self.keys
            ]

        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "segments": self.segments,
                "permutation": self.permutation,
                "uki": self._uki,
                "num_keys": len(self.keys) if isinstance(self.keys, Sequence) else 1,
                "keys": key_data,
                "key_objTypes": (
                    [key.objType for key in self.keys]
                    if isinstance(self.keys, Sequence)
                    else [self.keys.objType]
                ),
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self):
        """
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

        """
        from arkouda.numpy.util import unregister

        if not self.registered_name:
            raise RegistrationError(
                "This item does not have a name and does not appear to be registered."
            )

        unregister(self.registered_name)

        self.registered_name = None  # Clear our internal GroupBy object name

    def is_registered(self) -> bool:
        """
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

        """
        from arkouda.numpy.util import is_registered

        if self.registered_name is None:
            return False
        return is_registered(self.registered_name)


def broadcast(
    segments: pdarray,
    values: Union[pdarray, Strings],
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

    """
    from arkouda.client import generic_msg
    from arkouda.numpy.pdarraycreation import arange
    from arkouda.numpy.strings import Strings

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

    is_str = isinstance(values, Strings)
    if is_str:
        str_vals = values
        values = arange(str_vals.size)

    cmd = "gbbroadcast"
    repMsg = cast(
        str,
        generic_msg(
            cmd=cmd,
            args={
                "permName": pname,
                "segName": segments.name,
                "valName": values.name,
                "permute": permute,
                "size": size,
            },
        ),
    )
    broadcasted = create_pdarray(repMsg)
    return str_vals[broadcasted] if is_str else broadcasted
