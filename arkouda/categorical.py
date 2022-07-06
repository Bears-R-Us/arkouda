from __future__ import annotations

import itertools
from collections import defaultdict
from typing import (
    DefaultDict,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np  # type: ignore
from typeguard import typechecked
from arkouda.decorators import objtypedec

from arkouda.dtypes import bool as akbool
from arkouda.dtypes import int64 as akint64
from arkouda.dtypes import int_scalars, resolve_scalar_dtype, str_scalars, str_
from arkouda.groupbyclass import GroupBy, unique
from arkouda.infoclass import information, list_registry
from arkouda.logger import getArkoudaLogger
from arkouda.numeric import cast as akcast
from arkouda.numeric import where
from arkouda.pdarrayclass import (
    RegistrationError,
    create_pdarray,
    pdarray,
    unregister_pdarray_by_name,
)
from arkouda.pdarraycreation import arange, array, ones, zeros, zeros_like
from arkouda.pdarraysetops import concatenate, in1d
from arkouda.sorting import argsort
from arkouda.strings import Strings

__all__ = ["Categorical"]


@objtypedec
class Categorical:
    """
    Represents an array of values belonging to named categories. Converting a
    Strings object to Categorical often saves memory and speeds up operations,
    especially if there are many repeated values, at the cost of some one-time
    work in initialization.

    Parameters
    ----------
    values : Strings
        String values to convert to categories
    NAvalue : str scalar
        The value to use to represent missing/null data

    Attributes
    ----------
    categories : Strings
        The set of category labels (determined automatically)
    codes : pdarray, int64
        The category indices of the values or -1 for N/A
    permutation : pdarray, int64
        The permutation that groups the values in the same order as categories
    segments : pdarray, int64
        When values are grouped, the starting offset of each group
    size : Union[int,np.int64]
        The number of items in the array
    nlevels : Union[int,np.int64]
        The number of distinct categories
    ndim : Union[int,np.int64]
        The rank of the array (currently only rank 1 arrays supported)
    shape : tuple
        The sizes of each dimension of the array

    """

    BinOps = frozenset(["==", "!="])
    RegisterablePieces = frozenset(["categories", "codes", "permutation", "segments", "_akNAcode"])
    RequiredPieces = frozenset(["categories", "codes", "_akNAcode"])
    permutation = None
    segments = None

    def __init__(self, values, **kwargs) -> None:
        self.logger = getArkoudaLogger(name=__class__.__name__)  # type: ignore
        if "codes" in kwargs and "categories" in kwargs:
            # This initialization is called by Categorical.from_codes()
            # The values arg is ignored
            self.codes = kwargs["codes"]
            self.categories = kwargs["categories"]
            if (self.codes.min() < 0) or (self.codes.max() >= self.categories.size):
                raise ValueError(
                    f"Codes out of bounds for categories: min = {self.codes.min()},"
                    f" max = {self.codes.max()}, categories = {self.categories.size}"
                )
            self.permutation = kwargs.get("permutation", None)
            self.segments = kwargs.get("segments", None)
            if self.permutation is not None and self.segments is not None:
                # Permutation and segments should only ever be supplied together from
                # the .from_codes() method, not user input
                self.permutation = cast(pdarray, self.permutation)
                self.segments = cast(pdarray, self.segments)
                unique_codes = self.codes[self.permutation[self.segments]]
            else:
                unique_codes = unique(self.codes)
            self._categories_used = self.categories[unique_codes]
        else:
            # Typical initialization, called with values
            if not isinstance(values, Strings):
                raise ValueError(("Categorical: inputs other than " + "Strings not yet supported"))
            g = GroupBy(values)
            self.categories = g.unique_keys
            self.codes = g.broadcast(arange(self.categories.size), permute=True)
            self.permutation = cast(pdarray, g.permutation)
            self.segments = g.segments
            # Make a copy because N/A value must be added below
            self._categories_used = self.categories[:]

        # When read from file or attached, NA code will be passed as a pdarray
        # Otherwise, the NA value is set to a string
        if "_akNAcode" in kwargs:
            self._akNAcode = kwargs["_akNAcode"]
            self._NAcode = int(self._akNAcode[0])
            self.NAvalue = self.categories[self._NAcode]
        else:
            self.NAvalue = kwargs.get("NAvalue", "N/A")
            findNA = self.categories == self.NAvalue
            if findNA.any():
                self._NAcode = int(akcast(findNA, akint64).argmax())
            else:
                # Append NA value
                self.categories = concatenate((self.categories, array([self.NAvalue])))
                self._NAcode = self.categories.size - 1
            self._akNAcode = array([self._NAcode])
        # Always set these values
        self.size: int_scalars = self.codes.size
        self.nlevels = self.categories.size
        self.ndim = self.codes.ndim
        self.shape = self.codes.shape
        self.dtype = str_
        self.name: Optional[str] = None

    @property
    def objtype(self):
        return self.objtype

    @classmethod
    @typechecked
    def from_codes(
        cls, codes: pdarray, categories: Strings, permutation=None, segments=None, **kwargs
    ) -> Categorical:
        """
        Make a Categorical from codes and categories arrays. If codes and
        categories have already been pre-computed, this constructor saves
        time. If not, please use the normal constructor.

        Parameters
        ----------
        codes : pdarray, int64
            Category indices of each value
        categories : Strings
            Unique category labels
        permutation : pdarray, int64
            The permutation that groups the values in the same order
            as categories
        segments : pdarray, int64
            When values are grouped, the starting offset of each group

        Returns
        -------
        Categorical
           The Categorical object created from the input parameters

        Raises
        ------
        TypeError
            Raised if codes is not a pdarray of int64 objects or if
            categories is not a Strings object
        """
        if codes.dtype != akint64:
            raise TypeError("Codes must be pdarray of int64")
        return cls(
            None,
            codes=codes,
            categories=categories,
            permutation=permutation,
            segments=segments,
            **kwargs,
        )

    @classmethod
    def standardize_categories(cls, arrays, NAvalue="N/A"):
        """
        Standardize an array of Categoricals so that they share the same categories.

        Parameters
        ----------
        arrays : sequence of Categoricals
            The Categoricals to standardize
        NAvalue : str scalar
            The value to use to represent missing/null data

        Returns
        -------
        List of Categoricals
            A list of the original Categoricals remapped to the shared categories.
        """
        for arr in arrays:
            if not isinstance(arr, cls):
                raise TypeError(f"All arguments must be {cls.__name__}")
        new_categories = unique(concatenate([arr.categories for arr in arrays], ordered=False))
        findNA = new_categories == NAvalue
        if not findNA.any():
            # Append NA value
            new_categories = concatenate((new_categories, array([NAvalue])))
        return [arr.set_categories(new_categories, NAvalue=NAvalue) for arr in arrays]

    def set_categories(self, new_categories, NAvalue=None):
        """
        Set categories to user-defined values.

        Parameters
        ----------
        new_categories : Strings
            The array of new categories to use. Must be unique.
        NAvalue : str scalar
            The value to use to represent missing/null data

        Returns
        -------
        Categorical
            A new Categorical with the user-defined categories. Old values present
            in new categories will appear unchanged. Old values not present will
            be assigned the NA value.
        """
        if NAvalue is None:
            NAvalue = self.NAvalue
        findNA = new_categories == NAvalue
        if not findNA.any():
            # Append NA value
            new_categories = concatenate((new_categories, array([NAvalue])))
            NAcode = new_categories.size - 1
        else:
            NAcode = int(akcast(findNA, akint64).argmax())
        code_mapping = zeros(self.categories.size, dtype=akint64)
        code_mapping.fill(NAcode)
        # Concatenate old and new categories and unique codes
        bothcats = concatenate((self.categories, new_categories), ordered=False)
        bothcodes = concatenate(
            (arange(self.categories.size), arange(new_categories.size)), ordered=False
        )
        fromold = concatenate(
            (ones(self.categories.size, dtype=akbool), zeros(new_categories.size, dtype=akbool)),
            ordered=False,
        )
        # Group combined categories to find matches
        g = GroupBy(bothcats)
        ct = g.count()[1]
        if (ct > 2).any():
            raise ValueError("User-specified categories must be unique")
        # Matches have two hits in concatenated array
        present = g.segments[(ct == 2)]
        firstinds = g.permutation[present]
        firstcodes = bothcodes[firstinds]
        firstisold = fromold[firstinds]
        secondinds = g.permutation[present + 1]
        secondcodes = bothcodes[secondinds]
        # Matching pairs define a mapping of old codes to new codes
        scatterinds = where(firstisold, firstcodes, secondcodes)
        gatherinds = where(firstisold, secondcodes, firstcodes)
        # Make a lookup table where old code at scatterind maps to new code at gatherind
        code_mapping[scatterinds] = arange(new_categories.size)[gatherinds]
        # Apply the lookup to map old codes to new codes
        new_codes = code_mapping[self.codes]
        return self.__class__.from_codes(new_codes, new_categories, NAvalue=NAvalue)

    @staticmethod
    def from_return_msg(repMsg):
        """
        Return a categorical instance pointing to components created by the arkouda server.
        The user should not call this function directly.

        Parameters
        ----------
        repMsg : str
            ; delimited string containing the categories, codes, permutation, and segments
            details

        Returns
        -------
        categorical
            A categorical representing a set of strings and pdarray components on the server

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown in the process of creating
            the categorical instance
        """
        # parts[0] is "categorical". Used by the generic attach method to identify the
        # response message as a Categorical

        repParts = repMsg.split("+")
        stringsMsg = f"{repParts[1]}+{repParts[2]}"
        parts = {
            "categories": Strings.from_return_msg(stringsMsg),
            "codes": create_pdarray(repParts[3]),
            "_akNAcode": create_pdarray(repParts[4]),
        }

        if len(repParts) > 5:
            for i in range(5, len(repParts)):
                name = repParts[i].split()[1]
                if ".permutation" in name:
                    parts["permutation"] = create_pdarray(repParts[i])
                elif ".segments" in name:
                    parts["segments"] = create_pdarray(repParts[i])
                else:
                    raise ValueError(f"Unknown field, {name}, found in Categorical.")

        # To get the name split the message into Categories, Codes, Permutation, Segments
        # then split the categories into it's components, Name being second: name.categories
        # split the name on . and take the first half to get the given name
        # for example repParts[1] = "created user_defined_name.categories"
        name = repParts[1].split()[1].split(".")[0]

        c = Categorical(None, **parts)  # Call constructor with unpacked kwargs
        c.name = name  # Update our name
        return c

    def to_ndarray(self) -> np.ndarray:
        """
        Convert the array to a np.ndarray, transferring array data from
        the arkouda server to Python. This conversion discards category
        information and produces an ndarray of strings. If the arrays
        exceeds a built-in size limit, a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            A numpy ndarray of strings corresponding to the values in
            this array

        Notes
        -----
        The number of bytes in the array cannot exceed ``arkouda.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.maxTransferBytes to a larger
        value, but proceed with caution.
        """
        if self.categories.size > self.codes.size:
            newcat = self.reset_categories()
            idx = newcat.categories.to_ndarray()
            valcodes = newcat.codes.to_ndarray()
        else:
            idx = self.categories.to_ndarray()
            valcodes = self.codes.to_ndarray()
        return idx[valcodes]

    def __iter__(self):
        raise NotImplementedError(
            "Categorical does not support iteration. To force data transfer from server, use to_ndarray"
        )

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        # limit scope of import to pick up changes to global variable
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            vals = [f"'{self[i]}'" for i in range(self.size)]
        else:
            vals = [f"'{self[i]}'" for i in range(3)]
            vals.append("... ")
            vals.extend([self[i] for i in range(self.size - 3, self.size)])
        return "[{}]".format(", ".join(vals))

    def __repr__(self):
        return f"array({self.__str__()})"

    @typechecked
    def _binop(self, other: Union[Categorical, str_scalars], op: str_scalars) -> pdarray:
        """
        Executes the requested binop on this Categorical instance and returns
        the results within a pdarray object.

        Parameters
        ----------
        other : Union[Categorical,str_scalars]
            the other object is a Categorical object or string scalar
        op : str_scalars
            name of the binary operation to be performed

        Returns
        -------
        pdarray
            encapsulating the results of the requested binop

        Raises
        -----
        ValueError
            Raised if (1) the op is not in the self.BinOps set, or (2) if the
            sizes of this and the other instance don't match
        RuntimeError
            Raised if a server-side error is thrown while executing the
            binary operation
        """
        if op not in self.BinOps:
            raise NotImplementedError(f"Categorical: unsupported operator: {op}")
        if np.isscalar(other) and resolve_scalar_dtype(other) == "str":
            idxresult = self.categories._binop(other, op)
            return idxresult[self.codes]
        if self.size != cast(Categorical, other).size:
            raise ValueError(
                f"Categorical {op}: size mismatch {self.size} {cast(Categorical, other).size}"
            )
        if isinstance(other, Categorical):
            if (self.categories.size == other.categories.size) and (
                self.categories == other.categories
            ).all():
                # Because categories are identical, codes can be compared directly
                return self.codes._binop(other.codes, op)
            else:
                tmpself, tmpother = self.standardize_categories((self, other))
                return tmpself.codes._binop(tmpother.codes, op)
        else:
            raise NotImplementedError(
                "Operations between Categorical and non-Categorical not yet implemented."
                "Consider converting operands to Categorical."
            )

    @typechecked
    def _r_binop(self, other: Union[Categorical, str_scalars], op: str_scalars) -> pdarray:
        """
        Executes the requested reverse binop on this Categorical instance and
        returns the results within a pdarray object.

        Parameters
        ----------
        other : Union[Categorical,str_scalars]
            the other object is a Categorical object or string scalar
        op : str_scalars
            name of the binary operation to be performed

        Returns
        -------
        pdarray
            encapsulating the results of the requested binop

        Raises
        -----
        ValueError
            Raised if (1) the op is not in the self.BinOps set, or (2) if the
            sizes of this and the other instance don't match
        RuntimeError
            Raised if a server-side error is thrown while executing the
            binary operation
        """
        return self._binop(other, op)

    def __eq__(self, other):
        return self._binop(other, "==")

    def __ne__(self, other):
        return self._binop(other, "!=")

    def __getitem__(self, key) -> Categorical:
        if np.isscalar(key) and resolve_scalar_dtype(key) == "int64":
            return self.categories[self.codes[key]]
        else:
            # Don't reset categories because they might have been user-defined
            # Initialization now determines which categories are used
            return Categorical.from_codes(self.codes[key], self.categories)

    def isna(self):
        """
        Find where values are missing or null (as defined by self.NAvalue)
        """
        return self.codes == self._NAcode

    def reset_categories(self) -> Categorical:
        """
        Recompute the category labels, discarding any unused labels. This
        method is often useful after slicing or indexing a Categorical array,
        when the resulting array only contains a subset of the original
        categories. In this case, eliminating unused categories can speed up
        other operations.

        Returns
        -------
        Categorical
            A Categorical object generated from the current instance
        """
        g = GroupBy(self.codes)
        idx = self.categories[g.unique_keys]
        newvals = g.broadcast(arange(idx.size), permute=True)
        return Categorical.from_codes(
            newvals, idx, permutation=g.permutation, segments=g.segments, NAvalue=self.NAvalue
        )

    @typechecked
    def contains(self, substr: str) -> pdarray:
        """
        Check whether each element contains the given substring.

        Parameters
        ----------
        substr : str
            The substring to search for

        Returns
        -------
        pdarray, bool
            True for elements that contain substr, False otherwise

        Raises
        ------
        TypeError
            Raised if substr is not a str

        Notes
        -----
        This method can be significantly faster than the corresponding method
        on Strings objects, because it searches the unique category labels
        instead of the full array.

        See Also
        --------
        Categorical.startswith, Categorical.endswith
        """
        categoriescontains = self.categories.contains(substr)
        return categoriescontains[self.codes]

    @typechecked
    def startswith(self, substr: str) -> pdarray:
        """
        Check whether each element starts with the given substring.

        Parameters
        ----------
        substr : str
            The substring to search for

        Raises
        ------
        TypeError
            Raised if substr is not a str

        Returns
        -------
        pdarray, bool
            True for elements that contain substr, False otherwise

        Notes
        -----
        This method can be significantly faster than the corresponding
        method on Strings objects, because it searches the unique category
        labels instead of the full array.

        See Also
        --------
        Categorical.contains, Categorical.endswith
        """
        categoriesstartswith = self.categories.startswith(substr)
        return categoriesstartswith[self.codes]

    @typechecked
    def endswith(self, substr: str) -> pdarray:
        """
        Check whether each element ends with the given substring.

        Parameters
        ----------
        substr : str
            The substring to search for

        Raises
        ------
        TypeError
            Raised if substr is not a str

        Returns
        -------
        pdarray, bool
            True for elements that contain substr, False otherwise

        Notes
        -----
        This method can be significantly faster than the corresponding method
        on Strings objects, because it searches the unique category labels
        instead of the full array.

        See Also
        --------
        Categorical.startswith, Categorical.contains
        """
        categoriesendswith = self.categories.endswith(substr)
        return categoriesendswith[self.codes]

    @typechecked
    def in1d(self, test: Union[Strings, Categorical]) -> pdarray:
        """
        Test whether each element of the Categorical object is
        also present in the test Strings or Categorical object.

        Returns a boolean array the same length as `self` that is True
        where an element of `self` is in `test` and False otherwise.

        Parameters
        ----------
        test : Union[Strings,Categorical]
            The values against which to test each value of 'self`.

        Returns
        -------
        pdarray, bool
            The values `self[in1d]` are in the `test` Strings or Categorical object.

        Raises
        ------
        TypeError
            Raised if test is not a Strings or Categorical object

        See Also
        --------
        unique, intersect1d, union1d

        Notes
        -----
        `in1d` can be considered as an element-wise function version of the
        python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is logically
        equivalent to ``ak.array([item in b for item in a])``, but is much
        faster and scales to arbitrarily large ``a``.


        Examples
        --------
        >>> strings = ak.array([f'String {i}' for i in range(0,5)])
        >>> cat = ak.Categorical(strings)
        >>> ak.in1d(cat,strings)
        array([True, True, True, True, True])
        >>> strings = ak.array([f'String {i}' for i in range(5,9)])
        >>> catTwo = ak.Categorical(strings)
        >>> ak.in1d(cat,catTwo)
        array([False, False, False, False, False])
        """
        if isinstance(test, Categorical):
            # Must use test._categories_used instead of test.categories to avoid
            # falsely returning True when a value is present in test.categories
            # but not used in the array. On the other hand, we don't need to use
            # self._categories_used, because indexing with [self.codes] below ensures
            # that only results for categories used in self.codes will be returned.
            categoriesisin = in1d(self.categories, test._categories_used)
        else:
            categoriesisin = in1d(self.categories, test)
        return categoriesisin[self.codes]

    def unique(self) -> Categorical:
        # __doc__ = unique.__doc__
        return Categorical.from_codes(
            arange(self._categories_used.size), self._categories_used, NAvalue=self.NAvalue
        )

    def group(self) -> pdarray:
        """
        Return the permutation that groups the array, placing equivalent
        categories together. All instances of the same category are guaranteed
        to lie in one contiguous block of the permuted array, but the blocks
        are not necessarily ordered.

        Returns
        -------
        pdarray
            The permutation that groups the array by value

        See Also
        --------
        GroupBy, unique

        Notes
        -----
        This method is faster than the corresponding Strings method. If the
        Categorical was created from a Strings object, then this function
        simply returns the cached permutation. Even if the Categorical was
        created using from_codes(), this function will be faster than
        Strings.group() because it sorts dense integer values, rather than
        128-bit hash values.
        """
        if self.permutation is None:
            return argsort(self.codes)
        else:
            return self.permutation

    def _get_grouping_keys(self):
        """
        Private method for generating grouping keys used by GroupBy.

        API: this method must be defined by all groupable arrays, and it
        must return a list of arrays that can be (co)argsorted.
        """
        return [self.codes]

    def argsort(self):
        # __doc__ = argsort.__doc__
        idxperm = argsort(self.categories)
        inverse = zeros_like(idxperm)
        inverse[idxperm] = arange(idxperm.size)
        newvals = inverse[self.codes]
        return argsort(newvals)

    def sort(self):
        # __doc__ = sort.__doc__
        idxperm = argsort(self.categories)
        inverse = zeros_like(idxperm)
        inverse[idxperm] = arange(idxperm.size)
        newvals = inverse[self.codes]
        return Categorical.from_codes(newvals, self.categories[idxperm])

    @typechecked
    def concatenate(self, others: Sequence[Categorical], ordered: bool = True) -> Categorical:
        """
        Merge this Categorical with other Categorical objects in the array,
        concatenating the arrays and synchronizing the categories.

        Parameters
        ----------
        others : Sequence[Categorical]
            The Categorical arrays to concatenate and merge with this one
        ordered : bool
            If True (default), the arrays will be appended in the
            order given. If False, array data may be interleaved
            in blocks, which can greatly improve performance but
            results in non-deterministic ordering of elements.

        Returns
        -------
        Categorical
            The merged Categorical object

        Raises
        ------
        TypeError
            Raised if any others array objects are not Categorical objects

        Notes
        -----
        This operation can be expensive -- slower than concatenating Strings.
        """
        if isinstance(others, Categorical):
            others = [others]
        elif len(others) < 1:
            return self
        samecategories = True
        for c in others:
            if not isinstance(c, Categorical):
                raise TypeError("Categorical: can only merge/concatenate with other Categoricals")
            if (self.categories.size != c.categories.size) or not (
                self.categories == c.categories
            ).all():
                samecategories = False
        if samecategories:
            newvals = cast(
                pdarray, concatenate([self.codes] + [o.codes for o in others], ordered=ordered)
            )
            return Categorical.from_codes(newvals, self.categories)
        else:
            new_arrays = self.standardize_categories([self] + list(others), NAvalue=self.NAvalue)
            new_categories = new_arrays[0].categories
            new_codes = cast(pdarray, concatenate([arr.codes for arr in new_arrays], ordered=ordered))
            return Categorical.from_codes(new_codes, new_categories, NAvalue=self.NAvalue)

    @typechecked
    def save(
        self,
        prefix_path: str,
        dataset: str = "categorical_array",
        file_format: str = "HDF5",
        mode: str = "truncate",
    ) -> str:
        """
        Save the Categorical object to HDF5. The result is a collection of HDF5 files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path and dataset. Each locale saves its chunk of the Strings array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in HDF5 files (must not already exist)
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', create a new Categorical dataset within existing files.

        Returns
        -------
        String message indicating result of save operation

        Raises
        ------
        ValueError
            Raised if the lengths of columns and values differ, or the mode is
            neither 'truncate' nor 'append'
        TypeError
            Raised if prefix_path, dataset, or mode is not a str

        See Also
        --------
        pdarrayIO.save, pdarrayIO.load_all

        Notes
        -----
        Important implementation notes: (1) Strings state is saved as two datasets
        within an hdf5 group: one for the string characters and one for the
        segments corresponding to the start of each string, (2) the hdf5 group is named
        via the dataset parameter.
        """
        if mode.lower() not in ["append", "truncate"]:
            raise ValueError("Allowed modes are 'truncate' and 'append'")

        result = []
        comp_dict = {k: v for k, v in self._get_components_dict().items() if v is not None}

        if self.RequiredPieces.issubset(comp_dict.keys()):
            # Honor the first mode but switch to append for all others
            # since each following comp may wipe out the file
            first = True
            for k, v in comp_dict.items():
                result.append(
                    v.save(
                        prefix_path,
                        dataset=f"{dataset}.{k}",
                        file_format=file_format,
                        mode=(mode if first else "append"),
                    )
                )
                first = False
        else:
            raise Exception(
                "The required pieces of `categories` and `codes` were not populated on this Categorical"
            )
        return ";".join(result)

    @typechecked()
    def register(self, user_defined_name: str) -> Categorical:
        """
        Register this Categorical object and underlying components with the Arkouda server

        Parameters
        ----------
        user_defined_name : str
            user defined name the Categorical is to be registered under,
            this will be the root name for underlying components

        Returns
        -------
        Categorical
            The same Categorical which is now registered with the arkouda server and has an updated name.
            This is an in-place modification, the original is returned to support
            a fluid programming style.
            Please note you cannot register two different Categoricals with the same name.

        Raises
        ------
        TypeError
            Raised if user_defined_name is not a str
        RegistrationError
            If the server was unable to register the Categorical with the user_defined_name

        See also
        --------
        unregister, attach, unregister_categorical_by_name, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        [
            p.register(f"{user_defined_name}.{n}")
            for n, p in Categorical._get_components_dict(self).items()
        ]
        self.name = user_defined_name
        return self

    def unregister(self) -> None:
        """
        Unregister this Categorical object in the arkouda server which was previously
        registered using register() and/or attached to using attach()

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See also
        --------
        register, attach, unregister_categorical_by_name, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        if not self.name:
            raise RegistrationError(
                "This item does not have a name and does not appear to be registered."
            )
        [p.unregister() for p in Categorical._get_components_dict(self).values()]
        self.name = None  # Clear our internal Categorical object name

    def is_registered(self) -> np.bool_:
        """
         Return True iff the object is contained in the registry

        Returns
        -------
        numpy.bool
            Indicates if the object is contained in the registry

        Raises
        ------
        RegistrationError
            Raised if there's a server-side error or a mis-match of registered components

        See Also
        --------
        register, attach, unregister, unregister_categorical_by_name

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.
        """
        parts_registered: List[np.bool_] = [
            p.is_registered() for p in Categorical._get_components_dict(self).values()
        ]
        if np.any(parts_registered) and not np.all(parts_registered):  # test for error
            raise RegistrationError(
                f"Not all registerable components of Categorical {self.name} are registered."
            )

        return np.bool_(np.any(parts_registered))

    def _get_components_dict(self) -> Dict:
        """
        Internal function that returns a dictionary with all required or non-None components of self

        Required Categorical components (Codes and Categories) are always included in
        returned components_dict
        Optional Categorical components (Permutation and Segments) are only included if
        they've been set (are not None)

        Returns
        -------
        Dict
            Dictionary of all required or non-None components of self
                Keys: component names (Codes, Categories, Permutation, Segments)
                Values: components of self
        """
        return {
            piece_name: getattr(self, piece_name)
            for piece_name in Categorical.RegisterablePieces
            if piece_name in Categorical.RequiredPieces or getattr(self, piece_name) is not None
        }

    def _list_component_names(self) -> List[str]:
        """
        Internal function that returns a list of all component names

        Parameters
        ----------
        None

        Returns
        -------
        List[str]
            List of all component names
        """
        return list(
            itertools.chain.from_iterable(
                [p._list_component_names() for p in Categorical._get_components_dict(self).values()]
            )
        )

    def info(self) -> str:
        """
        Returns a JSON formatted string containing information about all components of self

        Parameters
        ----------
        None

        Returns
        -------
        str
            JSON string containing information about all components of self
        """
        return information(self._list_component_names())

    def pretty_print_info(self) -> None:
        """
        Prints information about all components of self in a human readable format

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        [p.pretty_print_info() for p in Categorical._get_components_dict(self).values()]

    @staticmethod
    @typechecked
    def attach(user_defined_name: str) -> Categorical:
        """
        Function to return a Categorical object attached to the registered name in the
        arkouda server which was registered using register()

        Parameters
        ----------
        user_defined_name : str
            user defined name which Categorical object was registered under

        Returns
        -------
        Categorical
            The Categorical object created by re-attaching to the corresponding server components

        Raises
        ------
        TypeError
            if user_defined_name is not a string

        See Also
        --------
        register, is_registered, unregister, unregister_categorical_by_name
        """
        # Build dict of registered components by invoking their corresponding Class.attach functions
        parts = {
            "categories": Strings.attach(f"{user_defined_name}.categories"),
            "codes": pdarray.attach(f"{user_defined_name}.codes"),
            "_akNAcode": pdarray.attach(f"{user_defined_name}._akNAcode"),
        }

        # Add optional pieces only if they're contained in the registry
        registry = list_registry()
        if f"{user_defined_name}.permutation" in registry:
            parts["permutation"] = pdarray.attach(f"{user_defined_name}.permutation")
        if f"{user_defined_name}.segments" in registry:
            parts["segments"] = pdarray.attach(f"{user_defined_name}.segments")

        c = Categorical(None, **parts)  # Call constructor with unpacked kwargs
        c.name = user_defined_name  # Update our name
        return c

    @staticmethod
    @typechecked
    def unregister_categorical_by_name(user_defined_name: str) -> None:
        """
        Function to unregister Categorical object by name which was registered
        with the arkouda server via register()

        Parameters
        ----------
        user_defined_name : str
            Name under which the Categorical object was registered

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
        # We have 4 subcomponents, unregister each of them
        Strings.unregister_strings_by_name(f"{user_defined_name}.categories")
        unregister_pdarray_by_name(f"{user_defined_name}.codes")
        unregister_pdarray_by_name(f"{user_defined_name}._akNAcode")

        # Unregister optional pieces only if they are contained in the registry
        registry = list_registry()
        if f"{user_defined_name}.permutation" in registry:
            unregister_pdarray_by_name(f"{user_defined_name}.permutation")
        if f"{user_defined_name}.segments" in registry:
            unregister_pdarray_by_name(f"{user_defined_name}.segments")

    @staticmethod
    @typechecked
    def parse_hdf_categoricals(
        d: Mapping[str, Union[pdarray, Strings]]
    ) -> Tuple[List[str], Dict[str, Categorical]]:
        """
        This function should be used in conjunction with the load_all function which reads hdf5 files
        and reconstitutes Categorical objects.
        Categorical objects use a naming convention and HDF5 structure so they can be identified and
        constructed for the user.

        In general you should not call this method directly

        Parameters
        ----------
        d : Dictionary of String to either Pdarray or Strings object

        Returns
        -------
        2-Tuple of List of strings containing key names which should be removed and Dictionary of
        base name to Categorical object

        See Also
        --------
        Categorical.save, load_all

        """
        removal_names: List[str] = []
        groups: DefaultDict[str, List[str]] = defaultdict(list)
        result_categoricals: Dict[str, Categorical] = {}
        for k in d.keys():  # build dict of str->list[components]
            if "." in k:
                groups[k.split(".")[0]].append(k)

        # for each of the groups, find categorical by testing values in the group for ".categories"
        for k, v in groups.items():  # str->list[str]
            if any([i.endswith(".categories") for i in v]):  # we have a categorical
                # gather categorical pieces and replace the original mapping with the categorical object
                cat_parts = {}
                base_name = ""
                for part in v:
                    removal_names.append(part)  # flag it for removal from original
                    cat_parts[part.split(".")[-1]] = d[part]  # put the part into our categorical parts
                    if part.endswith(".categories"):
                        base_name = ".".join(part.split(".categories")[0:-1])

                # Construct categorical and add it to the return_categoricals under the parent name
                result_categoricals[base_name] = Categorical.from_codes(**cat_parts)

        return removal_names, result_categoricals
