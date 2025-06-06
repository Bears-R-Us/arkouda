from __future__ import annotations

import itertools
import json
from collections import defaultdict
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from typing import cast as type_cast

import numpy as np
from pandas import Categorical as pd_Categorical
from pandas import Index as pd_Index
from typeguard import typechecked

from arkouda.client import generic_msg
from arkouda.groupbyclass import GroupBy, unique
from arkouda.infoclass import information
from arkouda.logger import getArkoudaLogger
from arkouda.numpy import cast as akcast
from arkouda.numpy import where
from arkouda.numpy.dtypes import bool_ as akbool
from arkouda.numpy.dtypes import bool_scalars
from arkouda.numpy.dtypes import dtype as akdtype
from arkouda.numpy.dtypes import int64 as akint64
from arkouda.numpy.dtypes import int_scalars, resolve_scalar_dtype, str_, str_scalars
from arkouda.numpy.pdarrayclass import RegistrationError
from arkouda.numpy.pdarrayclass import all as akall
from arkouda.numpy.pdarrayclass import create_pdarray, pdarray
from arkouda.numpy.pdarraycreation import arange, array, ones, zeros, zeros_like
from arkouda.numpy.pdarraysetops import concatenate, in1d
from arkouda.numpy.sorting import argsort
from arkouda.numpy.sorting import sort as pda_sort
from arkouda.numpy.strings import Strings

__all__ = ["Categorical"]


class Categorical:
    """
    Represents an array of values belonging to named categories.

    Converting a Strings object to Categorical often saves memory and speeds up operations,
    especially if there are many repeated values, at the cost of some one-time
    work in initialization.

    Parameters
    ----------
    values : Strings, Categorical, pd.Categorical
        Values to convert to categories
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
    segments : Union[pdarray, None]
        When values are grouped, the starting offset of each group
    size : int_scalars
        The number of items in the array
    nlevels : int_scalars
        The number of distinct categories
    ndim : int_scalars
        The rank of the array (currently only rank 1 arrays supported)
    shape : tuple
        The sizes of each dimension of the array

    """

    categories: Strings
    codes: pdarray
    permutation: Union[pdarray, None]
    segments: Union[pdarray, None]
    size: int_scalars
    nlevels: int_scalars
    ndim: int_scalars
    shape: tuple

    BinOps = frozenset(["==", "!="])
    RegisterablePieces = frozenset(["categories", "codes", "permutation", "segments", "_akNAcode"])
    RequiredPieces = frozenset(["categories", "codes", "_akNAcode"])
    segments = None
    objType = "Categorical"
    dtype = akdtype(str_)  # this is being set for now because Categoricals only supported on Strings

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
                self.permutation = type_cast(pdarray, self.permutation)
                self.segments = type_cast(pdarray, self.segments)
                unique_codes = self.codes[self.permutation[self.segments]]
            else:
                unique_codes = unique(self.codes)
            self._categories_used = self.categories[unique_codes]
        else:
            # Typical initialization, called with values
            if isinstance(values, pd_Categorical):
                self.categories = type_cast(Strings, array(values.categories))
                self.codes = type_cast(pdarray, array(values.codes.astype("int64")))
                self._categories_used = self.categories[unique(self.codes)]
                self.permutation = None
                self.segments = None
            elif isinstance(values, Categorical):
                self.categories = values.categories
                self.codes = values.codes
                self._categories_used = values._categories_used
                self.permutation = values.permutation
                self.segments = values.segments
            elif isinstance(values, Strings):
                g = GroupBy(values)
                if not isinstance(g.unique_keys, Strings):
                    raise TypeError(f"expected Strings, got {type(g.unique_keys).__name__!r}")
                self.categories = g.unique_keys
                self.codes = g.broadcast(arange(self.categories.size), permute=True)
                self.permutation = type_cast(pdarray, g.permutation)
                self.segments = g.segments
                # Make a copy because N/A value must be added below
                self._categories_used = self.categories[:]
            else:
                raise ValueError(
                    ("Categorical: inputs other than " + "Strings or pd.Categorical not yet supported")
                )
        # When read from file or attached, NA code will be passed as a pdarray
        # Otherwise, the NA value is set to a string
        if "_akNAcode" in kwargs and kwargs["_akNAcode"] is not None:
            self._akNAcode = kwargs["_akNAcode"]
            self._NAcode: int_scalars = int(self._akNAcode[0])
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
        self.dtype = akdtype(str_)
        self.registered_name: Optional[str] = None

    @property
    def nbytes(self):
        """
        The size of the Categorical in bytes.

        Returns
        -------
        int
            The size of the Categorical in bytes.

        """
        nbytes = 0
        if self.categories is not None:
            nbytes += self.categories.nbytes

        if isinstance(self.codes, pdarray):
            nbytes += self.codes.nbytes
        elif isinstance(self.codes, akdtype("int64")):
            nbytes += 1

        if isinstance(self.permutation, pdarray):
            nbytes += self.permutation.nbytes
        elif isinstance(self.permutation, akdtype("int64")):
            nbytes += 1

        if isinstance(self.segments, pdarray):
            nbytes += self.segments.nbytes
        elif isinstance(self.segments, akdtype("int64")):
            nbytes += 1

        return nbytes

    @property
    def inferred_type(self) -> str:
        """Return a string of the type inferred from the values."""
        return "categorical"

    @classmethod
    @typechecked
    def from_codes(
        cls, codes: pdarray, categories: Strings, permutation=None, segments=None, **kwargs
    ) -> Categorical:
        """
        Make a Categorical from codes and categories arrays.

        If codes and
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
    def from_return_msg(cls, rep_msg) -> Categorical:
        """
        Create categorical from return message from server.

        Notes
        -----
        This is currently only used when reading a Categorical from HDF5 files.

        """
        # parse return json
        eles = json.loads(rep_msg)
        codes = create_pdarray(eles["codes"])
        cats = Strings.from_return_msg(eles["categories"])
        na_code = create_pdarray(eles["_akNAcode"])

        segments = None
        perm = None
        if "segments" in eles and "permutation" in eles:
            segments = create_pdarray(eles["segments"])
            perm = create_pdarray(eles["permutation"])

        return cls.from_codes(codes, cats, permutation=perm, segments=segments, _akNAcode=na_code)

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

    def equals(self, other) -> bool_scalars:
        """
        Whether Categoricals are the same size and all entries are equal.

        Parameters
        ----------
        other : object
            object to compare.

        Returns
        -------
        bool_scalars
            True if the Categoricals are the same, o.w. False.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> c = Categorical(ak.array(["a", "b", "c"]))
        >>> c_cpy = Categorical(ak.array(["a", "b", "c"]))
        >>> c.equals(c_cpy)
        np.True_
        >>> c2 = Categorical(ak.array(["a", "x", "c"]))
        >>> c.equals(c2)
        np.False_

        """
        if isinstance(other, Categorical):
            if other.size != self.size:
                return False
            else:
                result = akall(self == other)
                if isinstance(result, (bool, np.bool_)):
                    return result

        return False

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
        ct = g.size()[1]
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

    def to_ndarray(self) -> np.ndarray:
        """
        Convert the array to a np.ndarray.

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
        The number of bytes in the array cannot exceed ``ak.client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.client.maxTransferBytes to a larger
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

    def to_pandas(self) -> pd_Categorical:
        """Return the equivalent Pandas Categorical."""
        return pd_Categorical.from_codes(
            codes=type_cast(List[int], self.codes.to_list()),
            categories=pd_Index(self.categories.to_ndarray()),
        )

    def to_list(self) -> List[str]:
        """
        Convert the Categorical to a list.

        Convert the Categorical to a list, transferring data from
        the arkouda server to Python. This conversion discards category
        information and produces a list of strings. If the arrays
        exceeds a built-in size limit, a RuntimeError is raised.

        Returns
        -------
        List[str]
            A list of strings corresponding to the values in
            this Categorical

        Notes
        -----
        The number of bytes in the Categorical cannot exceed ``ak.client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.client.maxTransferBytes to a larger
        value, but proceed with caution.

        """
        return type_cast(List[str], self.to_ndarray().tolist())

    def to_strings(self) -> Strings:
        """
        Convert the Categorical to Strings.

        Returns
        -------
        Strings
            A Strings object corresponding to the values in
            this Categorical.

        Examples
        --------
        >>> import arkouda as ak
        >>> ak.connect()
        >>> a = ak.array(["a","b","c"])
        >>> a
        array(['a', 'b', 'c'])
        >>> c = ak.Categorical(a)
        >>> c.to_strings()
        array(['a', 'b', 'c'])

        >>> isinstance(c.to_strings(), ak.Strings)
        True

        """
        return self.categories[self.codes]

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
            vals.extend([f"'{self[i]}'" for i in range(self.size - 3, self.size)])
        return "[{}]".format(", ".join(vals))

    def __repr__(self):
        return f"array({self.__str__()})"

    @typechecked
    def _binop(self, other: Union[Categorical, str_scalars], op: str_scalars) -> pdarray:
        """
        Execute the binop.

        Execute the requested binop on this Categorical instance
        and returns the results within a pdarray object.

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
        ------
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
        if self.size != type_cast(Categorical, other).size:
            raise ValueError(
                f"Categorical {op}: size mismatch {self.size} {type_cast(Categorical, other).size}"
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
        Execute the reverse binop.

        Execute the requested reverse binop on this Categorical instance
        and return the results within a pdarray object.

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
        ------
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
        if np.isscalar(key) and (resolve_scalar_dtype(key) in ["int64", "uint64"]):
            return self.categories[self.codes[key]]
        else:
            # Don't reset categories because they might have been user-defined
            # Initialization now determines which categories are used
            return Categorical.from_codes(self.codes[key], self.categories)

    def isna(self):
        """Find where values are missing or null (as defined by self.NAvalue)."""
        return self.codes == self._NAcode

    def reset_categories(self) -> Categorical:
        """
        Recompute the category labels, discarding any unused labels.

        This method is often useful after slicing or indexing a Categorical array,
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
    def contains(self, substr: Union[bytes, str_scalars], regex: bool = False) -> pdarray:
        """
        Check whether each element contains the given substring.

        Parameters
        ----------
        substr : Union[bytes, str_scalars]
            The substring to search for
        regex: bool
            Indicates whether substr is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        pdarray
            True for elements that contain substr, False otherwise

        Raises
        ------
        TypeError
            Raised if the substr parameter is not bytes or str_scalars
        ValueError
            Rasied if substr is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Categorical.startswith, Categorical.endswith

        Notes
        -----
        This method can be significantly faster than the corresponding method
        on Strings objects, because it searches the unique category labels
        instead of the full array.

        """
        categories_contains = self.categories.contains(substr, regex)
        return categories_contains[self.codes]

    @typechecked
    def startswith(self, substr: Union[bytes, str_scalars], regex: bool = False) -> pdarray:
        """
        Check whether each element starts with the given substring.

        Parameters
        ----------
        substr : Union[bytes, str_scalars]
            The substring to search for
        regex: bool
            Indicates whether substr is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        pdarray
            True for elements that start with substr, False otherwise

        Raises
        ------
        TypeError
            Raised if the substr parameter is not bytes or str_scalars
        ValueError
            Rasied if substr is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Categorical.contains, Categorical.endswith

        Notes
        -----
        This method can be significantly faster than the corresponding
        method on Strings objects, because it searches the unique category
        labels instead of the full array.

        """
        categories_ends_with = self.categories.startswith(substr, regex)
        return categories_ends_with[self.codes]

    @typechecked
    def endswith(self, substr: Union[bytes, str_scalars], regex: bool = False) -> pdarray:
        """
        Check whether each element ends with the given substring.

        Parameters
        ----------
        substr : Union[bytes, str_scalars]
            The substring to search for
        regex: bool
            Indicates whether substr is a regular expression
            Note: only handles regular expressions supported by re2
            (does not support lookaheads/lookbehinds)

        Returns
        -------
        pdarray
            True for elements that end with substr, False otherwise

        Raises
        ------
        TypeError
            Raised if the substr parameter is not bytes or str_scalars
        ValueError
            Rasied if substr is not a valid regex
        RuntimeError
            Raised if there is a server-side error thrown

        See Also
        --------
        Categorical.startswith, Categorical.contains

        Notes
        -----
        This method can be significantly faster than the corresponding method
        on Strings objects, because it searches the unique category labels
        instead of the full array.

        """
        categories_ends_with = self.categories.endswith(substr, regex)
        return categories_ends_with[self.codes]

    @typechecked
    def in1d(self, test: Union[Strings, Categorical]) -> pdarray:
        """
        Whether each element is also present in the test Strings or Categorical object.

        Returns a boolean array the same length as `self` that is True
        where an element of `self` is in `test` and False otherwise.

        Parameters
        ----------
        test : Union[Strings,Categorical]
            The values against which to test each value of 'self`.

        Returns
        -------
        pdarray
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
        >>> import arkouda as ak
        >>> strings = ak.array([f'String {i}' for i in range(0,5)])
        >>> cat = ak.Categorical(strings)
        >>> ak.in1d(cat,strings)
        array([True True True True True])
        >>> strings = ak.array([f'String {i}' for i in range(5,9)])
        >>> catTwo = ak.Categorical(strings)
        >>> ak.in1d(cat,catTwo)
        array([False False False False False])

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

    def hash(self) -> Tuple[pdarray, pdarray]:
        """
        Compute a 128-bit hash of each element of the Categorical.

        Returns
        -------
        Tuple[pdarray,pdarray]
            A tuple of two int64 pdarrays. The ith hash value is the concatenation
            of the ith values from each array.

        Notes
        -----
        The implementation uses SipHash128, a fast and balanced hash function (used
        by Python for dictionaries and sets). For realistic numbers of strings (up
        to about 10**15), the probability of a collision between two 128-bit hash
        values is negligible.

        """
        rep_msg = generic_msg(
            cmd="categoricalHash",
            args={"objType": self.objType, "categories": self.categories, "codes": self.codes},
        )
        hashes = json.loads(rep_msg)
        return create_pdarray(hashes["upperHash"]), create_pdarray(hashes["lowerHash"])

    def group(self) -> pdarray:
        """
        Return the permutation that groups the array, placing equivalent categories together.

        All instances of the same category are guaranteed
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

    def sort_values(self):
        # __doc__ = sort.__doc__
        idxperm = argsort(self.categories)
        inverse = zeros_like(idxperm)
        inverse[idxperm] = arange(idxperm.size)
        newvals = inverse[self.codes]
        return Categorical.from_codes(pda_sort(newvals), self.categories[idxperm])

    @typechecked
    def concatenate(self, others: Sequence[Categorical], ordered: bool = True) -> Categorical:
        """
        Merge this Categorical with other Categorical objects in the array.

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
            newvals = type_cast(
                pdarray, concatenate([self.codes] + [o.codes for o in others], ordered=ordered)
            )
            return Categorical.from_codes(newvals, self.categories)
        else:
            new_arrays = self.standardize_categories([self] + list(others), NAvalue=self.NAvalue)
            new_categories = new_arrays[0].categories
            new_codes = type_cast(
                pdarray, concatenate([arr.codes for arr in new_arrays], ordered=ordered)
            )
            return Categorical.from_codes(new_codes, new_categories, NAvalue=self.NAvalue)

    def to_hdf(
        self,
        prefix_path,
        dataset="categorical_array",
        mode="truncate",
        file_type="distribute",
    ):
        """
        Save the Categorical to HDF5.

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

        See Also
        --------
        load

        """
        from arkouda.io import _file_type_to_int, _mode_str_to_int

        args = {
            "codes": self.codes,
            "categories": self.categories,
            "dset": dataset,
            "write_mode": _mode_str_to_int(mode),
            "filename": prefix_path,
            "objType": "categorical",
            "file_format": _file_type_to_int(file_type),
            "NA_codes": self._akNAcode,
        }
        if self.permutation is not None and self.segments is not None:
            args["permutation"] = self.permutation
            args["segments"] = self.segments

        generic_msg(
            cmd="tohdf",
            args=args,
        )

    def update_hdf(self, prefix_path, dataset="categorical_array", repack=True):
        """
        Overwrite the dataset with the name provided with this Categorical object.

        If the dataset does not exist it is added.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in files
        repack: bool
            Default: True
            HDF5 does not release memory on delete. When True, the inaccessible
            data (that was overwritten) is removed. When False, the data remains, but is
            inaccessible. Setting to false will yield better performance, but will cause
            file sizes to expand.

        Raises
        ------
        RuntimeError
            Raised if a server-side error is thrown saving the Categorical

        Notes
        -----
        - If file does not contain File_Format attribute to indicate how it was saved,
          the file name is checked for _LOCALE#### to determine if it is distributed.
        - If the dataset provided does not exist, it will be added
        - Because HDF5 deletes do not release memory, the repack option allows for
          automatic creation of a file without the inaccessible data.

        """
        from arkouda.io import (
            _file_type_to_int,
            _get_hdf_filetype,
            _mode_str_to_int,
            _repack_hdf,
        )

        # determine the format (single/distribute) that the file was saved in
        file_type = _get_hdf_filetype(prefix_path + "*")

        args = {
            "codes": self.codes,
            "categories": self.categories,
            "dset": dataset,
            "write_mode": _mode_str_to_int("append"),
            "filename": prefix_path,
            "objType": "categorical",
            "overwrite": True,
            "file_format": _file_type_to_int(file_type),
            "NA_codes": self._akNAcode,
        }
        if self.permutation is not None and self.segments is not None:
            args["permutation"] = self.permutation
            args["segments"] = self.segments

        generic_msg(
            cmd="tohdf",
            args=args,
        )

        if repack:
            _repack_hdf(prefix_path)

    def to_parquet(
        self,
        prefix_path: str,
        dataset: str = "categorical_array",
        mode: str = "truncate",
        compression: Optional[str] = None,
    ) -> str:
        """
        [Not Yet Implemented] Save the Categorical to a Parquet dataset.

        !!! This method is currently not supported and will raise a RuntimeError. !!!
        Parquet support for Categorical is under development.

        When implemented, this method will write the Categorical to a set of Parquet
        files, one file per locale on the Arkouda server. Each file will be named
        using the `prefix_path` with locale-specific suffixes.

        Parameters
        ----------
        prefix_path : str
            The directory and filename prefix shared by all output files.
        dataset : str, default="categorical_array"
            The dataset name to use to create the Parquet files.
        mode : {'truncate', 'append'}, default='truncate'
            Specifies write behavior. Use 'truncate' to overwrite existing files or
            'append' to add to them. (Appending is not yet efficient.)
        compression : str, optional
            Compression algorithm to use when writing the file.
            Supported values include: 'snappy', 'gzip', 'brotli', 'zstd', 'lz4'.
            Default is None (no compression).

        Returns
        -------
        str
            A message indicating the result of the operation.

        Raises
        ------
        RuntimeError
            Always raised. Parquet export for Categorical is not yet supported.

        Notes
        -----
        - The specified `prefix_path` must be writable and accessible to the Arkouda server.
        - The user must have write permission.
        - Output files will be named as ``<prefix_path>_LOCALE<i>`` for each locale `i`.
        - Appending mode requires that the existing files already match the server’s locale layout.
        - Appending mode is supported, but is not efficient.
        - File extensions are not used to determine file type.
        - If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.

        See Also
        --------
        to_hdf : Save the Categorical to HDF5 format (currently supported).

        """
        # due to the possibility that components will be different sizes,
        # writing to Parquet is not supported at this time
        raise RuntimeError(
            "Categorical cannot be written to Parquet at this time due to its components "
            "potentially having different sizes."
        )
        result = []
        comp_dict = {k: v for k, v in self._get_components_dict().items() if v is not None}

        if self.RequiredPieces.issubset(comp_dict.keys()):
            # Honor the first mode but switch to append for all others
            # since each following comp may wipe out the file
            first = True
            for k, v in comp_dict.items():
                result.append(
                    v.to_parquet(
                        prefix_path,
                        dataset=f"{dataset}.{k}",
                        mode=(mode if first else "append"),
                        compression=compression,
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
        Register this Categorical object and underlying components with the Arkouda server.

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

        See Also
        --------
        unregister, attach, unregister_categorical_by_name, is_registered

        Notes
        -----
        Objects registered with the server are immune to deletion until
        they are unregistered.

        """
        if self.registered_name is not None and self.is_registered():
            raise RegistrationError(f"This object is already registered as {self.registered_name}")
        generic_msg(
            cmd="register",
            args={
                "name": user_defined_name,
                "objType": self.objType,
                "codes": self.codes,
                "categories": self.categories,
                "_akNAcode": self._akNAcode,
                "segments": self.segments if self.segments is not None else "",
                "permutation": self.permutation if self.permutation is not None else "",
            },
        )
        self.registered_name = user_defined_name
        return self

    def unregister(self) -> None:
        """
        Unregister this Categorical object.

        Unregister this Categorical object in the arkouda server which was previously
        registered using register() and/or attached to using attach()

        Raises
        ------
        RegistrationError
            If the object is already unregistered or if there is a server error
            when attempting to unregister

        See Also
        --------
        register, attach, unregister_categorical_by_name, is_registered

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

    def is_registered(self) -> np.bool_:
        """
        Return True iff the object is contained in the registry or is a component of a registered object.

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
        from arkouda.numpy.util import is_registered

        if self.registered_name is None:
            result = True
            result &= is_registered(self.codes.name, as_component=True)
            result &= is_registered(self.categories.name, as_component=True)
            result &= is_registered(self._akNAcode.name, as_component=True)
            if self.permutation is not None and self.segments is not None:
                result &= is_registered(self.permutation.name, as_component=True)
                result &= is_registered(self.segments.name, as_component=True)
            return np.bool_(result)
        else:
            return np.bool_(is_registered(self.registered_name))

    def _get_components_dict(self) -> Dict:
        """
        Return a dictionary with all required or non-None components of self.

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
        Return a list of all component names.

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
        Return a JSON formatted string containing information about all components of self.

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
        """Print information about all components of self in a human-readable format."""
        [p.pretty_print_info() for p in Categorical._get_components_dict(self).values()]

    @staticmethod
    @typechecked
    def _parse_hdf_categoricals(
        d: Mapping[str, Union[pdarray, Strings]],
    ) -> Tuple[List[str], Dict[str, Categorical]]:
        """
        Parse mapping of pdarray and Stings objects from hdf5 files.

        Parse mapping of pdarray and Stings objects
        in order to reconstitute Categoricals objects from hdf5 files.

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
            if any(i.endswith(".categories") for i in v):  # we have a categorical
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

    def transfer(self, hostname: str, port: int_scalars):
        """
        Send a Categorical object to a different Arkouda server.

        Parameters
        ----------
        hostname : str
            The hostname where the Arkouda server intended to
            receive the Categorical is running.
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
        A message indicating a complete transfer

        Raises
        ------
        ValueError
            Raised if the op is not within the pdarray.BinOps set
        TypeError
            Raised if other is not a pdarray or the pdarray.dtype is not
            a supported dtype

        """
        # hostname is the hostname to send to
        args = {
            "codes": self.codes,
            "categories": self.categories,
            "objType": self.objType,
            "NA_codes": self._akNAcode,
            "hostname": hostname,
            "port": port,
        }
        return generic_msg(
            cmd="sendArray",
            args=args,
        )
