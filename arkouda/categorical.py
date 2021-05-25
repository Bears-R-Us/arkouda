from __future__ import annotations
from typing import cast, List, Optional, Sequence, Union, Dict
import numpy as np # type: ignore
import itertools
from typeguard import typechecked
from arkouda.strings import Strings
from arkouda.pdarrayclass import pdarray, RegistrationError, unregister_pdarray_by_name
from arkouda.groupbyclass import GroupBy, broadcast
from arkouda.pdarraysetops import in1d, unique, concatenate
from arkouda.pdarraycreation import zeros, zeros_like, arange
from arkouda.dtypes import resolve_scalar_dtype, str_scalars
from arkouda.dtypes import int64 as akint64
from arkouda.sorting import argsort
from arkouda.logger import getArkoudaLogger
from arkouda.infoclass import information, list_registry

__all__ = ['Categorical']

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
    RegisterablePieces = frozenset(["categories", "codes", "permutation", "segments"])
    RequiredPieces = frozenset(["categories", "codes"])
    objtype = "category"
    permutation = None
    segments = None
    
    def __init__(self, values, **kwargs) -> None:
        self.logger = getArkoudaLogger(name=__class__.__name__)  # type: ignore
        if 'codes' in kwargs and 'categories' in kwargs:
            # This initialization is called by Categorical.from_codes()
            # The values arg is ignored
            self.codes = kwargs['codes']
            self.categories = kwargs['categories']            
            if 'permutation' in kwargs:
                self.permutation = cast(pdarray, kwargs['permutation'])
            if 'segments' in kwargs:
                self.segments = cast(pdarray,kwargs['segments'])
        else:
            # Typical initialization, called with values
            if not isinstance(values, Strings):
                raise ValueError(("Categorical: inputs other than " +
                                 "Strings not yet supported"))
            g = GroupBy(values)
            self.categories = g.unique_keys
            self.codes = zeros(values.size, dtype=np.int64)
            self.codes[cast(pdarray, g.permutation)] = \
                                g.broadcast(arange(self.categories.size))
            self.permutation = cast(pdarray, g.permutation)
            self.segments = g.segments
        # Always set these values
        self.size = self.codes.size
        self.nlevels = self.categories.size
        self.ndim = self.codes.ndim
        self.shape = self.codes.shape
        self.name : Optional[str] = None

    @classmethod
    @typechecked
    def from_codes(cls, codes : pdarray, categories : Strings, 
                          permutation=None, segments=None) -> Categorical:
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
        return cls(None, codes=codes, categories=categories, 
                            permutation=permutation, segments=segments)

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
        idx = self.categories.to_ndarray()
        valcodes = self.codes.to_ndarray()
        return idx[valcodes]

    def __iter__(self):
        raise NotImplementedError('Categorical does not support iteration. To force data transfer from server, use to_ndarray')
        
    def __len__(self):
        return self.shape[0]

    def __str__(self):
        # limit scope of import to pick up changes to global variable
        from arkouda.client import pdarrayIterThresh
        if self.size <= pdarrayIterThresh:
            vals = ["'{}'".format(self[i]) for i in range(self.size)]
        else:
            vals = ["'{}'".format(self[i]) for i in range(3)]
            vals.append('... ')
            vals.extend([self[i] for i in range(self.size-3, self.size)])
        return "[{}]".format(', '.join(vals))

    def __repr__(self):
        return "array({})".format(self.__str__())

    @typechecked
    def _binop(self, other : Union[Categorical,str_scalars], op : str_scalars) -> pdarray:
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
    -   -----
        ValueError
            Raised if (1) the op is not in the self.BinOps set, or (2) if the
            sizes of this and the other instance don't match
        RuntimeError
            Raised if a server-side error is thrown while executing the
            binary operation
        """
        if op not in self.BinOps:
            raise NotImplementedError("Categorical: unsupported operator: {}".\
                                      format(op))
        if np.isscalar(other) and resolve_scalar_dtype(other) == "str":
            idxresult = self.categories._binop(other, op)
            return idxresult[self.codes]
        if self.size != cast(Categorical,other).size:
            raise ValueError("Categorical {}: size mismatch {} {}".\
                             format(op, self.size, cast(Categorical,other).size))
        if isinstance(other, Categorical):
            if (self.categories.size == other.categories.size) and (self.categories == other.categories).all():
                # Because categories are identical, codes can be compared directly
                return self.codes._binop(other.codes, op)
            else:
                # Remap both codes to the union of categories
                union = unique(concatenate((self.categories, other.categories), ordered=False))
                newinds = arange(union.size)
                # Inds of self.categories in unioned categories
                selfnewinds = newinds[in1d(union, self.categories)]
                # Need a permutation and segments to broadcast new codes
                if self.permutation is None or self.segments is None:
                    g = GroupBy(self.codes)
                    self.permutation = g.permutation
                    self.segments = g.segments
                # Form new codes by broadcasting new indices for unioned categories
                selfnewcodes = broadcast(self.segments, selfnewinds, self.size, self.permutation)
                # Repeat for other
                othernewinds = newinds[in1d(union, other.categories)]
                if other.permutation is None or other.segments is None:
                    g = GroupBy(other.codes)
                    other.permutation = g.permutation
                    other.segments = g.segments
                othernewcodes = broadcast(other.segments, othernewinds, other.size, other.permutation)
                # selfnewcodes and othernewcodes now refer to same unioned categories
                # and can be compared directly
                return selfnewcodes._binop(othernewcodes, op)
        else:
            raise NotImplementedError(("Operations between Categorical and " +
                                "non-Categorical not yet implemented. " +
                                "Consider converting operands to Categorical."))

    @typechecked
    def _r_binop(self, other : Union[Categorical,str_scalars], 
                                              op : str_scalars) -> pdarray:
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
    -   -----
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
        if np.isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            return self.categories[self.codes[key]]
        else:
            return Categorical.from_codes(self.codes[key], self.categories)

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
        newvals = zeros(self.codes.size, akint64)
        newvals[g.permutation] = g.broadcast(arange(idx.size))
        return Categorical.from_codes(newvals, idx, permutation=g.permutation, 
                                      segments=g.segments)

    @typechecked
    def contains(self, substr : str) -> pdarray:
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
    def startswith(self, substr : str) -> pdarray:
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
    def endswith(self, substr : str) -> pdarray:
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
    def in1d(self, test : Union[Strings,Categorical]) -> pdarray:
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
        >>> strings = ak.array(['String {}'.format(i) for i in range(0,5)])
        >>> cat = ak.Categorical(strings)
        >>> ak.in1d(cat,strings)
        array([True, True, True, True, True])
        >>> strings = ak.array(['String {}'.format(i) for i in range(5,9)])
        >>> catTwo = ak.Categorical(strings)
        >>> ak.in1d(cat,catTwo)
        array([False, False, False, False, False])
        """
        if isinstance(test,Categorical):
            categoriesisin = in1d(self.categories, test.categories)
        else:
            categoriesisin = in1d(self.categories, test)
        return categoriesisin[self.codes]

    def unique(self) -> Categorical:
        #__doc__ = unique.__doc__
        return Categorical.from_codes(arange(self.categories.size), 
                                      self.categories)

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

    def argsort(self):
        #__doc__ = argsort.__doc__
        idxperm = argsort(self.categories)
        inverse = zeros_like(idxperm)
        inverse[idxperm] = arange(idxperm.size)
        newvals = inverse[self.codes]
        return argsort(newvals)

    def sort(self):
        #__doc__ = sort.__doc__
        idxperm = argsort(self.categories)
        inverse = zeros_like(idxperm)
        inverse[idxperm] = arange(idxperm.size)
        newvals = inverse[self.codes]
        return Categorical.from_codes(newvals, self.categories[idxperm])
    
    @typechecked
    def concatenate(self, others : Sequence[Categorical], ordered : bool=True) -> Categorical:
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
                raise TypeError(("Categorical: can only merge/concatenate " +
                                "with other Categoricals"))
            if (self.categories.size != c.categories.size) or not \
                                    (self.categories == c.categories).all():
                samecategories = False
        if samecategories:
            newvals = cast(pdarray, concatenate([self.codes] + [o.codes for o in others], ordered=ordered))
            return Categorical.from_codes(newvals, self.categories)
        else:
            g = GroupBy(concatenate([self.categories] + \
                                       [o.categories for o in others],
                                       ordered=True))
            newidx = g.unique_keys
            wherediditgo = g.broadcast(arange(newidx.size), permute=True)
            idxsizes = np.array([self.categories.size] + \
                                [o.categories.size for o in others])
            idxoffsets = np.cumsum(idxsizes) - idxsizes
            oldvals = concatenate([c + off for c, off in \
                                   zip([self.codes] + [o.codes for o in others], idxoffsets)],
                                  ordered=ordered)
            newvals = wherediditgo[oldvals]
            return Categorical.from_codes(newvals, newidx)

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
            This is an in-place modification, the original is returned to support a fluid programming style.
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
        [p.register(f"{user_defined_name}.{n}") for n, p in Categorical._get_components_dict(self).items()]
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
            raise RegistrationError("This item does not have a name and does not appear to be registered.")
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
        parts_registered: List[np.bool_] = [p.is_registered() for p in Categorical._get_components_dict(self).values()]
        if np.any(parts_registered) and not np.all(parts_registered):  # test for error
            raise RegistrationError(f"Not all registerable components of Categorical {self.name} are registered.")

        return np.bool_(np.any(parts_registered))

    def _get_components_dict(self) -> Dict:
        """
        Internal function that returns a dictionary with all required or non-None components of self

        Required Categorical components (Codes and Categories) are always included in returned components_dict
        Optional Categorical components (Permutation and Segments) are only included if they've been set (are not None)

        Returns
        -------
        Dict
            Dictionary of all required or non-None components of self
                Keys: component names (Codes, Categories, Permutation, Segments)
                Values: components of self
        """
        return {piece_name: getattr(self, piece_name) for piece_name in Categorical.RegisterablePieces
                if piece_name in Categorical.RequiredPieces or getattr(self, piece_name) is not None}

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
        return list(itertools.chain.from_iterable(
            [p._list_component_names() for p in Categorical._get_components_dict(self).values()]))

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

        # Unregister optional pieces only if they are contained in the registry
        registry = list_registry()
        if f"{user_defined_name}.permutation" in registry:
            unregister_pdarray_by_name(f"{user_defined_name}.permutation")
        if f"{user_defined_name}.segments" in registry:
            unregister_pdarray_by_name(f"{user_defined_name}.segments")
