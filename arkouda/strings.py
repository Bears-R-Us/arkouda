from arkouda.client import generic_msg, verbose, pdarrayIterThresh
from arkouda.pdarrayclass import pdarray, create_pdarray, parse_single_value
from arkouda.dtypes import *
from arkouda.dtypes import NUMBER_FORMAT_STRINGS
import numpy as np
import json

global verbose
global pdarrayIterThresh

__all__ = ['Strings']

class Strings:
    """
    Represents an array of strings whose data resides on the
    arkouda server. The user should not call this class directly;
    rather its instances are created by other arkouda functions.

    Attributes
    ----------
    offsets : pdarray
        The starting indices for each string
    bytes : pdarray
        The raw bytes of all strings, joined by nulls
    size : int
        The number of strings in the array
    nbytes : int
        The total number of bytes in all strings
    ndim : int
        The rank of the array (currently only rank 1 arrays supported)
    shape : tuple
        The sizes of each dimension of the array
    """

    BinOps = frozenset(["==", "!="])
    objtype = "str"

    def __init__(self, offset_attrib, bytes_attrib):
        if isinstance(offset_attrib, pdarray):
            self.offsets = offset_attrib
        else:
            self.offsets = create_pdarray(offset_attrib)
        if isinstance(bytes_attrib, pdarray):
            self.bytes = bytes_attrib
        else:
            self.bytes = create_pdarray(bytes_attrib)
        self.size = self.offsets.size
        self.nbytes = self.bytes.size
        self.ndim = self.offsets.ndim
        self.shape = self.offsets.shape

    def __iter__(self):
        # to_ndarray will error if array is too large to bring back
        a = self.to_ndarray()
        for s in a:
            yield s

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        if self.size <= pdarrayIterThresh:
            vals = ["'{}'".format(self[i]) for i in range(self.size)]
        else:
            vals = ["'{}'".format(self[i]) for i in range(3)]
            vals.append('... ')
            vals.extend([self[i] for i in range(self.size-3, self.size)])
        return "[{}]".format(', '.join(vals))

    def __repr__(self):
        return "array({})".format(self.__str__())

    def binop(self, other, op):
        if op not in self.BinOps:
            raise ValueError("Strings: unsupported operator: {}".format(op))
        if isinstance(other, Strings):
            if self.size != other.size:
                raise ValueError("Strings: size mismatch {} {}".format(self.size, other.size))
            msg = "segmentedBinopvv {} {} {} {} {} {} {}".format(op,
                                                                 self.objtype,
                                                                 self.offsets.name,
                                                                 self.bytes.name,
                                                                 other.objtype,
                                                                 other.offsets.name,
                                                                 other.bytes.name)
        elif resolve_scalar_dtype(other) == 'str':
            msg = "segmentedBinopvs {} {} {} {} {} {}".format(op,
                                                              self.objtype,
                                                              self.offsets.name,
                                                              self.bytes.name,
                                                              self.objtype,
                                                              json.dumps([other]))
        else:
            raise ValueError("Strings: {} not supported between Strings and {}".format(op, type(other)))
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    def __eq__(self, other):
        return self.binop(other, "==")

    def __ne__(self, other):
        return self.binop(other, "!=")

    def __getitem__(self, key):
        if np.isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if (key >= 0 and key < self.size):
                msg = "segmentedIndex {} {} {} {} {}".format('intIndex',
                                                             self.objtype,
                                                             self.offsets.name,
                                                             self.bytes.name,
                                                             key)
                repMsg = generic_msg(msg)
                _, value = repMsg.split(maxsplit=1)
                return parse_single_value(value)
            else:
                raise IndexError("[int] {} is out of bounds with size {}".format(orig_key,self.size))
        elif isinstance(key, slice):
            (start,stop,stride) = key.indices(self.size)
            if verbose: print(start,stop,stride)
            msg = "segmentedIndex {} {} {} {} {} {} {}".format('sliceIndex',
                                                               self.objtype,
                                                               self.offsets.name,
                                                               self.bytes.name,
                                                               start,
                                                               stop,
                                                               stride)
            repMsg = generic_msg(msg)
            offsets, values = repMsg.split('+')
            return Strings(offsets, values);
        elif isinstance(key, pdarray):
            kind, itemsize = translate_np_dtype(key.dtype)
            if kind not in ("bool", "int"):
                raise TypeError("unsupported pdarray index type {}".format(key.dtype))
            if kind == "bool" and self.size != key.size:
                raise ValueError("size mismatch {} {}".format(self.size,key.size))
            msg = "segmentedIndex {} {} {} {} {}".format('pdarrayIndex',
                                                         self.objtype,
                                                         self.offsets.name,
                                                         self.bytes.name,
                                                         key.name)
            repMsg = generic_msg(msg)
            offsets, values = repMsg.split('+')
            return Strings(offsets, values)
        else:
            raise TypeError("unsupported pdarray index type {}".format(type(key)))

    def get_lengths(self):
        """
        Return the length of each string in the array.

        Returns
        -------
        pdarray, int
            The length of each string
        """
        msg = "segmentLengths {} {} {}".format(self.objtype, self.offsets.name, self.bytes.name)
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    def contains(self, substr):
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

        See Also
        --------
        Strings.startswith, Strings.endswith
        """
        if isinstance(substr, bytes):
            substr = substr.decode()
        if not isinstance(substr, str):
            raise TypeError("Substring must be a string, not {}".format(type(substr)))
        msg = "segmentedEfunc {} {} {} {} {} {}".format("contains",
                                                        self.objtype,
                                                        self.offsets.name,
                                                        self.bytes.name,
                                                        "str",
                                                        json.dumps([substr]))
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    def startswith(self, substr):
        """
        Check whether each element starts with the given substring.

        Parameters
        ----------
        substr : str
            The prefix to search for

        Returns
        -------
        pdarray, bool
            True for elements that start with substr, False otherwise

        See Also
        --------
        Strings.contains, Strings.endswith
        """
        if isinstance(substr, bytes):
            substr = substr.decode()
        if not isinstance(substr, str):
            raise TypeError("Substring must be a string, not {}".format(type(substr)))
        msg = "segmentedEfunc {} {} {} {} {} {}".format("startswith",
                                                        self.objtype,
                                                        self.offsets.name,
                                                        self.bytes.name,
                                                        "str",
                                                        json.dumps([substr]))
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    def endswith(self, substr):
        """
        Check whether each element ends with the given substring.

        Parameters
        ----------
        substr : str
            The suffix to search for

        Returns
        -------
        pdarray, bool
            True for elements that end with substr, False otherwise

        See Also
        --------
        Strings.contains, Strings.startswith
        """
        if isinstance(substr, bytes):
            substr = substr.decode()
        if not isinstance(substr, str):
            raise TypeError("Substring must be a string, not {}".format(type(substr)))
        msg = "segmentedEfunc {} {} {} {} {} {}".format("endswith",
                                                        self.objtype,
                                                        self.offsets.name,
                                                        self.bytes.name,
                                                        "str",
                                                        json.dumps([substr]))
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    def peel(self, delimiter, times=1, includeDelimiter=False, keepPartial=False, fromRight=False):
        """
        Peel off one or more delimited fields from each string (similar 
        to string.partition), returning two new arrays of strings.

        Parameters
        ----------
        delimiter : str
            The separator where the split will occur
        times : int
            The number of times the delimiter is sought, i.e. skip over 
            the first (times-1) delimiters
        includeDelimiter : bool
            If true, append the delimiter to the end of the first return 
            array. By default, it is prepended to the beginning of the 
            second return array.
        keepPartial : bool
            If true, a string that does not contain <times> instances of 
            the delimiter will be returned in the first array. By default, 
            such strings are returned in the second array.
        fromRight : bool
            If true, peel from the right instead of the left (see also rpeel)

        Returns
        -------
        left : Strings
            The field(s) peeled from the end of each string (unless 
            fromRight is true)
        right : Strings
            The remainder of each string after peeling (unless fromRight 
            is true)
        
        See Also
        --------
        rpeel, stick, lstick

        Examples
        --------
        >>> s = ak.array(['a.b', 'c.d', 'e.f.g'])
        >>> s.peel('.')
        (array(['a', 'c', 'e']), array(['b', 'd', 'f.g']))
        >>> s.peel('.', includeDelimiter=True)
        (array(['a.', 'c.', 'e.']), array(['b', 'd', 'f.g']))
        >>> s.peel('.', times=2)
        (array(['', '', 'e.f']), array(['a.b', 'c.d', 'g']))
        >>> s.peel('.', times=2, keepPartial=True)
        (array(['a.b', 'c.d', 'e.f']), array(['', '', 'g']))
        """
        if isinstance(delimiter, bytes):
            delimiter = delimiter.decode()
        if not isinstance(delimiter, str):
            raise TypeError("Delimiter must be a string, not {}".format(type(delimiter)))
        if not np.isscalar(times) or resolve_scalar_dtype(times) != 'int64':
            raise TypeError("Times must be integer, not {}".format(type(times)))
        if times < 1:
            raise ValueError("Times must be > 0")
        msg = "segmentedEfunc {} {} {} {} {} {} {} {} {} {}".format("peel",
                                                                    self.objtype,
                                                                    self.offsets.name,
                                                                    self.bytes.name,
                                                                    "str",
                                                                    NUMBER_FORMAT_STRINGS['int64'].format(times),
                                                                    NUMBER_FORMAT_STRINGS['bool'].format(includeDelimiter),
                                                                    NUMBER_FORMAT_STRINGS['bool'].format(keepPartial),
                                                                    NUMBER_FORMAT_STRINGS['bool'].format(not fromRight),
                                                                    json.dumps([delimiter]))
        repMsg = generic_msg(msg)
        arrays = repMsg.split('+', maxsplit=3)
        leftStr = Strings(arrays[0], arrays[1])
        rightStr = Strings(arrays[2], arrays[3])
        return leftStr, rightStr

    def rpeel(self, delimiter, times=1, includeDelimiter=False, keepPartial=False):
        """
        Peel off one or more delimited fields from the end of each string 
        (similar to string.rpartition), returning two new arrays of strings.

        Parameters
        ----------
        delimiter : str
            The separator where the split will occur
        times : int
            The number of times the delimiter is sought, i.e. skip over 
            the last (times-1) delimiters
        includeDelimiter : bool
            If true, prepend the delimiter to the start of the first return 
            array. By default, it is appended to the end of the 
            second return array.
        keepPartial : bool
            If true, a string that does not contain <times> instances of 
            the delimiter will be returned in the second array. By default, 
            such strings are returned in the first array.

        Returns
        -------
        left : Strings
            The remainder of the string after peeling
        right : Strings
            The field(s) that were peeled from the right of each string
        
        See Also
        --------
        peel, stick, lstick

        Examples
        --------
        >>> s = ak.array(['a.b', 'c.d', 'e.f.g'])
        >>> s.rpeel('.')
        (array(['a', 'c', 'e.f']), array(['b', 'd', 'g']))
        # Compared against peel
        >>> s.peel('.')
        (array(['a', 'c', 'e']), array(['b', 'd', 'f.g']))
        """
        return self.peel(delimiter, times=times, includeDelimiter=includeDelimiter, keepPartial=keepPartial, fromRight=True)

    def stick(self, other, delimiter="", toLeft=False):
        """
        Join the strings from another array onto one end of the strings 
        of this array, optionally inserting a delimiter.

        Parameters
        ----------
        other : Strings
            The strings to join onto self's strings
        delimiter : str
            String inserted between self and other
        toLeft : bool
            If true, join other strings to the left of self. By default,
            other is joined to the right of self.

        Returns
        -------
        Strings
            The array of joined strings

        See Also
        --------
        lstick, peel, rpeel

        Examples
        --------
        >>> s = ak.array(['a', 'c', 'e'])
        >>> t = ak.array(['b', 'd', 'f'])
        >>> s.stick(t, delimiter='.')
        array(['a.b', 'c.d', 'e.f'])
        """
        if not isinstance(other, Strings):
            raise TypeError("stick: not supported between Strings and {}".format(type(other)))
        if isinstance(delimiter, bytes):
            delimiter = delimiter.decode()
        if not isinstance(delimiter, str):
            raise TypeError("Delimiter must be a string, not {}".format(type(delimiter)))
        msg = "segmentedBinopvv {} {} {} {} {} {} {} {} {}".format("stick",
                                                             self.objtype,
                                                             self.offsets.name,
                                                             self.bytes.name,
                                                             other.objtype,
                                                             other.offsets.name,
                                                             other.bytes.name,
                                                             NUMBER_FORMAT_STRINGS['bool'].format(toLeft),
                                                             json.dumps([delimiter]))
        repMsg = generic_msg(msg)
        return Strings(*repMsg.split('+'))

    def __add__(self, other):
        return self.stick(other)
    
    def lstick(self, other, delimiter=""):
        """
        Join the strings from another array onto the left of the strings 
        of this array, optionally inserting a delimiter.

        Parameters
        ----------
        other : Strings
            The strings to join onto self's strings
        delimiter : str
            String inserted between self and other

        Returns
        -------
        Strings
            The array of joined strings, as other + self

        See Also
        --------
        stick, peel, rpeel

        Examples
        --------
        >>> s = ak.array(['a', 'c', 'e'])
        >>> t = ak.array(['b', 'd', 'f'])
        >>> s.lstick(t, delimiter='.')
        array(['b.a', 'd.c', 'f.e'])
        """
        return self.stick(other, delimiter=delimiter, toLeft=True)

    def __radd__(self, other):
        return self.lstick(other)
    
    def hash(self):
        """
        Compute a 128-bit hash of each string.

        Returns
        -------
        (pdarray, pdarray)
            A pair of int64 pdarrays. The ith hash value is the concatenation
            of the ith values from each array.

        Notes
        -----
        The implementation uses SipHash128, a fast and balanced hash function (used
        by Python for dictionaries and sets). For realistic numbers of strings (up
        to about 10**15), the probability of a collision between two 128-bit hash
        values is negligible.
        """
        msg = "segmentedHash {} {} {}".format(self.objtype, self.offsets.name, self.bytes.name)
        repMsg = generic_msg(msg)
        h1, h2 = repMsg.split('+')
        return create_pdarray(h1), create_pdarray(h2)

    def group(self):
        """
        Return the permutation that groups the array, placing equivalent
        strings together. All instances of the same string are guaranteed to lie
        in one contiguous block of the permuted array, but the blocks are not
        necessarily ordered.

        Returns
        -------
        pdarray
            The permutation that groups the array by value

        See Also
        --------
        GroupBy, unique

        Notes
        -----
        If the arkouda server is compiled with "-sSegmentedArray.useHash=true",
        then arkouda uses 128-bit hash values to group strings, rather than sorting
        the strings directly. This method is fast, but the resulting permutation
        merely groups equivalent strings and does not sort them. If the "useHash"
        parameter is false, then a full sort is performed.
        """
        msg = "segmentedGroup {} {} {}".format(self.objtype, self.offsets.name, self.bytes.name)
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    def to_ndarray(self):
        """
        Convert the array to a np.ndarray, transferring array data from the
        arkouda server to Python. If the array exceeds a builtin size limit,
        a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            A numpy ndarray with the same strings as this array

        Notes
        -----
        The number of bytes in the array cannot exceed ``arkouda.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting ak.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        array

        Examples
        --------
        >>> a = ak.array(["hello", "my", "world"])
        >>> a.to_ndarray()
        array(['hello', 'my', 'world'], dtype='<U5')

        >>> type(a.to_ndarray())
        numpy.ndarray
        """
        # Get offsets and append total bytes for length calculation
        npoffsets = np.hstack((self.offsets.to_ndarray(), np.array([self.nbytes])))
        # Get contents of strings (will error if too large)
        npvalues = self.bytes.to_ndarray()
        # Compute lengths, discounting null terminators
        lengths = np.diff(npoffsets) - 1
        # Numpy dtype is based on max string length
        dt = '<U{}'.format(lengths.max())
        res = np.empty(self.size, dtype=dt)
        # Form a string from each segment and store in numpy array
        for i, (o, l) in enumerate(zip(npoffsets, lengths)):
            res[i] = np.str_(''.join(chr(b) for b in npvalues[o:o+l]))
        return res
