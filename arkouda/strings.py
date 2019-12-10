from arkouda.client import generic_msg, verbose, pdarrayIterThresh
from arkouda.pdarrayclass import pdarray, create_pdarray, parse_single_value
from arkouda.dtypes import *
import numpy as np

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
                                                              other)
        else:
            raise ValueError("Strings: {} not supported between Strings and {}".format(op, type(other)))
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)
        
    def __eq__(self, other):
        return self.binop(other, "==")

    def __neq__(self, other):
        return self.binop(other, "!=")

    def __getitem__(self, key):
        if np.isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            if (key >= 0 and key < self.size):
                msg = "segmentedIndex {} {} {} {} {}".format('intIndex',
                                                             self.objtype,
                                                             self.offsets.name,
                                                             self.bytes.name,
                                                             key)
                repMsg = generic_msg(msg)
                fields = repMsg.split()
                # value = fields[2]
                return parse_single_value(' '.join(fields[1:]))
            else:
                raise IndexError("[int] {} is out of bounds with size {}".format(key,self.size))
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
                                                        substr)
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
                                                        substr)
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
                                                        substr)
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    def group(self):
        """
        Return the permutation that groups the array, placing equivalent 
        strings together. This permutation does NOT sort the strings. All 
        instances of the same string are guaranteed to lie in one contiguous 
        block of the permuted array, but the blocks are not necessarily ordered.

        Returns
        -------
        pdarray
            The permutation that groups the array by value

        See Also
        --------
        GroupBy, unique
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
