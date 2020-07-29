import json, struct
import numpy as np

from arkouda.client import generic_msg, verbose, maxTransferBytes, pdarrayIterThresh
from arkouda.dtypes import *
from arkouda.dtypes import structDtypeCodes, NUMBER_FORMAT_STRINGS

__all__ = ["pdarray", "info", "any", "all", "is_sorted", "sum", "prod", "min", "max",
           "argmin", "argmax", "mean", "var", "std", "mink", "maxk"]

def parse_single_value(msg):
    """
    Attempt to convert a scalar return value from the arkouda server to a numpy
    scalar in Python. The user should not call this function directly.
    """
    dtname, value = msg.split(maxsplit=1)
    mydtype = dtype(dtname)
    if mydtype == bool:
        if value == "True":
            return bool(True)
        elif value == "False":
            return bool(False)
        else:
            raise ValueError("unsupported value from server {} {}".format(mydtype.name, value))
    try:
        if mydtype == str_:
            return mydtype.type(value.strip('"'))
        return mydtype.type(value)
    except:
        raise ValueError("unsupported value from server {} {}".format(mydtype.name, value))

# class for the pdarray
class pdarray:
    """
    The basic arkouda array class. This class contains only the
    attributies of the array; the data resides on the arkouda
    server. When a server operation results in a new array, arkouda
    will create a pdarray instance that points to the array data on
    the server. As such, the user should not initialize pdarray
    instances directly.

    Attributes
    ----------
    name : str
        The server-side identifier for the array
    dtype : dtype
        The element type of the array
    size : int
        The number of elements in the array
    ndim : int
        The rank of the array (currently only rank 1 arrays supported)
    shape : tuple
        The sizes of each dimension of the array
    itemsize : int
        The size in bytes of each element
    """

    BinOps = frozenset(["+", "-", "*", "/", "//", "%", "<", ">", "<=", ">=", "!=", "==", "&", "|", "^", "<<", ">>","**"])
    OpEqOps = frozenset(["+=", "-=", "*=", "/=", "//=", "&=", "|=", "^=", "<<=", ">>=","**="])
    objtype = "pdarray"

    __array_priority__ = 1000

    def __init__(self, name, mydtype, size, ndim, shape, itemsize):
        self.name = name
        self.dtype = dtype(mydtype)
        self.size = size
        self.ndim = ndim
        self.shape = shape
        self.itemsize = itemsize

    def __del__(self):
        try:
            generic_msg("delete {}".format(self.name))
        except:
            pass

    def __bool__(self):
        if self.size != 1:
            raise ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
        return bool(self[0])

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        global pdarrayIterThresh
        return generic_msg("str {} {}".format(self.name,pdarrayIterThresh) )

    def __repr__(self):
        global pdarrayIterTresh
        return generic_msg("repr {} {}".format(self.name,pdarrayIterThresh))

    def format_other(self, other):
        """
        Attempt to cast scalar other to the element dtype of this pdarray,
        and print the resulting value to a string (e.g. for sending to a
        server command). The user should not call this function directly.
        """
        try:
            other = self.dtype.type(other)
        except:
            raise TypeError("Unable to convert {} to {}".format(other, self.dtype.name))
        if self.dtype == bool:
            return str(other)
        fmt = NUMBER_FORMAT_STRINGS[self.dtype.name]
        return fmt.format(other)

    # binary operators
    def binop(self, other, op):
        if op not in self.BinOps:
            raise ValueError("bad operator {}".format(op))
        # pdarray binop pdarray
        if isinstance(other, pdarray):
            if self.size != other.size:
                raise ValueError("size mismatch {} {}".format(self.size,other.size))
            msg = "binopvv {} {} {}".format(op, self.name, other.name)
            repMsg = generic_msg(msg)
            return create_pdarray(repMsg)
        # pdarray binop scalar
        dt = resolve_scalar_dtype(other)
        if dt not in DTypes:
            raise TypeError("Unhandled scalar type: {} ({})".format(other, type(other)))
        msg = "binopvs {} {} {} {}".format(op, self.name, dt, NUMBER_FORMAT_STRINGS[dt].format(other))
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    # reverse binary operators
    # pdarray binop pdarray: taken care of by binop function
    def r_binop(self, other, op):
        if op not in self.BinOps:
            raise ValueError("bad operator {}".format(op))
        # pdarray binop scalar
        dt = resolve_scalar_dtype(other)
        if dt not in DTypes:
            raise TypeError("Unhandled scalar type: {} ({})".format(other, type(other)))
        msg = "binopsv {} {} {} {}".format(op, dt, NUMBER_FORMAT_STRINGS[dt].format(other), self.name)
        repMsg = generic_msg(msg)
        return create_pdarray(repMsg)

    # overload + for pdarray, other can be {pdarray, int, float}
    def __add__(self, other):
        return self.binop(other, "+")

    def __radd__(self, other):
        return self.r_binop(other, "+")

    # overload - for pdarray, other can be {pdarray, int, float}
    def __sub__(self, other):
        return self.binop(other, "-")

    def __rsub__(self, other):
        return self.r_binop(other, "-")

    # overload * for pdarray, other can be {pdarray, int, float}
    def __mul__(self, other):
        return self.binop(other, "*")

    def __rmul__(self, other):
        return self.r_binop(other, "*")

    # overload / for pdarray, other can be {pdarray, int, float}
    def __truediv__(self, other):
        return self.binop(other, "/")

    def __rtruediv__(self, other):
        return self.r_binop(other, "/")

    # overload // for pdarray, other can be {pdarray, int, float}
    def __floordiv__(self, other):
        return self.binop(other, "//")

    def __rfloordiv__(self, other):
        return self.r_binop(other, "//")

    def __mod__(self, other):
        return self.binop(other, "%")

    def __rmod__(self, other):
        return self.r_binop(other, "%")

    # overload << for pdarray, other can be {pdarray, int}
    def __lshift__(self, other):
        return self.binop(other, "<<")

    def __rlshift__(self, other):
        return self.r_binop(other, "<<")

    # overload >> for pdarray, other can be {pdarray, int}
    def __rshift__(self, other):
        return self.binop(other, ">>")

    def __rrshift__(self, other):
        return self.r_binop(other, ">>")

    # overload & for pdarray, other can be {pdarray, int}
    def __and__(self, other):
        return self.binop(other, "&")

    def __rand__(self, other):
        return self.r_binop(other, "&")

    # overload | for pdarray, other can be {pdarray, int}
    def __or__(self, other):
        return self.binop(other, "|")

    def __ror__(self, other):
        return self.r_binop(other, "|")

    # overload | for pdarray, other can be {pdarray, int}
    def __xor__(self, other):
        return self.binop(other, "^")

    def __rxor__(self, other):
        return self.r_binop(other, "^")

    def __pow__(self,other):
        return self.binop(other,"**")

    def __rpow__(self,other):
        return self.r_binop(other,"**")

    # overloaded comparison operators
    def __lt__(self, other):
        return self.binop(other, "<")

    def __gt__(self, other):
        return self.binop(other, ">")

    def __le__(self, other):
        return self.binop(other, "<=")

    def __ge__(self, other):
        return self.binop(other, ">=")

    def __eq__(self, other):
        return self.binop(other, "==")

    def __ne__(self, other):
        return self.binop(other, "!=")

    # overload unary- for pdarray implemented as pdarray*(-1)
    def __neg__(self):
        return self.binop(-1, "*")

    # overload unary~ for pdarray implemented as pdarray^(~0)
    def __invert__(self):
        if self.dtype == int64:
            return self.binop(~0, "^")
        if self.dtype == bool:
            return self.binop(True, "^")
        raise TypeError("Unhandled dtype: {} ({})".format(self, self.dtype))

    # op= operators
    def opeq(self, other, op):
        if op not in self.OpEqOps:
            raise ValueError("bad operator {}".format(op))
        # pdarray op= pdarray
        if isinstance(other, pdarray):
            if self.size != other.size:
                raise ValueError("size mismatch {} {}".format(self.size,other.size))
            generic_msg("opeqvv {} {} {}".format(op, self.name, other.name))
            return self
        # pdarray binop scalar
        # opeq requires scalar to be cast as pdarray dtype
        try:
            other = self.dtype.type(other)
        except: # Can't cast other as dtype of pdarray
            raise TypeError("Unhandled scalar type: {} ({})".format(other, type(other)))

        msg = "opeqvs {} {} {} {}".format(op, self.name, self.dtype.name, self.format_other(other))
        generic_msg(msg)
        return self

    # overload += pdarray, other can be {pdarray, int, float}
    def __iadd__(self, other):
        return self.opeq(other, "+=")

    # overload -= pdarray, other can be {pdarray, int, float}
    def __isub__(self, other):
        return self.opeq(other, "-=")

    # overload *= pdarray, other can be {pdarray, int, float}
    def __imul__(self, other):
        return self.opeq(other, "*=")

    # overload /= pdarray, other can be {pdarray, int, float}
    def __itruediv__(self, other):
        return self.opeq(other, "/=")

    # overload //= pdarray, other can be {pdarray, int, float}
    def __ifloordiv__(self, other):
        return self.opeq(other, "//=")

    # overload <<= pdarray, other can be {pdarray, int, float}
    def __ilshift__(self, other):
        return self.opeq(other, "<<=")

    # overload >>= pdarray, other can be {pdarray, int, float}
    def __irshift__(self, other):
        return self.opeq(other, ">>=")

    # overload &= pdarray, other can be {pdarray, int, float}
    def __iand__(self, other):
        return self.opeq(other, "&=")

    # overload |= pdarray, other can be {pdarray, int, float}
    def __ior__(self, other):
        return self.opeq(other, "|=")

    # overload ^= pdarray, other can be {pdarray, int, float}
    def __ixor__(self, other):
        return self.opeq(other, "^=")
    def __ipow__(self, other):
        return self.opeq(other,"**=")

    # overload a[] to treat like list
    def __getitem__(self, key):
        if np.isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if (key >= 0 and key < self.size):
                repMsg = generic_msg("[int] {} {}".format(self.name, key))
                fields = repMsg.split()
                # value = fields[2]
                return parse_single_value(' '.join(fields[1:]))
            else:
                raise IndexError("[int] {} is out of bounds with size {}".format(orig_key,self.size))
        if isinstance(key, slice):
            (start,stop,stride) = key.indices(self.size)
            if verbose: print(start,stop,stride)
            repMsg = generic_msg("[slice] {} {} {} {}".format(self.name, start, stop, stride))
            return create_pdarray(repMsg);
        if isinstance(key, pdarray):
            kind, itemsize = translate_np_dtype(key.dtype)
            if kind not in ("bool", "int"):
                raise TypeError("unsupported pdarray index type {}".format(key.dtype))
            if kind == "bool" and self.size != key.size:
                raise ValueError("size mismatch {} {}".format(self.size,key.size))
            repMsg = generic_msg("[pdarray] {} {}".format(self.name, key.name))
            return create_pdarray(repMsg);
        else:
            raise TypeError("Unhandled key type: {} ({})".format(key, type(key)))

    def __setitem__(self, key, value):
        if np.isscalar(key) and resolve_scalar_dtype(key) == 'int64':
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if (key >= 0 and key < self.size):
                generic_msg("[int]=val {} {} {} {}".format(self.name, key, self.dtype.name, self.format_other(value)))
            else:
                raise IndexError("index {} is out of bounds with size {}".format(orig_key,self.size))
        elif isinstance(key, pdarray):
            if isinstance(value, pdarray):
                generic_msg("[pdarray]=pdarray {} {} {}".format(self.name,key.name,value.name))
            else:
                generic_msg("[pdarray]=val {} {} {} {}".format(self.name, key.name, self.dtype.name, self.format_other(value)))
        elif isinstance(key, slice):
            (start,stop,stride) = key.indices(self.size)
            if verbose: print(start,stop,stride)
            if isinstance(value, pdarray):
                generic_msg("[slice]=pdarray {} {} {} {} {}".format(self.name,start,stop,stride,value.name))
            else:
                generic_msg("[slice]=val {} {} {} {} {} {}".format(self.name, start, stop, stride, self.dtype.name, self.format_other(value)))
        else:
            raise TypeError("Unhandled key type: {} ({})".format(key, type(key)))

    def fill(self, value):
        """
        Fill the array (in place) with a constant value.
        """
        generic_msg("set {} {} {}".format(self.name, self.dtype.name, self.format_other(value)))

    def any(self):
        """
        Return True iff any element of the array evaluates to True.
        """
        return any(self)

    def all(self):
        """
        Return True iff all elements of the array evaluate to True.
        """
        return all(self)

    def is_sorted(self):
        """
        Return True iff the array is monotonically non-decreasing.
        """
        return is_sorted(self)

    def sum(self):
        """
        Return the sum of all elements in the array.
        """
        return sum(self)

    def prod(self):
        """
        Return the product of all elements in the array. Return value is
        always a float.
        """
        return prod(self)

    def min(self):
        """
        Return the minimum value of the array.
        """
        return min(self)

    def max(self):
        """
        Return the maximum value of the array.
        """
        return max(self)

    def argmin(self):
        """
        Return the index of the first minimum value of the array.
        """
        return argmin(self)

    def argmax(self):
        """
        Return the index of the first maximum value of the array.
        """
        return argmax(self)

    def mean(self):
        """
        Return the mean of the array.
        """
        return mean(self)

    def var(self, ddof=0):
        """
        Compute the variance. See ``arkouda.var`` for details.
        """
        return var(self, ddof=ddof)

    def std(self, ddof=0):
        """
        Compute the standard deviation. See ``arkouda.std`` for details.
        """
        return std(self, ddof=ddof)

    def mink(self, k):
        """
        Compute the minimum "k" values.
        """
        return mink(self,k)


    def maxk(self, k):
        """
        Compute the maximum "k" values.
        """
        return maxk(self,k)

    
    def to_ndarray(self):
        """
        Convert the array to a np.ndarray, transferring array data from the
        arkouda server to Python. If the array exceeds a builtin size limit,
        a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            A numpy ndarray with the same attributes and data as the pdarray

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
        >>> a = ak.arange(0, 5, 1)
        >>> a.to_ndarray()
        array([0, 1, 2, 3, 4])

        >>> type(a.to_ndarray())
        numpy.ndarray
        """
        # Total number of bytes in the array data
        arraybytes = self.size * self.dtype.itemsize
        # Guard against overflowing client memory
        if arraybytes > maxTransferBytes:
            raise RuntimeError("Array exceeds allowed size for transfer. Increase ak.maxTransferBytes to allow")
        # The reply from the server will be a bytes object
        rep_msg = generic_msg("tondarray {}".format(self.name), recv_bytes=True)
        # Make sure the received data has the expected length
        if len(rep_msg) != self.size*self.dtype.itemsize:
            raise RuntimeError("Expected {} bytes but received {}".format(self.size*self.dtype.itemsize, len(rep_msg)))
        # Use struct to interpret bytes as a big-endian numeric array
        fmt = '>{:n}{}'.format(self.size, structDtypeCodes[self.dtype.name])
        # Return a numpy ndarray
        return np.array(struct.unpack(fmt, rep_msg))

    def to_cuda(self):
        """
        Convert the array to a Numba DeviceND array, transferring array data from the
        arkouda server to Python via ndarray. If the array exceeds a builtin size limit,
        a RuntimeError is raised.

        Returns
        -------
        numba.DeviceNDArray
            A Numba ndarray with the same attributes and data as the pdarray; on GPU

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
        >>> a = ak.arange(0, 5, 1)
        >>> a.to_cuda()
        array([0, 1, 2, 3, 4])

        >>> type(a.to_cuda())
        numpy.devicendarray
        """
        try:
            from numba import cuda
            if not(cuda.is_available()):
                raise ImportError('CUDA is not available. Check for the CUDA toolkit and ensure a GPU is installed.')
                return
        except:
            raise ModuleNotFoundError('Numba is not enabled or installed and is required for GPU support.')
            return

        # Total number of bytes in the array data
        arraybytes = self.size * self.dtype.itemsize
        # Guard against overflowing client memory
        if arraybytes > maxTransferBytes:
            raise RuntimeError("Array exceeds allowed size for transfer. Increase ak.maxTransferBytes to allow")
        # The reply from the server will be a bytes object
        rep_msg = generic_msg("tondarray {}".format(self.name), recv_bytes=True)
        # Make sure the received data has the expected length
        if len(rep_msg) != self.size*self.dtype.itemsize:
            raise RuntimeError("Expected {} bytes but received {}".format(self.size*self.dtype.itemsize, len(rep_msg)))
        # Use struct to interpret bytes as a big-endian numeric array
        fmt = '>{:n}{}'.format(self.size, structDtypeCodes[self.dtype.name])
        # Return a numba devicendarray
        return cuda.to_device(struct.unpack(fmt, rep_msg))

    def save(self, prefix_path, dataset='array', mode='truncate'):
        """
        Save the pdarray to HDF5. The result is a collection of HDF5 files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            Name of the dataset to create in HDF5 files (must not already exist)
        mode : {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', attempt to create new dataset in existing files.

        See Also
        --------
        save_all, load, read_hdf, read_all

        Notes
        -----
        The prefix_path must be visible to the arkouda server and the user must have
        write permission.

        Output files have names of the form ``<prefix_path>_LOCALE<i>.hdf``, where ``<i>``
        ranges from 0 to ``numLocales``. If any of the output files already exist and
        the mode is 'truncate', they will be overwritten. If the mode is 'append'
        and the number of output files is less than the number of locales or a
        dataset with the same name already exists, a ``RuntimeError`` will result.

        Examples
        --------
        >>> a = ak.arange(0, 100, 1)
        >>> a.save('arkouda_range', dataset='array')

        Array is saved in numLocales files with names like ``tmp/arkouda_range_LOCALE0.hdf``

        The array can be read back in as follows

        >>> b = ak.load('arkouda_range', dataset='array')
        >>> (a == b).all()
        True
        """
        if mode.lower() in 'append':
            m = 1
        elif mode.lower() in 'truncate':
            m = 0
        else:
            raise ValueError("Allowed modes are 'truncate' and 'append'")
        rep_msg = generic_msg("tohdf {} {} {} {}".format(self.name, dataset, m, json.dumps([prefix_path])))



# creates pdarray object
#   only after:
#       all values have been checked by python module and...
#       server has created pdarray already befroe this is called
def create_pdarray(repMsg):
    """
    Return a pdarray instance pointing to an array created by the arkouda server.
    The user should not call this function directly.
    """
    fields = repMsg.split()
    name = fields[1]
    mydtype = fields[2]
    size = int(fields[3])
    ndim = int(fields[4])
    shape = [int(el) for el in fields[5][1:-1].split(',')]
    itemsize = int(fields[6])
    if verbose: print("{} {} {} {} {} {}".format(name,mydtype,size,ndim,shape,itemsize))
    return pdarray(name,mydtype,size,ndim,shape,itemsize)

def info(pda):
    if isinstance(pda, pdarray):
        return generic_msg("info {}".format(pda.name))
    elif isinstance(pda, str):
        return generic_msg("info {}".format(pda))
    else:
        raise TypeError("info: must be pdarray or string {}".format(pda))

def any(pda):
    """
    Return True iff any element of the array evaluates to True.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("any", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def all(pda):
    """
    Return True iff all elements of the array evaluate to True.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("all", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def is_sorted(pda):
    """
    Return True iff the array is monotonically non-decreasing.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("is_sorted", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def sum(pda):
    """
    Return the sum of all elements in the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("sum", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def prod(pda):
    """
    Return the product of all elements in the array. Return value is
    always a float.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("prod", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def min(pda):
    """
    Return the minimum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("min", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def max(pda):
    """
    Return the maximum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("max", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def argmin(pda):
    """
    Return the index of the first minimum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmin", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def argmax(pda):
    """
    Return the index of the first maximum value of the array.
    """
    if isinstance(pda, pdarray):
        repMsg = generic_msg("reduction {} {}".format("argmax", pda.name))
        return parse_single_value(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def mean(pda):
    """
    Return the mean of the array.
    """
    return pda.sum() / pda.size

def var(pda, ddof=0):
    """
    Return the variance of values in the array.

    Parameters
    ----------
    pda : pdarray
        Values for which to find the variance
    ddof : int
        "Delta Degrees of Freedom" used in calculating mean

    Returns
    -------
    float
        The scalar variance of the array

    See Also
    --------
    mean, std

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    i.e.,  ``var = mean((x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used
    instead.  In standard statistical practice, ``ddof=1`` provides an
    unbiased estimator of the variance of a hypothetical infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.
    """
    if not isinstance(pda, pdarray):
        raise TypeError("must be pdarray {}".format(pda))
    if ddof >= pda.size:
        raise ValueError("var: ddof must be less than number of values")
    m = mean(pda)
    return ((pda - m)**2).sum() / (pda.size - ddof)

def std(pda, ddof=0):
    """
    Return the standard deviation of values in the array. The standard
    deviation is implemented as the square root of the variance.

    Parameters
    ----------
    pda : pdarray
        values for which to find the variance
    ddof : int
        "Delta Degrees of Freedom" used in calculating mean

    Returns
    -------
    float
        The scalar standard deviation of the array

    See Also
    --------
    mean, var

    Notes
    -----
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
    """
    return np.sqrt(var(pda, ddof=ddof))

def mink(pda, k):
    """
    Find the `k` minimum values of an array.

    Returns the smallest `k` values of an array, sorted

    Parameters
    ----------
    pda : pdarray
        Input array.
    k : integer
        The desired count of minimum values to be returned by the output.

    Returns
    -------
    pdarray, int
        The minimum `k` values from pda

    Notes
    -----
    Currently only works on integers, could be exended to also work for floats.

    Examples
    --------
    >>> A = ak.array([10,5,1,3,7,2,9,0])
    >>> ak.mink(A, 3)
    array([0, 1, 2])
    """
    if isinstance(pda, pdarray):
        if k == 0:
            return []
        if pda.dtype != int or pda.size == 0:
            raise TypeError("must be a non-empty pdarray {} of type int".format(pda))
        repMsg = generic_msg("mink {} {}".format(pda.name, k))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))

def maxk(pda, k):
    """
    Find the `k` maximum values of an array.

    Returns the largest `k` values of an array, sorted

    Parameters
    ----------
    pda : pdarray
        Input array.
    k : integer
        The desired count of maximum values to be returned by the output.

    Returns
    -------
    pdarray, int
        The maximum `k` values from pda

    Notes
    -----
    Currently only works on integers, could be exended to also work for floats.

    Examples
    --------
    >>> A = ak.array([10,5,1,3,7,2,9,0])
    >>> ak.maxk(A, 3)
    array([7, 9, 10])
    """
    if isinstance(pda, pdarray):
        if k == 0:
            return []
        if pda.dtype != int or pda.size == 0:
            raise TypeError("must be a non-empty pdarray {} of type int".format(pda))
        repMsg = generic_msg("maxk {} {}".format(pda.name, k))
        return create_pdarray(repMsg)
    else:
        raise TypeError("must be pdarray {}".format(pda))
