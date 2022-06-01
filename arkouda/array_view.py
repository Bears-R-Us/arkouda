import json
from enum import Enum

import numpy as np  # type: ignore

from arkouda.client import generic_msg
from arkouda.dtypes import resolve_scalar_dtype, translate_np_dtype
from arkouda.numeric import cast as akcast
from arkouda.numeric import cumprod, where
from arkouda.pdarrayclass import create_pdarray, parse_single_value, pdarray
from arkouda.pdarraycreation import arange, array, ones, zeros
from arkouda.pdarraysetops import concatenate

OrderType = Enum("OrderType", ["ROW_MAJOR", "COLUMN_MAJOR"])


class ArrayView:
    """
    A multi-dimensional view of a pdarray. Arkouda ``ArraryView`` behaves similarly to numpy's ndarray.
    The base pdarray is stored in 1-dimension but can be indexed and treated logically
    as if it were multi-dimensional

    Attributes
    ----------
    base: pdarray
        The base pdarray that is being viewed as a multi-dimensional object
    dtype: dtype
        The element type of the base pdarray (equivalent to base.dtype)
    size: int_scalars
        The number of elements in the base pdarray (equivalent to base.size)
    shape: pdarray[int]
        A pdarray specifying the sizes of each dimension of the array
    ndim: int_scalars
         Number of dimensions (equivalent to shape.size)
    itemsize: int_scalars
        The size in bytes of each element (equivalent to base.itemsize)
    order: str {'C'/'row_major' | 'F'/'column_major'}
        Index order to read and write the elements.
        By default or if 'C'/'row_major', read and write data in row_major order
        If 'F'/'column_major', read and write data in column_major order
    """

    def __init__(self, base: pdarray, shape, order="row_major"):
        self.objtype = type(self).__name__
        self.shape = array(shape)
        if not isinstance(self.shape, pdarray):
            raise TypeError(f"ArrayView Shape cannot be type {type(self.shape)}. Expecting pdarray.")
        if base.size != self.shape.prod():
            raise ValueError(f"cannot reshape array of size {base.size} into shape {self.shape}")
        self.base = base
        self.size = base.size
        self.dtype = base.dtype
        self.ndim = self.shape.size
        self.itemsize = self.base.itemsize
        if order.upper() in {"C", "ROW_MAJOR"}:
            self.order = OrderType.ROW_MAJOR
        elif order.upper() in {"F", "COLUMN_MAJOR"}:
            self.order = OrderType.COLUMN_MAJOR
        else:
            raise ValueError(f"cannot traverse with order={order}")
        # cache _reverse_shape which is reversed if we're row_major
        self._reverse_shape = self.shape if self.order is OrderType.COLUMN_MAJOR else self.shape[::-1]
        if self.shape.min() == 0:
            # avoid divide by 0 if any of the dimensions are 0
            self._dim_prod = zeros(self.shape.size, self.dtype)
        else:
            # cache dim_prod to avoid recalculation, reverse if row_major
            self._dim_prod = (
                cumprod(self.shape) // self.shape
                if self.order is OrderType.COLUMN_MAJOR
                else cumprod(self._reverse_shape) // self._reverse_shape
            )

    def __len__(self):
        return self.size

    def __repr__(self):
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            return self.to_ndarray().__repr__()
        else:
            edge_items = np.get_printoptions()["edgeitems"]
            vals = [f"'{self.base[i]}'" for i in range(edge_items)]
            vals.append("... ")
            vals.extend([f"'{self.base[i]}'" for i in range(self.size - edge_items, self.size)])
        return f"array([{', '.join(vals)}]), shape {self.shape}"

    def __str__(self):
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            return self.to_ndarray().__str__()
        else:
            edge_items = np.get_printoptions()["edgeitems"]
            vals = [f"'{self.base[i]}'" for i in range(edge_items)]
            vals.append("... ")
            vals.extend([f"'{self.base[i]}'" for i in range(self.size - edge_items, self.size)])
        return f"[{', '.join(vals)}], shape {self.shape}"

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            key = [key]
        elif isinstance(key, tuple):
            key = list(key)
        if len(key) > self.ndim:
            raise IndexError(
                f"too many indices for array: array is {self.ndim}-dimensional, "
                f"but {len(key)} were indexed"
            )
        if len(key) < self.ndim:
            # append self.ndim-len(key) many ':'s to fill in the missing dimensions
            for i in range(self.ndim - len(key)):
                key.append(slice(None, None, None))
        try:
            # attempt to convert to a pdarray (allows for view[0,2,1] instead of view[ak.array([0,2,1])]
            # but pass on RuntimeError to allow things like
            # view[0,:,[True,False,True]] to be correctly handled
            key = array(key)
        except (RuntimeError, TypeError, ValueError, DeprecationWarning):
            pass
        if isinstance(key, pdarray):
            kind, _ = translate_np_dtype(key.dtype)
            if kind not in ("int", "uint", "bool"):
                raise TypeError(f"unsupported pdarray index type {key.dtype}")
            if kind == "bool":
                if key.all():
                    # every dimension is True, so return this arrayview with shape = [1, self.shape]
                    return self.base.reshape(
                        concatenate([ones(1, dtype=self.dtype), self.shape]), order=self.order.name
                    )
                else:
                    # at least one dimension is False,
                    # so return empty arrayview with shape = [0, self.shape]
                    return array([], dtype=self.dtype).reshape(
                        concatenate([zeros(1, dtype=self.dtype), self.shape]), order=self.order.name
                    )
            # Interpret negative key as offset from end of array
            key = where(key < 0, akcast(key + self.shape, kind), key)
            # Capture the indices which are still out of bounds
            out_of_bounds = (key < 0) | (self.shape <= key)
            if out_of_bounds.any():
                out = arange(key.size)[out_of_bounds][0]
                raise IndexError(
                    f"index {key[out]} is out of bounds for axis {out} with size {self.shape[out]}"
                )
            coords = key if self.order is OrderType.COLUMN_MAJOR else key[::-1]
            repMsg = generic_msg(
                cmd="arrayViewIntIndex", args=f"{self.base.name} {self._dim_prod.name} {coords.name}"
            )
            fields = repMsg.split()
            return parse_single_value(" ".join(fields[1:]))
        elif isinstance(key, list):
            indices = []
            reshape_dim_list = []
            index_dim_list = []
            key = key if self.order is OrderType.COLUMN_MAJOR else key[::-1]
            for i in range(len(key)):
                x = key[i]
                if np.isscalar(x) and (resolve_scalar_dtype(x) == "int64" or "uint64"):
                    orig_key = x
                    if x < 0:
                        # Interpret negative key as offset from end of array
                        x += self._reverse_shape[i]
                    if 0 <= x < self._reverse_shape[i]:
                        indices.append("int")
                        # have to cast to int because JSON doesn't recognize numpy dtypes
                        indices.append(json.dumps(int(x)))
                        index_dim_list.append(1)
                    else:
                        raise IndexError(
                            f"index {orig_key} is out of bounds for axis {i} "
                            f"with size {self._reverse_shape[i]}"
                        )
                elif isinstance(x, slice):
                    (start, stop, stride) = x.indices(self._reverse_shape[i])
                    indices.append("slice")
                    indices.append(json.dumps((start, stop, stride)))
                    slice_size = len(range(*(start, stop, stride)))
                    index_dim_list.append(slice_size)
                    reshape_dim_list.append(slice_size)
                elif isinstance(x, pdarray) or isinstance(x, list):
                    raise TypeError(f"Advanced indexing is not yet supported {x} ({type(x)})")
                    # x = array(x)
                    # kind, _ = translate_np_dtype(x.dtype)
                    # if kind not in ("bool", "int"):
                    #     raise TypeError("unsupported pdarray index type {}".format(x.dtype))
                    # if kind == "bool" and dim != x.size:
                    #     raise ValueError("size mismatch {} {}".format(dim, x.size))
                    # indices.append('pdarray')
                    # indices.append(x.name)
                    # index_dim_list.append(x.size)
                    # reshape_dim_list.append(x.size)
                    # arrays.append(x)
                else:
                    raise TypeError(f"Unhandled key type: {x} ({type(x)})")
            index_dim = array(index_dim_list)
            repMsg = generic_msg(
                cmd="arrayViewMixedIndex",
                args="{} {} {} {} {}".format(
                    self.base.name, index_dim.name, self.ndim, self._dim_prod.name, json.dumps(indices)
                ),
            )
            reshape_dim = (
                reshape_dim_list if self.order is OrderType.COLUMN_MAJOR else reshape_dim_list[::-1]
            )
            return create_pdarray(repMsg).reshape(reshape_dim, order=self.order.name)
        else:
            raise TypeError(f"Unhandled key type: {key} ({type(key)})")

    def __setitem__(self, key, value):
        if isinstance(key, int) or isinstance(key, slice):
            key = [key]
        elif isinstance(key, tuple):
            key = list(key)
        if len(key) > self.ndim:
            raise IndexError(
                f"too many indices for array: array is {self.ndim}-dimensional, "
                f"but {len(key)} were indexed"
            )
        if len(key) < self.ndim:
            # append self.ndim-len(key) many ':'s to fill in the missing dimensions
            for i in range(self.ndim - len(key)):
                key.append(slice(None, None, None))
        try:
            # attempt to convert to a pdarray (allows for view[0,2,1] instead of view[ak.array([0,2,1])]
            # but pass on RuntimeError to allow things like
            # view[0,:,[True,False,True]] to be correctly handled
            key = array(key)
        except (RuntimeError, TypeError, ValueError, DeprecationWarning):
            pass
        if isinstance(key, pdarray):
            kind, _ = translate_np_dtype(key.dtype)
            if kind not in ("int", "uint", "bool"):
                raise TypeError(f"unsupported pdarray index type {key.dtype}")
            if kind == "bool":
                if key.all():
                    # every dimension is True, so fill arrayview with value
                    # if any dimension is False, we don't update anything
                    self.base.fill(value)
            else:
                # Interpret negative key as offset from end of array
                key = where(key < 0, akcast(key + self.shape, kind), key)
                # Capture the indices which are still out of bounds
                out_of_bounds = (key < 0) | (self.shape <= key)
                if out_of_bounds.any():
                    out = arange(key.size)[out_of_bounds][0]
                    raise IndexError(
                        f"index {key[out]} is out of bounds for axis {out} with size {self.shape[out]}"
                    )
                coords = key if self.order is OrderType.COLUMN_MAJOR else key[::-1]
                generic_msg(
                    cmd="arrayViewIntIndexAssign",
                    args="{} {} {} {} {}".format(
                        self.base.name,
                        self.dtype.name,
                        self._dim_prod.name,
                        coords.name,
                        self.base.format_other(value),
                    ),
                )
        elif isinstance(key, list):
            raise NotImplementedError("Setting via slicing and advanced indexing is not yet supported")
        else:
            raise TypeError(f"Unhandled key type: {key} ({type(key)})")

    def to_ndarray(self) -> np.ndarray:
        """
        Convert the ArrayView to a np.ndarray, transferring array data from the
        Arkouda server to client-side Python. Note: if the ArrayView size exceeds
        client.maxTransferBytes, a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            A numpy ndarray with the same attributes and data as the ArrayView

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown, if the ArrayView size
            exceeds the built-in client.maxTransferBytes size limit, or if the bytes
            received does not match expected number of bytes
        Notes
        -----
        The number of bytes in the array cannot exceed ``client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting client.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        array

        Examples
        --------
        >>> a = ak.arange(6).reshape(2,3)
        >>> a.to_ndarray()
        array([[0, 1, 2],
               [3, 4, 5]])
        >>> type(a.to_ndarray())
        numpy.ndarray
        """
        if self.order is OrderType.ROW_MAJOR:
            return self.base.to_ndarray().reshape(self.shape.to_ndarray())
        else:
            return self.base.to_ndarray().reshape(self.shape.to_ndarray(), order="F")
