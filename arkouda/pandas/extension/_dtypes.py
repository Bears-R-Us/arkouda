"""
Arkouda–pandas extension dtypes.

This module defines pandas ``ExtensionDtype`` subclasses that wrap
Arkouda server-backed array types (``pdarray``, ``Strings``,
``Categorical``) so they can be used transparently inside pandas
objects such as ``Series``, ``Index``, and ``DataFrame``.

Supported dtypes
----------------
- :class:`ArkoudaInt64Dtype` – backed by Arkouda ``pdarray<int64>``.
- :class:`ArkoudaUint64Dtype` – backed by Arkouda ``pdarray<uint64>``.
- :class:`ArkoudaUint8Dtype` – backed by Arkouda ``pdarray<uint8>``.
- :class:`ArkoudaFloat64Dtype` – backed by Arkouda ``pdarray<float64>``.
- :class:`ArkoudaBoolDtype` – backed by Arkouda ``pdarray<bool>``.
- :class:`ArkoudaBigintDtype` – backed by Arkouda ``pdarray<bigint>``.
- :class:`ArkoudaStringDtype` – backed by Arkouda ``Strings``.
- :class:`ArkoudaCategoricalDtype` – backed by Arkouda ``Categorical``.

Base class
----------
All concrete dtypes inherit from :class:`_ArkoudaBaseDtype`, which
centralizes shared behavior and convenience methods such as
:meth:`_ArkoudaBaseDtype.construct_from_string`.

Design notes
------------
- Each dtype provides a ``construct_array_type`` hook, which tells
  pandas which ExtensionArray class to use (e.g. ``ArkoudaArray``,
  ``ArkoudaStringArray``).
- ``numpy_dtype`` ensures that pandas and NumPy interop fall back
  gracefully to the closest native type.
- ``na_value`` defines the sentinel used for missing data; for numeric
  types this is usually ``-1`` or ``np.nan``, for strings ``""``, and
  for categoricals the code ``-1``.

This module is part of Arkouda’s effort to implement the full pandas
ExtensionArray API, enabling zero-copy interoperability between pandas
and Arkouda’s distributed arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import dtype as np_dtype
from pandas.api.extensions import ExtensionDtype

from arkouda.numpy.dtypes import bigint
from arkouda.numpy.dtypes import bool as ak_bool
from arkouda.numpy.dtypes import dtype as ak_dtype
from arkouda.numpy.dtypes import float64, int64, str_, uint8, uint64

# ---- Base dtype -------------------------------------------------------------


class _ArkoudaBaseDtype(ExtensionDtype):
    """
    Shared behavior for Arkouda dtypes.

    Subclasses must define:
      - name (str)
      - type (python scalar type, or 'object' if no single scalar)
      - kind (str one-letter numpy kind, often 'O' for object-backed EAs)
      - numpy_dtype (np.dtype)
      - construct_array_type()
      - na_value
    """

    _numpy_dtype: np_dtype[Any]
    # Pandas feature flags (class properties in pandas 2.x)
    _is_boolean = False
    _is_numeric = False
    _is_string = False

    @classmethod
    def construct_from_string(cls, string: str) -> "_ArkoudaBaseDtype":
        """
        Construct an Arkouda ExtensionDtype from a string specifier.

        This enables pandas-style dtype construction such as:

        - ``pd.Series(..., dtype="arkouda.int64")``
        - ``pd.Series(..., dtype="int64")``
        - ``pd.Series(..., dtype="string")``
        - ``pd.Series(..., dtype="category")``

        Parameters
        ----------
        string : str
            String representation of the dtype. Can be either a plain
            numpy-style dtype name (e.g., ``"int64"``, ``"str_"``),
            an Arkouda dtype alias (e.g., ``"arkouda.int64"``),
            or special cases like ``"category"``.

        Returns
        -------
        _ArkoudaBaseDtype
            The matching Arkouda-backed dtype object, such as
            :class:`ArkoudaInt64Dtype`, :class:`ArkoudaStringDtype`,
            or :class:`ArkoudaCategoricalDtype`.

        Raises
        ------
        TypeError
            If the input string does not correspond to a known Arkouda dtype.

        Examples
        --------
        >>> from arkouda.pandas.extension._dtypes import _ArkoudaBaseDtype
        >>> _ArkoudaBaseDtype.construct_from_string("int64")
        ArkoudaInt64Dtype('int64')

        >>> _ArkoudaBaseDtype.construct_from_string("category")
        ArkoudaCategoricalDtype('category')

        >>> _ArkoudaBaseDtype.construct_from_string("str_")
        ArkoudaStringDtype('string')
        """
        if string == "category":
            return ArkoudaCategoricalDtype()

        dtype = ak_dtype(string)
        registry = {
            ak_dtype(int64): ArkoudaInt64Dtype(),
            ak_dtype(ak_bool): ArkoudaBoolDtype(),
            ak_dtype(str_): ArkoudaStringDtype(),
            ak_dtype(uint64): ArkoudaUint64Dtype(),
            ak_dtype(uint8): ArkoudaUint8Dtype(),
            ak_dtype(bigint): ArkoudaBigintDtype(),
            ak_dtype(float64): ArkoudaFloat64Dtype(),
        }
        try:
            return registry[dtype]
        except KeyError:
            raise TypeError(f"Cannot construct an Arkouda dtype from string {string!r}")

    @property
    def numpy_dtype(self) -> np.dtype:
        """
        The closest matching NumPy dtype for this Arkouda-backed dtype.

        This property provides pandas and downstream libraries with a
        standard NumPy dtype representation for compatibility. Each
        concrete Arkouda dtype subclass (e.g., ``ArkoudaInt64Dtype``,
        ``ArkoudaStringDtype``) sets its own ``_numpy_dtype`` to the
        appropriate NumPy type.

        Returns
        -------
        numpy.dtype
            The NumPy dtype corresponding to this Arkouda dtype.

        Notes
        -----
        - Pandas 2.x calls this property in certain operations that
          require fallback to NumPy semantics.
        - For example, :class:`ArkoudaInt64Dtype` may return
          ``np.dtype("int64")`` while :class:`ArkoudaStringDtype`
          may return ``np.dtype("object")``.

        Examples
        --------
        >>> ArkoudaInt64Dtype().numpy_dtype
        dtype('int64')

        >>> ArkoudaStringDtype().numpy_dtype
        dtype('<U')
        """
        return self._numpy_dtype  # each subclass sets this

    def __repr__(self) -> str:
        """
        Return a concise, unambiguous string representation of the dtype.

        The representation includes the class name and the dtype's
        registered ``name``. This is primarily for developer-facing
        output in interactive sessions, debugging, and logs. It does
        not attempt to be a full constructor expression, but should
        make it easy to identify the dtype type and variant.

        Returns
        -------
        str
            A string of the form ``ClassName('dtype_name')``.

        Examples
        --------
        >>> ArkoudaInt64Dtype().__repr__()
        "ArkoudaInt64Dtype('int64')"

        >>> ArkoudaStringDtype().__repr__()
        "ArkoudaStringDtype('string')"
        """
        return f"{self.__class__.__name__}({self.name!r})"


# ---- Concrete dtypes --------------------------------------------------------


#   TODO: register the dtypes
#   Registering them could override default pandas behavior,
#   so we should wait until the extension arrays are production ready and
#   fully tested.
# @register_extension_dtype
class ArkoudaInt64Dtype(_ArkoudaBaseDtype):
    """
    Extension dtype for Arkouda-backed 64-bit integers.

    This dtype allows seamless use of Arkouda's distributed ``int64``
    arrays inside pandas objects (``Series``, ``Index``, ``DataFrame``).
    It is backed by :class:`arkouda.pdarray` with ``dtype='int64'``
    and integrates with pandas via the
    :class:`~arkouda.pandas.extension._arkouda_array.ArkoudaArray`
    extension array.

    Methods
    -------
    construct_array_type()
        Return the associated extension array class
        (:class:`ArkoudaArray`).
    """

    name = "int64"
    kind = "i"  # numpy integer kind
    type = np.int64  # pandas uses this for scalar conversions
    _numpy_dtype = np.dtype("int64")
    na_value = -1  # choose your sentinel
    _is_numeric = True

    @classmethod
    def construct_array_type(cls):
        """
        Return the associated pandas ExtensionArray type.

        This is part of the pandas ExtensionDtype interface and is used
        internally by pandas when constructing arrays of this dtype.
        It ensures that operations like ``Series(..., dtype=ArkoudaInt64Dtype())``
        produce the correct Arkouda-backed extension array.

        Returns
        -------
        type
            The :class:`ArkoudaArray` class that implements the storage
            and behavior for this dtype.

        Notes
        -----
        - This hook tells pandas which ExtensionArray to instantiate
          whenever this dtype is requested.
        - All Arkouda dtypes defined in this module will return
          :class:`ArkoudaArray` (or a subclass thereof).

        Examples
        --------
        >>> from arkouda.pandas.extension import ArkoudaInt64Dtype
        >>> ArkoudaInt64Dtype.construct_array_type()
        <class 'arkouda.pandas.extension._arkouda_array.ArkoudaArray'>
        """
        from ._arkouda_array import ArkoudaArray

        return ArkoudaArray


# @register_extension_dtype
class ArkoudaUint64Dtype(_ArkoudaBaseDtype):
    """
    Arkouda-backed unsigned 64-bit integer dtype.

    This dtype integrates Arkouda’s ``uint64`` arrays with pandas,
    allowing users to create :class:`pandas.Series` or
    :class:`pandas.DataFrame` objects that store their data on
    the Arkouda server while still conforming to the pandas
    ExtensionArray API.

    Methods
    -------
    construct_array_type()
        Return the :class:`ArkoudaArray` class used as the storage
        container for this dtype.

    Examples
    --------
    >>> import arkouda as ak
    >>> import pandas as pd
    >>> from arkouda.pandas.extension import ArkoudaUint64Dtype, ArkoudaArray

    >>> arr = ArkoudaArray(ak.array([1, 2, 3], dtype="uint64"))
    >>> s = pd.Series(arr, dtype=ArkoudaUint64Dtype())
    >>> s
    0    1
    1    2
    2    3
    dtype: uint64
    """

    name = "uint64"
    kind = "u"  # numpy integer kind
    type = np.uint64  # pandas uses this for scalar conversions
    _numpy_dtype = np.dtype("uint64")
    na_value = -1  # choose your sentinel
    _is_numeric = True

    @classmethod
    def construct_array_type(cls):
        """
        Return the ExtensionArray class associated with this dtype.

        This is required by the pandas ExtensionDtype API. It tells pandas
        which :class:`~pandas.api.extensions.ExtensionArray` subclass should
        be used to hold data of this dtype inside a :class:`pandas.Series`
        or :class:`pandas.DataFrame`.

        Returns
        -------
        type
            The :class:`ArkoudaArray` class, which implements the storage
            and operations for Arkouda-backed arrays.
        """
        from ._arkouda_array import ArkoudaArray

        return ArkoudaArray


# @register_extension_dtype
class ArkoudaUint8Dtype(_ArkoudaBaseDtype):
    """
    Arkouda-backed unsigned 8-bit integer dtype.

    This dtype integrates Arkouda's ``uint8`` arrays with the pandas
    ExtensionArray API, allowing pandas ``Series`` and ``DataFrame``
    objects to store and operate on Arkouda-backed unsigned 8-bit
    integers. The underlying storage is an Arkouda ``pdarray<uint8>``,
    exposed through the :class:`ArkoudaArray` extension array.

    Methods
    -------
    construct_array_type()
        Returns the :class:`ArkoudaArray` type that provides the storage
        and behavior for this dtype.
    """

    name = "uint8"
    kind = "u"  # numpy integer kind
    type = np.uint8  # pandas uses this for scalar conversions
    _numpy_dtype = np.dtype("uint8")
    na_value = -1  # choose your sentinel
    _is_numeric = True

    @classmethod
    def construct_array_type(cls):
        """
        Return the ExtensionArray subclass that handles storage for this dtype.

        This method is required by the pandas ExtensionDtype interface.
        It tells pandas which ExtensionArray class to use when creating
        arrays of this dtype (for example, when calling
        ``Series(..., dtype="arkouda.uint8")``).

        Returns
        -------
        type
            The :class:`ArkoudaArray` class associated with this dtype.
        """
        from ._arkouda_array import ArkoudaArray

        return ArkoudaArray


# @register_extension_dtype
class ArkoudaFloat64Dtype(_ArkoudaBaseDtype):
    """
    Arkouda-backed 64-bit floating-point dtype.

    This dtype integrates Arkouda's server-backed `pdarray<float64>` with
    the pandas ExtensionArray interface via :class:`ArkoudaArray`. It allows
    pandas objects (Series, DataFrame) to store and manipulate large
    distributed float64 arrays without materializing them on the client.

    Methods
    -------
    construct_array_type()
        Returns the :class:`ArkoudaArray` class used for storage.
    """

    name = "float64"
    kind = "f"  # numpy integer kind
    type = float64  # pandas uses this for scalar conversions
    _numpy_dtype = np.dtype("float64")
    na_value = np.nan  # choose your sentinel
    _is_numeric = True

    @classmethod
    def construct_array_type(cls):
        """
        Return the ExtensionArray subclass that handles storage for this dtype.

        Returns
        -------
        type
            The :class:`ArkoudaArray` class associated with this dtype.
        """
        from ._arkouda_array import ArkoudaArray

        return ArkoudaArray


# @register_extension_dtype
class ArkoudaBoolDtype(_ArkoudaBaseDtype):
    """
    Arkouda-backed boolean dtype.

    This dtype integrates Arkouda's server-backed `pdarray<bool>` with
    the pandas ExtensionArray interface via :class:`ArkoudaArray`. It allows
    pandas objects (Series, DataFrame) to store and manipulate distributed
    boolean arrays without materializing them on the client.

    Methods
    -------
    construct_array_type()
        Returns the :class:`ArkoudaArray` class used for storage.
    """

    name = "bool_"
    kind = "b"
    type = ak_bool
    _numpy_dtype = np.dtype("bool")
    # For boolean, pandas prefers NA rather than False as sentinel.
    na_value = False
    _is_boolean = True

    @classmethod
    def construct_array_type(cls):
        """
        Return the ExtensionArray subclass that handles storage for this dtype.

        Returns
        -------
        type
            The :class:`ArkoudaArray` class associated with this dtype.
        """
        from ._arkouda_array import ArkoudaArray

        return ArkoudaArray


# @register_extension_dtype
class ArkoudaBigintDtype(_ArkoudaBaseDtype):
    """
    Arkouda-backed arbitrary-precision integer dtype.

    This dtype integrates Arkouda's server-backed ``pdarray<bigint>`` with
    the pandas ExtensionArray interface via :class:`ArkoudaArray`. It enables
    pandas objects (Series, DataFrame) to hold and operate on very large
    integers that exceed 64-bit precision, while keeping the data distributed
    on the Arkouda server.

    Methods
    -------
    construct_array_type()
        Returns the :class:`ArkoudaArray` class used for storage.
    """

    name = "bigint"
    kind = "O"
    type = object
    _numpy_dtype = np.dtype("O")
    na_value = -1  # category code sentinel (align with your EA take/fill semantics)

    @classmethod
    def construct_array_type(cls):
        """
        Return the ExtensionArray subclass that handles storage for this dtype.

        Returns
        -------
        type
            The :class:`ArkoudaArray` class associated with this dtype.
        """
        from ._arkouda_array import ArkoudaArray

        return ArkoudaArray


# @register_extension_dtype
class ArkoudaStringDtype(_ArkoudaBaseDtype):
    """
    Arkouda-backed string dtype.

    This dtype integrates Arkouda's distributed ``Strings`` type with the
    pandas ExtensionArray interface via :class:`ArkoudaStringArray`. It
    enables pandas objects (Series, DataFrame) to hold large, server-backed
    string columns without converting to NumPy or Python objects.

    Methods
    -------
    construct_array_type()
        Returns the :class:`ArkoudaStringArray` used as the storage class.
    """

    name = "string"
    kind = "O"
    type = object
    _numpy_dtype = np.dtype("str_")
    na_value = ""
    _is_string = True

    @classmethod
    def construct_array_type(cls):
        """
        Return the ExtensionArray subclass that handles storage for this dtype.

        Returns
        -------
        type
            The :class:`ArkoudaStringArray` class associated with this dtype.
        """
        from ._arkouda_string_array import ArkoudaStringArray

        return ArkoudaStringArray


# @register_extension_dtype
class ArkoudaCategoricalDtype(_ArkoudaBaseDtype):
    """
    Arkouda-backed categorical dtype.

    This dtype integrates Arkouda's distributed ``Categorical`` type with
    the pandas ExtensionArray interface via :class:`ArkoudaCategoricalArray`.
    It enables pandas objects (Series, DataFrame) to hold categorical data
    stored and processed on the Arkouda server, while exposing familiar
    pandas APIs.

    Methods
    -------
    construct_array_type()
        Returns the :class:`ArkoudaCategoricalArray` used as the storage class.
    """

    name = "category"
    kind = "O"
    type = object
    _numpy_dtype = np.dtype("O")
    na_value = -1  # category code sentinel (align with your EA take/fill semantics)

    @classmethod
    def construct_array_type(cls):
        """
        Return the ExtensionArray subclass that handles storage for this dtype.

        Returns
        -------
        type
            The :class:`ArkoudaCategoricalArray` class associated with this dtype.
        """
        from ._arkouda_categorical_array import ArkoudaCategoricalArray

        return ArkoudaCategoricalArray
