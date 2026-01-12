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
from pandas.api.extensions import ExtensionDtype, register_extension_dtype

from arkouda.numpy.dtypes import bigint, float64, int64, str_, uint8, uint64
from arkouda.numpy.dtypes import bool as ak_bool
from arkouda.numpy.dtypes import dtype as ak_dtype


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
        Construct an Arkouda ``ExtensionDtype`` from a string specifier.

        This method resolves **Arkouda-prefixed** dtype strings only.
        Supported forms include:

        - ``"ak.int64"``, ``"ak_int64"``, ``"akint64"``
        - ``"ak.uint64"``, ``"ak_uint64"``, ``"akuint64"``
        - ``"ak.float64"``, ``"ak_float64"``, ``"akfloat64"``
        - ``"ak.bool"``, ``"ak_bool"``, ``"akbool"``
        - ``"ak.string"``, ``"ak_string"``, ``"akstring"``
        - ``"ak.category"``, ``"ak_category"``, ``"akcategory"``
        - and the long form ``"arkouda.int64"`` and similar.

        Plain dtype names such as ``"int64"``, ``"float64"``, ``"bool"``, or
        ``"string"`` are **not** Arkouda dtypes. In normal pandas/NumPy
        dtype resolution, those plain names should be handled upstream and
        never reach this method. However, if they are passed directly to
        :meth:`construct_from_string`, they will raise :class:`TypeError`.
        This is intentional: Arkouda dtype strings must include an ``ak``
        prefix (for example, ``"ak_int64"``) so they can be distinguished
        from standard pandas/NumPy dtypes.

        Parameters
        ----------
        string : str
            String dtype specifier to interpret as an Arkouda dtype. Must
            include an Arkouda ``ak``-style prefix (e.g. ``"ak_int64"``,
            ``"ak.float64"``, ``"akstring"``); plain dtype names without an
            Arkouda prefix are not accepted here.

        Returns
        -------
        _ArkoudaBaseDtype
            The corresponding Arkouda ExtensionDtype instance.

        Raises
        ------
        TypeError
            If ``string`` does not represent a valid Arkouda-prefixed dtype
            (including the case where it is a plain pandas/NumPy dtype like
            ``"int64"`` or ``"float64"``).
        """
        normalized = string.strip().lower()

        # Recognize Arkouda-style prefixes
        prefixes = ("ak.", "ak_", "arkouda.")
        matched_prefix = None
        for p in prefixes:
            if normalized.startswith(p):
                matched_prefix = p
                break

        # Extra: support compact forms like "akint64", "akstring", "akcategory"
        if matched_prefix is None and normalized.startswith("ak") and len(normalized) > 2:
            matched_prefix = "ak"

        if matched_prefix is None:
            # IMPORTANT: let pandas/NumPy handle bare "int64", "float64", etc.
            raise TypeError(
                f"Invalid Arkouda dtype string {string!r}. "
                "Arkouda dtype names must begin with the 'ak' prefix "
                "(e.g., 'ak_int64', 'ak_float64', 'ak_bool'). "
                "If you intended a standard NumPy/pandas dtype, use it without the 'ak_' prefix."
            )

        base = normalized[len(matched_prefix) :]

        # Strip any accidental leading punctuation after the prefix (e.g. "ak..int64")
        base = base.lstrip("._")

        # Small alias tweaks: "string" -> "str_"
        if base == "category":
            return ArkoudaCategoricalDtype()
        if base == "string":
            base = "str_"

        # Now interpret using Arkouda's dtype system
        dtype = ak_dtype(base)

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


# ---- Generic dtype -------------------------------------------------------------


@register_extension_dtype
class ArkoudaDtype(ExtensionDtype):
    """
    Generic Arkouda-backed dtype for pandas construction.

    Using dtype="ak" triggers ArkoudaExtensionArray._from_sequence, which
    dispatches to ArkoudaArray / ArkoudaStringArray / ArkoudaCategoricalArray.
    """

    name = "ak"
    type = object  # pandas requires something
    kind = "O"

    @classmethod
    def construct_from_string(cls, string):
        if string == "ak":
            return cls()
        raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    def construct_array_type(self):
        # Important: return the base class that implements factory dispatch.
        from arkouda.pandas.extension._arkouda_extension_array import ArkoudaExtensionArray

        return ArkoudaExtensionArray


# ---- Concrete dtypes --------------------------------------------------------


@register_extension_dtype
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


@register_extension_dtype
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


@register_extension_dtype
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


@register_extension_dtype
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


@register_extension_dtype
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


@register_extension_dtype
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


@register_extension_dtype
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


@register_extension_dtype
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
