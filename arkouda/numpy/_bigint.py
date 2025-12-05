"""
Arkouda BigInt Dtype and Scalar Implementation.

This module defines Arkouda's variable-width integer dtype, exposed as
:class:`bigint`, along with its scalar representation :class:`bigint_`.

Overview
--------
Arkouda represents arbitrary-precision integers using a dedicated dtype
sentinel, ``ak.bigint()``, which behaves similarly to a NumPy dtype object.
Users may construct bigint scalars via ``ak.bigint(value)`` or directly by
instantiating :class:`bigint_`.

The core components provided here are:

* ``bigint`` – a dtype sentinel used to mark Arkouda arrays as variable-width
  integer arrays. Instances of this class behave like dtype objects:
  they compare equal to strings like ``"bigint"``, provide dtype metadata,
  and are singletons via an internal cache.

* ``bigint_`` – the scalar type corresponding to ``bigint`` arrays. It
  inherits directly from :class:`int`, ensuring full Python arbitrary-precision
  semantics while offering a dtype attribute compatible with Arkouda.

Key Behaviors
-------------
* ``ak.bigint()`` (no arguments) returns the dtype sentinel, analogous to
  ``np.int64``.
* ``ak.bigint(x)`` returns a :class:`bigint_` scalar whose value is ``x``.
* ``bigint_`` scalars round-trip cleanly through Arkouda arrays and preserve
  arbitrary precision.
* Strings and bytes passed to ``bigint_`` are interpreted via Python’s
  ``int(x, 0)`` to support hexadecimal, binary, and other literal formats.

Examples
--------
>>> import arkouda as ak
>>> ak.bigint()
dtype(bigint)

>>> ak.bigint(123)
ak.bigint_(123)

>>> x = ak.bigint_("0xff")
>>> x
ak.bigint_(255)
>>> x.dtype
dtype(bigint)

Notes
-----
This module contains no server-side logic; it provides only the Python-side
dtype and scalar scaffolding used by Arkouda’s bigint array implementation.
"""

from __future__ import annotations

import builtins

from typing import Any, ClassVar, Sequence


__all__ = [
    "bigint",
    "bigint_",
]


def _datatype_check(the_dtype: Any, allowed_list: Sequence, name: str):
    """
    Validate that a provided dtype is among an allowed set.

    This helper function raises a ``TypeError`` if ``the_dtype`` is not one of
    the permitted dtypes listed in ``allowed_list``.  It is used internally to
    enforce type constraints for Arkouda scalar and dtype constructors.

    Parameters
    ----------
    the_dtype : Any
        The dtype or scalar type to validate.
    allowed_list : Sequence
        A sequence of allowed dtype identifiers (types, strings, or dtype
        objects) against which ``the_dtype`` will be checked.
    name : str
        The name of the class or function performing the check, included in the
        error message for clarity.

    Raises
    ------
    TypeError
        If ``the_dtype`` is not contained in ``allowed_list``.

    Examples
    --------
    >>> _datatype_check("int64", ["int64", "uint64"], "MyType")  # OK

    >>> _datatype_check("float64", ["int64", "uint64"], "MyType")
    Traceback (most recent call last):
        ...
    TypeError: MyType only implements types ['int64', 'uint64']
    """
    if the_dtype not in allowed_list:
        raise TypeError(f"{name} only implements types {allowed_list}")


class _BigIntMeta(type):
    """
    Metaclass implementing NumPy-like construction semantics for ``ak.bigint``.

    This metaclass customizes ``__call__`` so that ``bigint`` behaves like a
    dtype sentinel when called with no arguments, but produces a ``bigint_``
    scalar when called with a value. This mirrors NumPy’s pattern where
    ``np.int64()`` returns the dtype object while ``np.int64(1)`` returns a
    scalar instance.

    Behavior
    --------
    * ``ak.bigint()``
      Returns the dtype sentinel (a singleton instance of :class:`bigint`).

    * ``ak.bigint(value)``
      Returns a :class:`bigint_` scalar initialized with ``value``. The scalar
      inherits from :class:`int` and preserves arbitrary-precision semantics.

    This dual behavior allows ``bigint`` to act both as a dtype object in array
    construction and as a scalar constructor for convenience.

    Notes
    -----
    ``_BigIntMeta`` only overrides ``__call__``; all other behavior is inherited
    from :class:`type`. The dtype sentinel instance is created via
    :meth:`bigint.__new__`, which ensures singleton behavior.

    """

    def __call__(cls, *args, **kwargs):
        if not args and not kwargs:
            return super().__call__()
        # ak.bigint(1) -> scalar, like np.int64(1)
        return bigint_(args[0] if args else 0)


class bigint(metaclass=_BigIntMeta):
    """
    Dtype sentinel for Arkouda's variable-width (arbitrary-precision) integers.

    This class represents the dtype object used by Arkouda arrays storing
    arbitrary-precision integers. It behaves similarly to NumPy’s dtype
    objects (such as ``np.int64``), but corresponds to an unbounded,
    variable-width integer type. Instances of :class:`bigint` are singletons
    created through ``__new__`` so that all dtype references share the same
    object.

    Construction semantics follow NumPy’s pattern: calling ``ak.bigint()``
    returns the dtype sentinel, and calling ``ak.bigint(value)`` returns a
    :class:`bigint_` scalar constructed from ``value``.

    bigint instances compare equal to the ``bigint`` class, to themselves,
    to strings such as ``"bigint"``, and to other dtype-like objects whose
    ``name`` attribute equals ``"bigint"``. This allows interoperability
    across Arkouda’s dtype-resolution system.

    Notes
    -----
    This class represents only the dtype. Scalar values use :class:`bigint_`,
    which inherits from :class:`int`.

    Attributes
    ----------
    name : str
        Canonical dtype name (``"bigint"``).
    kind : str
        Dtype category code (``"ui"``).
    itemsize : int
        Nominal bit-width (128). Actual precision is unbounded.
    ndim : int
        Number of dimensions for the dtype object (always 0).
    shape : tuple
        Empty tuple, following NumPy's dtype API.

    Examples
    --------
    >>> import arkouda as ak
    >>> ak.bigint()
    dtype(bigint)

    >>> ak.bigint(42)
    ak.bigint_(42)

    >>> dtype = ak.bigint()
    >>> dtype.name
    'bigint'
    """

    __slots__ = ()
    name: str = "bigint"
    kind: str = "ui"
    itemsize: int = 128
    ndim: int = 0
    shape: tuple = ()
    _INSTANCE: ClassVar["bigint | None"] = None

    def __new__(cls):
        inst = getattr(cls, "_INSTANCE", None)
        if inst is None:
            inst = super().__new__(cls)
            cls._INSTANCE = inst
        return inst

    def __reduce__(self):
        return (bigint, ())

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"dtype({self.name})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if other is self or other is bigint:
            return True
        if isinstance(other, str):
            return other.lower() == "bigint"
        name = getattr(other, "name", None)
        return name == "bigint"

    def __ne__(self, other):
        return not self.__eq__(other)

    # dtype-like conversion hook (kept for compatibility)
    def type(self, x):
        return int(x)

    @property
    def is_signed(self) -> builtins.bool:
        return True

    @property
    def is_variable_width(self) -> builtins.bool:
        return True


class bigint_(int):
    """
    Scalar type for Arkouda's variable-width integer dtype.

    ``bigint_`` represents an individual arbitrary-precision integer value within
    Arkouda arrays that use the :class:`bigint` dtype. It inherits directly from
    Python's built-in :class:`int`, ensuring full arbitrary-precision semantics
    while providing dtype metadata compatible with Arkouda.

    This class is typically constructed indirectly via ``ak.bigint(value)``,
    which invokes the :class:`_BigIntMeta` metaclass. Direct instantiation is
    also supported.

    Construction
    ------------
    ``bigint_(x)`` converts ``x`` to an integer using Python's ``int``:

    * If ``x`` is ``str`` or ``bytes``, it is interpreted using ``int(x, 0)``,
      allowing hexadecimal (``"0xff"``), binary (``"0b1010"``), and other
      literal forms.
    * Otherwise, ``x`` is passed directly to ``int(x)``.

    Properties
    ----------
    dtype : bigint
        Returns the corresponding dtype sentinel, ``ak.bigint()``.
        This mirrors NumPy scalar behavior (e.g., ``np.int64(1).dtype``).

    Methods
    -------
    item()
        Return the underlying Python ``int`` value, matching NumPy scalar
        semantics.

    Notes
    -----
    ``bigint_`` values behave exactly like Python ``int`` objects in arithmetic,
    hashing, comparison, and formatting. Arkouda arrays wrap and distribute many
    such scalars but do not impose fixed-width limits.

    Examples
    --------
    >>> import arkouda as ak
    >>> x = ak.bigint(123)
    >>> x
    ak.bigint_(123)

    >>> x + 10
    133

    >>> ak.bigint_("0xff")
    ak.bigint_(255)

    >>> x.dtype
    dtype(bigint)

    >>> x.item()
    123
    """

    __slots__ = ()

    def __new__(cls, x=0):
        if isinstance(x, (str, bytes)):
            val = int(x, 0)
        else:
            val = int(x)
        return super().__new__(cls, val)

    @property
    def dtype(self):
        return bigint()

    def item(self):
        return int(self)

    def __repr__(self):
        return f"ak.bigint_({int(self)})"
