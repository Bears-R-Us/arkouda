# arkouda/pandas/extension/__init__.py

"""
Experimental pandas extension types backed by Arkouda arrays.

This subpackage provides experimental implementations of
:pandas:`pandas.api.extensions.ExtensionArray` and corresponding
extension dtypes that wrap Arkouda distributed arrays.

These classes make it possible to use Arkouda arrays inside pandas
objects such as ``Series`` and ``DataFrame``. They aim to provide
familiar pandas semantics while leveraging Arkouda's distributed,
high-performance backend.

.. warning::
   This module is **experimental**. The API is not stable and may
   change without notice between releases. Use with caution in
   production environments.
"""

from ._arkouda_array import ArkoudaArray
from ._arkouda_categorical_array import ArkoudaCategoricalArray
from ._arkouda_extension_array import ArkoudaExtensionArray
from ._arkouda_string_array import ArkoudaStringArray
from ._dtypes import (
    ArkoudaBigintDtype,
    ArkoudaBoolDtype,
    ArkoudaCategoricalDtype,
    ArkoudaFloat64Dtype,
    ArkoudaInt64Dtype,
    ArkoudaStringDtype,
    ArkoudaUint8Dtype,
    ArkoudaUint64Dtype,
)
from ._index_accessor import ArkoudaIndexAccessor


__all__ = [
    "ArkoudaInt64Dtype",
    "ArkoudaUint64Dtype",
    "ArkoudaUint8Dtype",
    "ArkoudaBigintDtype",
    "ArkoudaBoolDtype",
    "ArkoudaFloat64Dtype",
    "ArkoudaStringDtype",
    "ArkoudaCategoricalDtype",
    "ArkoudaArray",
    "ArkoudaStringArray",
    "ArkoudaCategoricalArray",
    "ArkoudaExtensionArray",
    "ArkoudaIndexAccessor",
]
