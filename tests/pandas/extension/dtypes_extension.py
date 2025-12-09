import re

import numpy as np
import pandas as pd
import pytest

from arkouda.pandas.extension import ArkoudaArray, ArkoudaCategoricalArray, ArkoudaStringArray

# Module under test
from arkouda.pandas.extension._dtypes import (
    ArkoudaBigintDtype,
    ArkoudaBoolDtype,
    ArkoudaCategoricalDtype,
    ArkoudaFloat64Dtype,
    ArkoudaInt64Dtype,
    ArkoudaStringDtype,
    ArkoudaUint8Dtype,
    ArkoudaUint64Dtype,
    _ArkoudaBaseDtype,
)


class TestArkoudaDtypesExtension:
    def test_array_extension_docstrings(self):
        import doctest

        from arkouda.pandas.extension import _dtypes

        result = doctest.testmod(_dtypes, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        assert result.failed == 0, f"Doctest failed: {result.failed} failures"

    @pytest.mark.parametrize(
        "spec, expected_cls",
        [
            ("ak.int64", ArkoudaInt64Dtype),
            ("ak.uint64", ArkoudaUint64Dtype),
            ("ak.uint8", ArkoudaUint8Dtype),
            ("ak.float64", ArkoudaFloat64Dtype),
            ("ak.bool_", ArkoudaBoolDtype),
            ("ak.string", ArkoudaStringDtype),
            ("ak.category", ArkoudaCategoricalDtype),
            ("ak.bigint", ArkoudaBigintDtype),
        ],
    )
    def test_construct_from_string_core_dtypes(self, spec, expected_cls):
        """
        The primary Arkouda-prefixed spec for each dtype should resolve to
        the correct Arkouda*Dtype subclass with its canonical name.
        """
        dt = _ArkoudaBaseDtype.construct_from_string(spec)
        assert isinstance(dt, expected_cls)
        # Don't hard-code the name string; use the class' own canonical name.
        assert dt.name == expected_cls().name

    @pytest.mark.parametrize(
        "spec, expected_cls",
        [
            # int64 aliases
            ("ak.int64", ArkoudaInt64Dtype),
            ("ak_int64", ArkoudaInt64Dtype),
            ("akint64", ArkoudaInt64Dtype),
            ("arkouda.int64", ArkoudaInt64Dtype),
            # uint64 aliases
            ("ak.uint64", ArkoudaUint64Dtype),
            ("ak_uint64", ArkoudaUint64Dtype),
            ("akuint64", ArkoudaUint64Dtype),
            ("arkouda.uint64", ArkoudaUint64Dtype),
            # uint8 aliases
            ("ak.uint8", ArkoudaUint8Dtype),
            ("ak_uint8", ArkoudaUint8Dtype),
            ("akuint8", ArkoudaUint8Dtype),
            ("arkouda.uint8", ArkoudaUint8Dtype),
            # float64 aliases
            ("ak.float64", ArkoudaFloat64Dtype),
            ("ak_float64", ArkoudaFloat64Dtype),
            ("akfloat64", ArkoudaFloat64Dtype),
            ("arkouda.float64", ArkoudaFloat64Dtype),
            # bool aliases
            ("ak.bool_", ArkoudaBoolDtype),
            ("ak_bool", ArkoudaBoolDtype),
            ("akbool", ArkoudaBoolDtype),
            ("arkouda.bool_", ArkoudaBoolDtype),
            # string aliases
            ("ak.string", ArkoudaStringDtype),
            ("ak_string", ArkoudaStringDtype),
            ("akstring", ArkoudaStringDtype),
            ("arkouda.String", ArkoudaStringDtype),
            # category aliases
            ("ak.category", ArkoudaCategoricalDtype),
            ("ak_Category", ArkoudaCategoricalDtype),
            ("akcategory", ArkoudaCategoricalDtype),
            ("arkouda.category", ArkoudaCategoricalDtype),
            # bigint aliases
            ("ak.bigint", ArkoudaBigintDtype),
            ("ak_bigint", ArkoudaBigintDtype),
            ("akBigint", ArkoudaBigintDtype),
            ("arkouda.bigint", ArkoudaBigintDtype),
        ],
    )
    def test_construct_from_string_arkouda_aliases(self, spec, expected_cls):
        """
        All Arkouda-prefixed aliases should resolve to the correct Arkouda*Dtype
        subclass, with the dtype's canonical name.
        """
        dtype = _ArkoudaBaseDtype.construct_from_string(spec)
        assert isinstance(dtype, expected_cls)
        assert dtype.name == expected_cls().name

    @pytest.mark.parametrize(
        "spec",
        [
            "int64",
            "uint64",
            "uint8",
            "float64",
            "bool",
            "bool_",
            "string",
            "category",
            "bigint",
        ],
    )
    def test_construct_from_string_rejects_plain_names(self, spec):
        """
        Bare NumPy/pandas-style names should NOT be claimed by Arkouda; they
        must fall through to pandas/NumPy, so _ArkoudaBaseDtype must raise
        TypeError for them.
        """
        with pytest.raises(TypeError):
            _ArkoudaBaseDtype.construct_from_string(spec)

    @pytest.mark.parametrize(
        "dtype_spec",
        [
            "int64",
            "uint64",
            "uint8",
            "float64",
            "bool",
            "bool_",
            "string",
            "category",
            # NOTE: no "bigint" here; pandas itself doesn't support dtype="bigint"
        ],
    )
    def test_pd_array_with_plain_dtypes_is_not_arkouda(self, dtype_spec):
        """
        Plain pandas/NumPy dtype strings that pandas *does* understand must
        not produce an ArkoudaArray.
        """
        from arkouda.pandas.extension import ArkoudaArray

        arr = pd.array([1, 3, 4], dtype=dtype_spec)
        assert not isinstance(arr, ArkoudaArray)

    def test_pd_array_with_plain_bigint_raises_typeerror(self):
        """
        Pandas itself does not recognize dtype='bigint'; ensure that this
        still raises TypeError and is not intercepted by Arkouda.
        """
        with pytest.raises(TypeError):
            pd.array([1, 3, 4], dtype="bigint")

    @pytest.mark.parametrize(
        "spec, expected_dtype_cls, expected_array_cls, values",
        [
            # numeric + bigint -> ArkoudaArray
            ("ak.int64", ArkoudaInt64Dtype, ArkoudaArray, [1, 3, 4]),
            ("ak.uint64", ArkoudaUint64Dtype, ArkoudaArray, [1, 3, 4]),
            ("ak.uint8", ArkoudaUint8Dtype, ArkoudaArray, [1, 3, 4]),
            ("ak.float64", ArkoudaFloat64Dtype, ArkoudaArray, [1.0, 3.5, 4.25]),
            ("ak.bool_", ArkoudaBoolDtype, ArkoudaArray, [True, False, True]),
            ("ak.bigint", ArkoudaBigintDtype, ArkoudaArray, [1, 3, 4]),
            # string -> ArkoudaStringArray
            ("ak.string", ArkoudaStringDtype, ArkoudaStringArray, ["a", "b", "c"]),
            # category -> ArkoudaCategoricalArray
            ("ak.category", ArkoudaCategoricalDtype, ArkoudaCategoricalArray, ["a", "b", "a"]),
        ],
    )
    def test_pd_array_with_arkouda_aliases_is_arkouda(
        self, spec, expected_dtype_cls, expected_array_cls, values
    ):
        arr = pd.array(values, dtype=spec)

        # storage class should be the appropriate Arkouda EA
        assert isinstance(arr, expected_array_cls)

        # dtype should be the expected Arkouda*Dtype
        assert isinstance(arr.dtype, expected_dtype_cls)
        assert arr.dtype.name == expected_dtype_cls().name

    def test_construct_from_string_invalid(self):
        with pytest.raises(TypeError):
            _ArkoudaBaseDtype.construct_from_string("not-a-real-dtype")

    @pytest.mark.parametrize(
        "cls,expected_np,flags",
        [
            (ArkoudaInt64Dtype, np.dtype("int64"), {"_is_numeric": True}),
            (ArkoudaUint64Dtype, np.dtype("uint64"), {"_is_numeric": True}),
            (ArkoudaUint8Dtype, np.dtype("uint8"), {"_is_numeric": True}),
            (ArkoudaFloat64Dtype, np.dtype("float64"), {"_is_numeric": True}),
            (ArkoudaBoolDtype, np.dtype("bool"), {"_is_boolean": True}),
            (ArkoudaBigintDtype, np.dtype("O"), {}),
            (ArkoudaStringDtype, np.dtype("str_"), {"_is_string": True}),
            (ArkoudaCategoricalDtype, np.dtype("O"), {}),
        ],
    )
    def test_numpy_dtype_and_flags(self, cls, expected_np, flags):
        dt = cls()
        assert dt.numpy_dtype == expected_np

        # Feature flags (pandas 2.x style)
        for flag, expected in flags.items():
            assert getattr(dt, flag) is expected
        # Non-set flags should remain False
        for f in ("_is_numeric", "_is_boolean", "_is_string"):
            if f not in flags:
                assert getattr(dt, f) is False

    @pytest.mark.parametrize(
        "cls,expected_name",
        [
            (ArkoudaInt64Dtype, "int64"),
            (ArkoudaUint64Dtype, "uint64"),
            (ArkoudaUint8Dtype, "uint8"),
            (ArkoudaFloat64Dtype, "float64"),
            (ArkoudaBoolDtype, "bool_"),
            (ArkoudaBigintDtype, "bigint"),
            (ArkoudaStringDtype, "string"),
            (ArkoudaCategoricalDtype, "category"),
        ],
    )
    def test_name_and_repr(self, cls, expected_name):
        dt = cls()
        assert dt.name == expected_name
        # repr should look like ClassName('name')
        assert re.fullmatch(rf"{cls.__name__}\('{re.escape(expected_name)}'\)", repr(dt))

    @pytest.mark.parametrize(
        "cls",
        [
            ArkoudaInt64Dtype,
            ArkoudaUint64Dtype,
            ArkoudaUint8Dtype,
            ArkoudaFloat64Dtype,
            ArkoudaBoolDtype,
            ArkoudaBigintDtype,
            ArkoudaStringDtype,
            ArkoudaCategoricalDtype,
        ],
    )
    def test_construct_array_type_returns_class(self, cls):
        array_type = cls.construct_array_type()
        assert isinstance(array_type, type), "construct_array_type() must return a class"
        # basic sanity: class must implement pandas EA interface attributes
        for required in ("__len__", "dtype"):
            assert hasattr(array_type, required), f"{array_type} missing {required}"

    @pytest.mark.parametrize(
        "cls,na_is_nan,na_type",
        [
            (ArkoudaInt64Dtype, False, (int, np.integer)),
            (ArkoudaUint64Dtype, False, (int, np.integer)),
            (ArkoudaUint8Dtype, False, (int, np.integer)),
            (ArkoudaFloat64Dtype, True, float),
            (ArkoudaBoolDtype, False, (bool, np.bool_)),
            (ArkoudaBigintDtype, False, object),
            (ArkoudaStringDtype, False, str),
            (ArkoudaCategoricalDtype, False, (int, np.integer)),
        ],
    )
    def test_na_value_shape(self, cls, na_is_nan, na_type):
        dt = cls()
        na = dt.na_value
        if na_is_nan:
            assert isinstance(na, float) and np.isnan(na)
        else:
            # accept duck-typing for numeric sentinels
            if isinstance(na_type, tuple):
                assert isinstance(na, na_type)
            else:
                assert isinstance(na, na_type)

    @pytest.mark.parametrize(
        "dtype_cls, data, expect_dtype_cls",
        [
            (ArkoudaInt64Dtype, [1, 2, -1], ArkoudaInt64Dtype),
            (ArkoudaFloat64Dtype, [1.0, np.nan, 3.5], ArkoudaFloat64Dtype),
            (ArkoudaUint8Dtype, [1, 2, 3], ArkoudaUint8Dtype),
            (ArkoudaBoolDtype, [True, False, True], ArkoudaBoolDtype),
        ],
    )
    def test_series_roundtrip_with_arkouda_array(self, dtype_cls, data, expect_dtype_cls):
        """
        Construct a pandas Series from an ArkoudaArray and ensure dtype
        round-trips to the expected Arkouda ExtensionDtype.
        """
        import pandas as pd
        import pandas.testing as pdt

        import arkouda as ak

        from arkouda.pandas.extension._arkouda_array import ArkoudaArray

        ak_arr = ak.array(data, dtype=dtype_cls().name)
        ea = ArkoudaArray(ak_arr)
        s = pd.Series(ea)
        assert isinstance(s.dtype, expect_dtype_cls)
        # len and basic values materialize correctly
        assert len(s) == len(data)

        expected = pd.Series(np.array(data, dtype=object), dtype=object)
        pdt.assert_series_equal(s.astype(object), expected, check_names=False)

    def test_series_with_strings_dtype(self):
        import arkouda as ak

        from arkouda.pandas.extension._arkouda_string_array import ArkoudaStringArray

        a = ak.array(["a", "b", ""])
        sea = ArkoudaStringArray(a)
        s = pd.Series(sea)
        assert isinstance(s.dtype, ArkoudaStringDtype)
        assert s.dtype.name == "string"
        assert s.iloc[2] == ""

    def test_series_with_categorical_dtype(self):
        import arkouda as ak

        from arkouda.pandas.extension._arkouda_categorical_array import (
            ArkoudaCategoricalArray,
        )

        a = ak.Categorical(ak.array(["x", "y", "x"]))
        cea = ArkoudaCategoricalArray(a)
        s = pd.Series(cea)
        assert isinstance(s.dtype, ArkoudaCategoricalDtype)
        # categories round-trip to Python scalars on materialization
        assert list(s.to_numpy()) == ["x", "y", "x"]
