import re

import numpy as np
import pandas as pd
import pytest

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

    def test_construct_from_string_core_numeric(self):
        # int64
        dt = _ArkoudaBaseDtype.construct_from_string("int64")
        assert isinstance(dt, ArkoudaInt64Dtype)
        assert dt.name == "int64"

        # uint64
        dt = _ArkoudaBaseDtype.construct_from_string("uint64")
        assert isinstance(dt, ArkoudaUint64Dtype)
        assert dt.name == "uint64"

        # uint8
        dt = _ArkoudaBaseDtype.construct_from_string("uint8")
        assert isinstance(dt, ArkoudaUint8Dtype)
        assert dt.name == "uint8"

        # float64
        dt = _ArkoudaBaseDtype.construct_from_string("float64")
        assert isinstance(dt, ArkoudaFloat64Dtype)
        assert dt.name == "float64"

    def test_construct_from_string_strings_and_category(self):
        # strings
        dt = _ArkoudaBaseDtype.construct_from_string("str_")
        assert isinstance(dt, ArkoudaStringDtype)
        assert dt.name == "string"

        # category alias
        dt = _ArkoudaBaseDtype.construct_from_string("category")
        assert isinstance(dt, ArkoudaCategoricalDtype)
        assert dt.name == "category"

    def test_construct_from_string_bigint(self):
        dt = _ArkoudaBaseDtype.construct_from_string("bigint")
        assert isinstance(dt, ArkoudaBigintDtype)
        assert dt.name == "bigint"

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
        assert list(s.astype(object)) == ["x", "y", "x"]
