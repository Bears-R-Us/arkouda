import operator as op_
import warnings

from itertools import product

import numpy as np
import pytest

import arkouda as ak

from arkouda.testing import assert_almost_equivalent, assert_arkouda_array_equivalent


NUMERIC_TYPES = ["int64", "float64", "bool", "uint64"]
NOT_FLOAT_TYPES = ["bool", "uint64", "int64", "bigint"]
ARKOUDA_SUPPORTED_TYPES = ["bool", "uint64", "int64", "bigint", "float64"]
VALUE_OPS = ["&", "|", "^", "+", "-", "/", "*", "//", "%", "<<", ">>", "**", "<<<", ">>>"]
SMALL_OPS = ["&", "|", "^", "+", "-", "/", "*", "//", "%", "<<<", ">>>"]
LARGE_OPS = ["<<", ">>", "**"]
ARITHMETIC_OPS = ["+", "-", "/", "*", "//", "%", "**"]
BOOL_OPS = ["<", "<=", ">", ">=", "==", "!="]

OP_MAP = {
    # value / bitwise / arithmetic
    "&": op_.and_,
    "|": op_.or_,
    "^": op_.xor,
    "+": op_.add,
    "-": op_.sub,
    "/": op_.truediv,
    "*": op_.mul,
    "//": op_.floordiv,
    "%": op_.mod,
    "<<": op_.lshift,
    ">>": op_.rshift,  # arithmetic right shift in Python
    "**": op_.pow,
    # comparisons
    "<": op_.lt,
    "<=": op_.le,
    ">": op_.gt,
    ">=": op_.ge,
    "==": op_.eq,
    "!=": op_.ne,
}


def make_np_arrays(size, dtype):
    if dtype == "int64":
        return np.random.randint(-(2**32), 2**32, size=size, dtype=dtype)
    elif dtype == "uint64":
        return np.random.randint(-(2**32), 2**32, size=size).astype(dtype)
    elif dtype == "float64":
        return np.random.uniform(-(2**32), 2**32, size=size)
    elif dtype == "bool":
        return np.random.randint(0, 1, size=size, dtype=dtype)
    return None


class TestOperator:
    def test_numpy_equivalency(self, size=100, verbose=pytest.verbose):
        # ignore numpy warnings like divide by 0
        np.seterr(all="ignore")
        global pdarrays
        pdarrays = {
            "int64": ak.arange(0, size, 1),
            "uint64": ak.array(np.arange(2**64 - size, 2**64, 1, dtype=np.uint64)),
            "float64": ak.linspace(-2, 2, size),
            "bool": (ak.arange(0, size, 1) % 2) == 0,
        }
        global ndarrays
        ndarrays = {
            "int64": np.arange(0, size, 1),
            "uint64": np.arange(2**64 - size, 2**64, 1, dtype=np.uint64),
            "float64": np.linspace(-2, 2, size),
            "bool": (np.arange(0, size, 1) % 2) == 0,
        }
        global scalars
        scalars = {
            "int64": 5,
            "uint64": np.uint64(2**63 + 1),
            "float64": -3.14159,
            "bool": True,
        }
        dtypes = pdarrays.keys()
        if verbose:
            print("Operators: ", ak.pdarray.BinOps)
            print("Dtypes: ", dtypes)
            print("pdarrays: ")
            for k, v in pdarrays.items():
                print(k, ": ", v)
            print("ndarrays: ")
            for k, v in ndarrays.items():
                print(k, ": ", v)
            print("scalars: ")
            for k, v in scalars.items():
                print(k, ": ", v)

        def do_op(lt, rt, ls, rs, isarkouda, oper):
            evalstr = ""
            if ls:
                evalstr += f'scalars["{lt}"]'
            else:
                evalstr += f'{("ndarrays", "pdarrays")[isarkouda]}["{lt}"]'
            evalstr += f" {oper} "
            if rs:
                evalstr += f'scalars["{rt}"]'
            else:
                evalstr += f'{("ndarrays", "pdarrays")[isarkouda]}["{rt}"]'
            res = eval(evalstr)
            return res

        results = {
            "neither_implement": [],  # (expression, ak_error)
            "arkouda_minus_numpy": [],  # (expression, ak_result, error_on_exec?)
            "numpy_minus_arkouda": [],  # (expression, ak_result, error_on_exec?)
            "both_implement": [],
        }  # (expression, ak_result, error_on_exec?, dtype_mismatch?, value_mismatch?)
        tests = 0
        for ltype, rtype, op in product(dtypes, dtypes, ak.pdarray.BinOps):
            if op in ("<<<", ">>>"):
                continue
            for lscalar, rscalar in ((False, False), (False, True), (True, False)):
                tests += 1
                expression = (
                    f"{ltype}({('array', 'scalar')[lscalar]}) "
                    f"{op} {rtype}({('array', 'scalar')[rscalar]})"
                )
                try:
                    npres = do_op(ltype, rtype, lscalar, rscalar, False, op)
                except TypeError:  # numpy doesn't implement operation
                    try:
                        akres = do_op(ltype, rtype, lscalar, rscalar, True, op)
                    except RuntimeError as e:
                        if "not implemented" or "unrecognized type" in str(
                            e
                        ):  # neither numpy nor arkouda implement
                            results["neither_implement"].append((expression, str(e)))
                        else:  # arkouda implements with error, np does not implement
                            results["arkouda_minus_numpy"].append((expression, str(e), True))
                        continue
                    except TypeError as e:
                        results["neither_implement"].append((expression, str(e)))
                    # arkouda implements but not numpy
                    results["arkouda_minus_numpy"].append((expression, str(akres), False))
                    continue
                try:
                    akres = do_op(ltype, rtype, lscalar, rscalar, True, op)
                except RuntimeError as e:
                    if "not implemented" or "unrecognized type" in str(
                        e
                    ):  # numpy implements but not arkouda
                        results["numpy_minus_arkouda"].append((expression, str(e), True))
                    else:  # both implement, but arkouda errors
                        results["both_implement"].append((expression, str(e), True, False, False))
                    continue
                # both numpy and arkouda execute without error
                try:
                    akrestype = akres.dtype
                except Exception:
                    warnings.warn(
                        f"Cannot detect return dtype of ak result: {akres} (np result: {npres})"
                    )
                    results["both_implement"].append((expression, str(akres), False, True, False))
                    continue

                if akrestype != npres.dtype:
                    restypes = f"{npres.dtype}(np) vs. {akrestype}(ak)"
                    results["both_implement"].append((expression, restypes, False, True, False))
                    continue
                try:
                    akasnp = akres.to_ndarray()
                except Exception:
                    warnings.warn(f"Could not convert to ndarray: {akres}")
                    results["both_implement"].append((expression, str(akres), True, False, False))
                    continue
                if not np.allclose(akasnp, npres, equal_nan=True):
                    res = f"np: {npres}\nak: {akasnp}"
                    results["both_implement"].append((expression, res, False, False, True))
                    continue
                # Finally, both numpy and arkouda agree on result
                results["both_implement"].append((expression, "", False, False, False))

        print(f"# ops not implemented by numpy or arkouda: {len(results['neither_implement'])}")
        if verbose:
            for expression, err in results["neither_implement"]:
                print(expression)
        print(f"# ops implemented by numpy but not arkouda: {len(results['numpy_minus_arkouda'])}")
        if verbose:
            for expression, err, flag in results["numpy_minus_arkouda"]:
                print(expression)
        print(f"# ops implemented by arkouda but not numpy: {len(results['arkouda_minus_numpy'])}")
        if verbose:
            for expression, res, flag in results["arkouda_minus_numpy"]:
                print(expression, " -> ", res)
        nboth = len(results["both_implement"])
        print(f"# ops implemented by both: {nboth}")
        matches = 0
        execerrors = []
        dtypeerrors = []
        valueerrors = []
        for expression, res, ex, dt, val in results["both_implement"]:
            matches += not any((ex, dt, val))
            if ex:
                execerrors.append((expression, res))
            if dt:
                dtypeerrors.append((expression, res))
            if val:
                valueerrors.append((expression, res))
        print(f"  Matching results:         {matches} / {nboth}")
        print(f"  Arkouda execution errors: {len(execerrors)} / {nboth}")
        if verbose:
            print("\n".join(map(": ".join, execerrors)))
        print(f"  Dtype mismatches:         {len(dtypeerrors)} / {nboth}")
        if verbose:
            print("\n".join(map(": ".join, dtypeerrors)))
        print(f"  Value mismatches:         {len(valueerrors)} / {nboth}")
        if verbose:
            print("\n".join(map(": ".join, valueerrors)))

    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    def test_pdarray_and_scalar_ops(self, dtype):
        pda = ak.ones(100, dtype=dtype)
        npa = np.ones(100, dtype=dtype)
        for scal in 1, np.int64(1):
            for ak_add, np_add in zip((pda + scal, scal + pda), (npa + scal, scal + npa)):
                assert isinstance(ak_add, ak.pdarrayclass.pdarray)
                assert np.allclose(ak_add.to_ndarray(), np_add)

        for scal in 2, np.int64(2):
            for ak_sub, np_sub in zip((pda - scal, scal - pda), (npa - scal, scal - npa)):
                assert isinstance(ak_sub, ak.pdarrayclass.pdarray)
                assert np.allclose(ak_sub.to_ndarray(), np_sub)

        for scal in 5, np.int64(5):
            for ak_mul, np_mul in zip((pda * scal, scal * pda), (npa * scal, scal * npa)):
                assert isinstance(ak_mul, ak.pdarrayclass.pdarray)
                assert np.allclose(ak_mul.to_ndarray(), np_mul)

        if dtype not in ["bool", "uint64"]:
            pda *= 15
            npa *= 15
            for scal in 3, np.int64(3):
                for ak_div, np_div in zip((pda / scal, scal / pda), (npa / scal, scal / npa)):
                    assert isinstance(ak_div, ak.pdarrayclass.pdarray)
                    assert np.allclose(ak_div.to_ndarray(), np_div)

    @pytest.mark.parametrize("dtype", NUMERIC_TYPES)
    def test_concatenation(self, dtype):
        size = 100
        npa = make_np_arrays(size, dtype)
        npa2 = make_np_arrays(size, dtype)

        np_concat = np.concatenate([npa, npa2])
        ak_concat = ak.concatenate([ak.array(npa), ak.array(npa2)])
        assert 200 == len(ak_concat)
        assert dtype == ak_concat.dtype.name
        assert np.allclose(ak_concat.to_ndarray(), np_concat)

    def test_max_bits_concatenation(self):
        # reproducer for issue #2802
        concatenated = ak.concatenate([ak.arange(5, max_bits=3), ak.arange(2**200 - 1, 2**200 + 4)])
        assert concatenated.max_bits == 3
        assert [0, 1, 2, 3, 4, 7, 0, 1, 2, 3] == concatenated.tolist()

    def test_fixed_concatenate(self):
        for pda1, pda2 in zip(
            (ak.arange(4), ak.linspace(0, 3, 4)),
            (ak.arange(4, 7), ak.linspace(4, 6, 3)),
        ):
            ans = list(range(7))
            assert ak.concatenate([pda1, pda2]).tolist() == ans
            assert ak.concatenate([pda2, pda1]).tolist() == (ans[4:] + ans[:4])

    def test_invert(self):
        ak_invert = ~ak.arange(10, dtype=ak.uint64)
        np_invert = ~np.arange(10, dtype=np.uint)
        assert ak_invert.tolist() == np_invert.tolist()

    def test_bool_bool_addition_binop(self):
        np_x = np.array([True, True, False, False])
        np_y = np.array([True, False, True, False])
        ak_x = ak.array(np_x)
        ak_y = ak.array(np_y)
        # Vector-Vector Case
        assert (np_x + np_y).tolist() == (ak_x + ak_y).tolist()
        # Scalar-Vector Case
        assert (np_x[0] + np_y).tolist() == (ak_x[0] + ak_y).tolist()
        assert (np_x[-1] + np_y).tolist() == (ak_x[-1] + ak_y).tolist()
        # Vector-Scalar Case
        assert (np_x + np_y[0]).tolist() == (ak_x + ak_y[0]).tolist()
        assert (np_x + np_y[-1]).tolist() == (ak_x + ak_y[-1]).tolist()

    def test_bool_bool_addition_opeq(self):
        np_x = np.array([True, True, False, False])
        np_y = np.array([True, False, True, False])
        ak_x = ak.array(np_x)
        ak_y = ak.array(np_y)
        np_x += np_y
        ak_x += ak_y
        # Vector-Vector Case
        assert np_x.tolist() == ak_x.tolist()
        # Scalar-Vector Case
        # True
        np_true = np_x[0]
        ak_true = ak_x[0]
        np_true += np_y
        ak_true += ak_y
        assert np_x.tolist() == ak_x.tolist()
        # False
        np_false = np_x[-1]
        ak_false = ak_x[-1]
        np_false += np_y
        ak_false += ak_y
        assert np_x.tolist() == ak_x.tolist()

    def test_uint_bool_binops(self):
        # Test fix for issue #1932
        # Adding support to binopvv to correctly handle uint and bool types
        ak_uint = ak.arange(10, dtype=ak.uint64)
        ak_bool = ak_uint % 2 == 0
        assert (ak_uint + ak_bool).tolist() == (ak.arange(10) + ak_bool).tolist()

    def test_int_uint_binops(self):
        np_int = np.arange(-5, 5)
        ak_int = ak.array(np_int)

        np_uint = np.arange(2**64 - 10, 2**64, dtype=np.uint64)
        ak_uint = ak.array(np_uint)

        # Vector-Vector Case (Division and Floor Division)
        assert np.allclose((ak_uint / ak_uint).to_ndarray(), np_uint / np_uint, equal_nan=True)
        assert np.allclose((ak_int / ak_uint).to_ndarray(), np_int / np_uint, equal_nan=True)
        assert np.allclose((ak_uint / ak_int).to_ndarray(), np_uint / np_int, equal_nan=True)
        assert np.allclose((ak_uint // ak_uint).to_ndarray(), np_uint // np_uint, equal_nan=True)
        assert np.allclose((ak_int // ak_uint).to_ndarray(), np_int // np_uint, equal_nan=True)
        assert np.allclose((ak_uint // ak_int).to_ndarray(), np_uint // np_int, equal_nan=True)

        # Scalar-Vector Case (Division and Floor Division)
        assert np.allclose((ak_uint[0] / ak_uint).to_ndarray(), np_uint[0] / np_uint, equal_nan=True)
        assert np.allclose((ak_int[0] / ak_uint).to_ndarray(), np_int[0] / np_uint, equal_nan=True)
        assert np.allclose((ak_uint[0] / ak_int).to_ndarray(), np_uint[0] / np_int, equal_nan=True)
        assert np.allclose((ak_uint[0] // ak_uint).to_ndarray(), np_uint[0] // np_uint, equal_nan=True)
        assert np.allclose((ak_int[0] // ak_uint).to_ndarray(), np_int[0] // np_uint, equal_nan=True)
        assert np.allclose((ak_uint[0] // ak_int).to_ndarray(), np_uint[0] // np_int, equal_nan=True)

        # Vector-Scalar Case (Division and Floor Division)
        assert np.allclose((ak_uint / ak_uint[0]).to_ndarray(), np_uint / np_uint[0], equal_nan=True)
        assert np.allclose((ak_int / ak_uint[0]).to_ndarray(), np_int / np_uint[0], equal_nan=True)
        assert np.allclose((ak_uint / ak_int[0]).to_ndarray(), np_uint / np_int[0], equal_nan=True)
        assert np.allclose((ak_uint // ak_uint[0]).to_ndarray(), np_uint // np_uint[0], equal_nan=True)
        assert np.allclose((ak_int // ak_uint[0]).to_ndarray(), np_int // np_uint[0], equal_nan=True)
        assert np.allclose((ak_uint // ak_int[0]).to_ndarray(), np_uint // np_int[0], equal_nan=True)

    def test_float_uint_binops(self):
        # Test fix for issue #1620
        np_uint = make_np_arrays(10, "uint64")
        ak_uint = ak.array(np_uint)
        scalar_uint = np.uint64(2**63 + 1)

        np_float = make_np_arrays(10, "float64")
        ak_float = ak.array(np_float)
        scalar_float = -3.14

        ak_uints = [ak_uint, scalar_uint]
        np_uints = [np_uint, scalar_uint]
        ak_floats = [ak_float, scalar_float]
        np_floats = [np_float, scalar_float]
        for aku, akf, npu, npf in zip(ak_uints, ak_floats, np_uints, np_floats):
            assert np.allclose((ak_uint + akf).to_ndarray(), np_uint + npf, equal_nan=True)
            assert np.allclose((akf + ak_uint).to_ndarray(), npf + np_uint, equal_nan=True)
            assert np.allclose((ak_float + aku).to_ndarray(), np_float + npu, equal_nan=True)
            assert np.allclose((aku + ak_float).to_ndarray(), npu + np_float, equal_nan=True)

            assert np.allclose((ak_uint - akf).to_ndarray(), np_uint - npf, equal_nan=True)
            assert np.allclose((akf - ak_uint).to_ndarray(), npf - np_uint, equal_nan=True)
            assert np.allclose((ak_float - aku).to_ndarray(), np_float - npu, equal_nan=True)
            assert np.allclose((aku - ak_float).to_ndarray(), npu - np_float, equal_nan=True)

            assert np.allclose((ak_uint * akf).to_ndarray(), np_uint * npf, equal_nan=True)
            assert np.allclose((akf * ak_uint).to_ndarray(), npf * np_uint, equal_nan=True)
            assert np.allclose((ak_float * aku).to_ndarray(), np_float * npu, equal_nan=True)
            assert np.allclose((aku * ak_float).to_ndarray(), npu * np_float, equal_nan=True)

            assert np.allclose((ak_uint / akf).to_ndarray(), np_uint / npf, equal_nan=True)
            assert np.allclose((akf / ak_uint).to_ndarray(), npf / np_uint, equal_nan=True)
            assert np.allclose((ak_float / aku).to_ndarray(), np_float / npu, equal_nan=True)
            assert np.allclose((aku / ak_float).to_ndarray(), npu / np_float, equal_nan=True)

            assert np.allclose((ak_uint // akf).to_ndarray(), np_uint // npf, equal_nan=True)
            assert np.allclose((akf // ak_uint).to_ndarray(), npf // np_uint, equal_nan=True)
            assert np.allclose((ak_float // aku).to_ndarray(), np_float // npu, equal_nan=True)
            assert np.allclose((aku // ak_float).to_ndarray(), npu // np_float, equal_nan=True)

            assert np.allclose((ak_uint**akf).to_ndarray(), np_uint**npf, equal_nan=True)
            assert np.allclose((akf**ak_uint).to_ndarray(), npf**np_uint, equal_nan=True)
            assert np.allclose((ak_float**aku).to_ndarray(), np_float**npu, equal_nan=True)
            assert np.allclose((aku**ak_float).to_ndarray(), npu**np_float, equal_nan=True)

            assert np.allclose((ak_float % aku).to_ndarray(), np_float % npu, equal_nan=True)
            assert np.allclose((aku % ak_float).to_ndarray(), npu % np_float, equal_nan=True)

    def test_shift_maxbits_binop(self):
        # This tests for a bug when left shifting by a value >=64 bits for int/uint, Issue #2099
        max_bits = 2**63 - 1
        for dtype in "int64", "uint64":
            # Value arrays
            ak_arr = ak.array([max_bits, max_bits, max_bits, max_bits], dtype=dtype)
            np_arr = np.array([max_bits, max_bits, max_bits, max_bits], dtype=dtype)

            # Shifting value arrays
            ak_shift = ak.array([62, 63, 64, 65], dtype=dtype)
            np_shift = np.array([62, 63, 64, 65], dtype=dtype)

            # Binopvs case
            for i in range(62, 66):
                assert np.allclose((ak_arr << i).to_ndarray(), np_arr << i)
                assert np.allclose((ak_arr >> i).to_ndarray(), np_arr >> i)

            # Binopsv case
            assert (max_bits << ak_shift).tolist() == (max_bits << np_shift).tolist()
            assert (max_bits >> ak_shift).tolist() == (max_bits >> np_shift).tolist()

            # Binopvv case, Same type
            assert (ak_arr << ak_shift).tolist() == (np_arr << np_shift).tolist()
            assert (ak_arr >> ak_shift).tolist() == (np_arr >> np_shift).tolist()

    def test_shift_bool_int64_binop(self):
        # This tests for a missing implementation of bit shifting booleans and ints, Issue #2945
        np_int = make_np_arrays(10, "int64")
        ak_int = ak.array(np_int)
        np_bool = make_np_arrays(10, "bool")
        ak_bool = ak.array(np_bool)

        # Binopvv case
        assert np.allclose((ak_int >> ak_bool).to_ndarray(), np_int >> np_bool)
        assert np.allclose((ak_int << ak_bool).to_ndarray(), np_int << np_bool)
        assert np.allclose((ak_bool >> ak_int).to_ndarray(), np_bool >> np_int)
        assert np.allclose((ak_bool << ak_int).to_ndarray(), np_bool << np_int)

        # Binopvs case
        assert np.allclose((ak_int >> ak_bool[0]).to_ndarray(), np_int >> np_bool[0])
        assert np.allclose((ak_int << ak_bool[0]).to_ndarray(), np_int << np_bool[0])
        assert np.allclose((ak_bool >> ak_int[0]).to_ndarray(), np_bool >> np_int[0])
        assert np.allclose((ak_bool << ak_int[0]).to_ndarray(), np_bool << np_int[0])

        # Binopsv case
        assert np.allclose((ak_int[0] >> ak_bool).to_ndarray(), np_int[0] >> np_bool)
        assert np.allclose((ak_int[0] << ak_bool).to_ndarray(), np_int[0] << np_bool)
        assert np.allclose((ak_bool[0] >> ak_int).to_ndarray(), np_bool[0] >> np_int)
        assert np.allclose((ak_bool[0] << ak_int).to_ndarray(), np_bool[0] << np_int)

    @pytest.mark.parametrize("dtype", [ak.int64, ak.uint64])
    def test_shift_equals_scalar_binops(self, dtype):
        ak_vector = ak.arange(0, 5, dtype=dtype)
        np_vector = np.arange(5, dtype=dtype)
        shift_scalars = [
            dtype(1),
            dtype(5),
            1,
            5,
            True,
            False,
        ]

        for x in shift_scalars:
            assert ak_vector.tolist() == np_vector.tolist()

            ak_vector <<= x
            np_vector <<= x
            assert ak_vector.tolist() == np_vector.tolist()

            ak_vector >>= x
            np_vector >>= x
            assert ak_vector.tolist() == np_vector.tolist()

    def test_shift_equals_vector_binops(self):
        vector_pairs = [
            (ak.arange(0, 5, dtype=ak.int64), np.arange(5, dtype=np.int64)),
            (ak.arange(0, 5, dtype=ak.uint64), np.arange(5, dtype=np.uint64)),
        ]
        shift_vectors = [
            ak.ones(5, dtype=ak.int64),
            ak.zeros(5, dtype=ak.int64),
            ak.ones(5, dtype=ak.uint64),
            ak.zeros(5, dtype=ak.uint64),
            ak.array([1, 0, 1, 0, 1], dtype=bool),
            ak.array([1, 1, 1, 1, 1], dtype=bool),
        ]

        for ak_vector, np_vector in vector_pairs:
            for v in shift_vectors:
                if (v[0].dtype.kind != "b") and (ak_vector[0].dtype.kind != v[0].dtype.kind):
                    continue

                assert ak_vector.tolist() == np_vector.tolist()

                ak_vector <<= v
                np_vector <<= v.to_ndarray()
                assert ak_vector.tolist() == np_vector.tolist()

                ak_vector >>= v
                np_vector >>= v.to_ndarray()
                assert ak_vector.tolist() == np_vector.tolist()

    def test_concatenate_type_preservation(self):
        # Test that concatenate preserves special pdarray types (IPv4, Datetime, BitVector, ...)
        from arkouda.numpy.util import generic_concat as akuconcat

        pda_one, pda_two = ak.arange(1, 4), ak.arange(4, 7)
        pda_concat = ak.concatenate([pda_one, pda_two])

        for special_type in ak.IPv4, ak.Datetime, ak.Timedelta, ak.BitVector:
            special_one, special_two = special_type(pda_one), special_type(pda_two)
            special_concat = ak.concatenate([special_one, special_two])
            assert isinstance(special_concat, special_type)
            assert special_type(pda_concat).tolist() == special_concat.tolist()

            # test single and empty
            assert isinstance(ak.concatenate([special_one]), special_type)
            assert special_one.tolist() == ak.concatenate([special_one]).tolist()
            assert isinstance(
                ak.concatenate([special_type(ak.array([], dtype=ak.int64))]),
                special_type,
            )

            # verify ak.util.concatenate still works
            special_aku_concat = akuconcat([special_one, special_two])
            assert isinstance(special_aku_concat, special_type)
            assert special_type(pda_concat).tolist() == special_aku_concat.tolist()

        # Test failure with mixed types
        with pytest.raises(TypeError):
            ak.concatenate([ak.Datetime(pda_one), ak.BitVector(pda_two)])

    def test_floor_div_edge_cases(self):
        scalar_edge_cases = [-np.inf, -3.14, -0.0, np.nan, 0.0, 3.14, np.inf]
        np_edge_cases = np.array(scalar_edge_cases)
        ak_edge_cases = ak.array(np_edge_cases)

        for s in scalar_edge_cases:
            assert np.allclose((ak_edge_cases // s).to_ndarray(), np_edge_cases // s, equal_nan=True)
            assert np.allclose((s // ak_edge_cases).to_ndarray(), s // np_edge_cases, equal_nan=True)

            # test both vector // vector
            n_vect = np.full(len(scalar_edge_cases), s)
            a_vect = ak.array(n_vect)
            assert np.allclose(
                (ak_edge_cases // a_vect).to_ndarray(),
                np_edge_cases // n_vect,
                equal_nan=True,
            )
            assert np.allclose(
                (a_vect // ak_edge_cases).to_ndarray(),
                n_vect // np_edge_cases,
                equal_nan=True,
            )

    def test_pda_power(self):
        n = np.array([10, 5, 2])
        a = ak.array(n)

        assert ak.power(a, 2).tolist() == np.power(n, 2).tolist()
        assert ak.power(a, ak.array([2, 3, 4])).tolist() == np.power(n, [2, 3, 4]).tolist()

        # Test a singleton with and without a Boolean argument
        a = ak.array([7])
        assert ak.power(a, 3, True).tolist() == ak.power(a, 3).tolist()
        assert ak.power(a, 3, False).tolist() == a.tolist()

        # Test an with and without a Boolean argument, all the same
        a = ak.array([0, 0.0, 1, 7.0, 10])
        assert ak.power(a, 3, ak.ones(5, bool)).tolist() == ak.power(a, 3).tolist()
        assert ak.power(a, 3, ak.zeros(5, bool)).tolist() == a.tolist()

        # Test a singleton with a mixed Boolean argument
        a = ak.arange(10)
        assert [i if i % 2 else i**2 for i in range(10)] == ak.power(a, 2, a % 2 == 0).tolist()

        # Test invalid input, negative
        n = np.array([-1.0, -3.0])
        a = ak.array(n)
        assert np.allclose(ak.power(a, 0.5).to_ndarray(), np.power(n, 0.5), equal_nan=True)

        # Test edge case input, inf
        infs = [np.inf, -np.inf]
        assert (np.power(np.array(infs), 2) == ak.power(ak.array(infs), 2).to_ndarray()).all()

    def test_pda_sqrt(self):
        n = np.array([4, 16.0, -1, 0, np.inf])
        a = ak.array(n)
        assert np.allclose(ak.sqrt(a).to_ndarray(), np.sqrt(n), equal_nan=True)

        # Test with a mixed Boolean array
        a = ak.arange(5)
        assert [i if i % 2 else i**0.5 for i in range(5)] == ak.sqrt(a, a % 2 == 0).tolist()

    def test_uint_and_bigint_operation_equals(self):
        def declare_arrays():
            u_arr = ak.array(
                [0, 1, 2, 3, 4, 2**64 - 5, 2**64 - 4, 2**64 - 3, 2**64 - 2, 2**64 - 1],
                dtype=ak.uint64,
            )
            bi_arr = ak.array(
                [0, 1, 2, 3, 4, 2**64 - 5, 2**64 - 4, 2**64 - 3, 2**64 - 2, 2**64 - 1],
                dtype=ak.bigint,
                max_bits=64,
            )
            np_arr = np.array(
                [0, 1, 2, 3, 4, 2**64 - 5, 2**64 - 4, 2**64 - 3, 2**64 - 2, 2**64 - 1],
                dtype=np.uint,
            )
            return u_arr, bi_arr, np_arr

        u_arr, bi_arr, np_arr = declare_arrays()
        i_arr = ak.arange(10)
        f_arr = ak.linspace(1, 5, 10)
        b_arr = i_arr % 2 == 0
        u, i, f, b = np.uint(7), 7, 3.14, True

        # test uint opequals uint functionality against numpy
        for arr in u_arr, bi_arr, np_arr:
            arr += u
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        for arr in u_arr, bi_arr, np_arr:
            arr += arr
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        for arr in u_arr, bi_arr, np_arr:
            arr -= u
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        for arr in u_arr, bi_arr, np_arr:
            arr -= arr
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        # redeclare because subtract by self zeroed out
        u_arr, bi_arr, np_arr = declare_arrays()

        for arr in u_arr, bi_arr, np_arr:
            arr *= u
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        for arr in u_arr, bi_arr, np_arr:
            arr *= arr
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        for arr in u_arr, bi_arr, np_arr:
            arr **= u
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        for arr in u_arr, bi_arr, np_arr:
            arr **= arr
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        for arr in u_arr, bi_arr, np_arr:
            arr %= u
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        for arr in u_arr, bi_arr, np_arr:
            arr //= u
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        # redeclare because divide zeroed out
        u_arr, bi_arr, np_arr = declare_arrays()

        for arr in u_arr, bi_arr, np_arr:
            arr //= arr
        assert u_arr.tolist() == np_arr.tolist()
        assert bi_arr.tolist() == np_arr.tolist()

        # the only arrays that can be added in place are uint and bool
        # scalars are cast to same type if possible
        u_arr = ak.arange(10, dtype=ak.uint64)
        bi_arr = ak.arange(10, dtype=ak.bigint)
        for v in [b_arr, u, b, i, f]:
            u_tmp = u_arr[:]
            bi_tmp = bi_arr[:]
            i_tmp = i_arr[:]
            u_tmp += v
            bi_tmp += v
            i_tmp += v
            assert u_tmp.tolist() == i_tmp.tolist()
            assert u_tmp.tolist() == bi_tmp.tolist()

        # adding a float or int inplace could have a result which is not a uint
        for e in [i_arr, f_arr]:
            with pytest.raises(RuntimeError):
                u_arr += e

        with pytest.raises(RuntimeError):
            bi_arr += f_arr

        # verify other types can have uint applied to them
        f_arr += u_arr
        f_arr += u

    @pytest.mark.parametrize("op", SMALL_OPS)
    @pytest.mark.parametrize("other_type", NOT_FLOAT_TYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_bigint_binops_not_float_dtype_small_ops(self, op, other_type, size):
        seed = pytest.seed if pytest.seed is not None else 1
        bigint_arr = ak.randint(0, 2**64, size, dtype=ak.uint64, seed=seed) + 2**200
        other_dtype = other_type if other_type != "bigint" else "uint64"
        min_val = 1
        max_val = 2**64

        if other_type == "bool":
            max_val = 2
        if other_type == "uint8":  # Although not yet supported, the hope is that it will be
            max_val = 256
        if other_type == "int64":
            min_val = -(2**63)
            max_val = 2**63
        other_arr = ak.randint(min_val, max_val, size, dtype=other_dtype, seed=seed + 1)
        if other_dtype == "int64":
            other_arr = ak.where(other_arr == 0, 1, other_arr)
        if other_type == "bigint":
            other_arr = other_arr + 2**200
        if op in {"<<<", ">>>"}:
            bigint_arr.max_bits = 256
            # Just testing that it doesn't crash, since numpy doesn't have this
            if op == "<<<":
                bigint_arr.rotl(other_arr)
                if other_type != "bool":
                    other_arr.rotl(bigint_arr)
            else:
                bigint_arr.rotr(other_arr)
                if other_type != "bool":
                    other_arr.rotr(bigint_arr)
            return

        np_bigint = bigint_arr.to_ndarray()  # noqa: F841
        np_other = other_arr.to_ndarray()  # noqa: F841

        op_fcn = OP_MAP[op]

        ak_result = op_fcn(bigint_arr, other_arr)
        np_result = op_fcn(np_bigint, np_other)

        if op == "/":
            np_result = ak.array(np_result, dtype="float64")
            assert_almost_equivalent(ak_result, np_result)
        else:
            assert_arkouda_array_equivalent(ak_result, np_result)

        ak_result = op_fcn(other_arr, bigint_arr)
        np_result = op_fcn(np_other, np_bigint)

        if op == "/":
            np_result = ak.array(np_result, dtype="float64")
            assert_almost_equivalent(ak_result, np_result)
        else:
            assert_arkouda_array_equivalent(ak_result, np_result)

    @pytest.mark.parametrize("op", ARITHMETIC_OPS)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_bigint_binops_float_dtype_arith_ops(self, op, size):
        seed = pytest.seed if pytest.seed is not None else 1
        bigint_arr = ak.randint(0, 2**64, size, dtype=ak.uint64, seed=seed) + 2**200
        min_val = 1
        max_val = 2**64

        other_arr = ak.randint(min_val, max_val, size, dtype=ak.float64, seed=seed + 1)

        if op == "**":
            other_arr = other_arr % 5

        np_bigint = bigint_arr.to_ndarray()  # noqa: F841
        np_other = other_arr.to_ndarray()  # noqa: F841

        op_fcn = OP_MAP[op]

        ak_result = op_fcn(bigint_arr, other_arr)
        np_result = op_fcn(np_bigint, np_other)

        np_result = ak.array(np_result, dtype="float64")
        assert_almost_equivalent(ak_result, np_result)

    @pytest.mark.parametrize("op", BOOL_OPS)
    @pytest.mark.parametrize("other_type", ARKOUDA_SUPPORTED_TYPES)
    @pytest.mark.parametrize("size", pytest.prob_size)
    def test_bigint_comparison_ops(self, op, other_type, size):
        seed = pytest.seed if pytest.seed is not None else 1
        bigint_arr = ak.randint(0, 2**64, size, dtype=ak.uint64, seed=seed) + 2**200
        other_dtype = other_type if other_type != "bigint" else "uint64"
        min_val = 1
        max_val = 2**64

        if other_type == "bool":
            max_val = 2
        if other_type == "uint8":  # Although not yet supported, the hope is that it will be
            max_val = 255
        if other_type == "int64":
            max_val = 2**63
        other_arr = ak.randint(min_val, max_val, size, dtype=other_dtype, seed=seed + 1)
        if other_type == "bigint":
            other_arr = other_arr + 2**200

        np_bigint = bigint_arr.to_ndarray()  # noqa: F841
        np_other = other_arr.to_ndarray()  # noqa: F841

        op_fcn = OP_MAP[op]

        ak_result = op_fcn(bigint_arr, other_arr)
        np_result = op_fcn(np_bigint, np_other)

        assert_arkouda_array_equivalent(ak_result, np_result)

        ak_result = op_fcn(bigint_arr, other_arr)
        np_result = op_fcn(np_bigint, np_other)

        assert_almost_equivalent(ak_result, np_result)

    def test_error_handling(self):
        # Test NotImplmentedError that prevents pdarray / Strings iteration
        for arr in ak.ones(100), ak.array([f"String {i}" for i in range(10)]):
            with pytest.raises(NotImplementedError):
                iter(arr)

        with pytest.raises(TypeError):
            ak.ones(100).any([0])

        with pytest.raises(AttributeError):
            ak.unique(list(range(10)))

        with pytest.raises(ValueError):
            ak.concatenate([ak.ones(100), ak.array([True])])

    def test_str_repr(self):
        ak.client.pdarrayIterThresh = 5
        assert "[1 2 3]" == ak.array([1, 2, 3]).__str__()
        assert "[1 2 3 ... 17 18 19]" == ak.arange(1, 20).__str__()

        # Printing floating point values changed precision in Chapel 1.32
        answers = ["[1.100000e+00 2.300000e+00 5.000000e+00]", "[1.1 2.3 5.0]"]
        assert ak.array([1.1, 2.3, 5]).__str__() in answers

        answers = [
            "[0.000000e+00 5.263158e-01 1.052632e+00 ... 8.947368e+00 9.473684e+00 1.000000e+01]",
            "[0.0 0.526316 1.05263 ... 8.94737 9.47368 10.0]",
        ]
        assert ak.linspace(0, 10, 20).__str__() in answers
        assert "[False False False]" == ak.isnan(ak.array([1.1, 2.3, 5])).__str__()
        assert "[False False False ... False False False]" == ak.isnan(ak.linspace(0, 10, 20)).__str__()

        # Test __repr__()
        assert "array([1 2 3])" == ak.array([1, 2, 3]).__repr__()
        assert "array([1 2 3 ... 17 18 19])" == ak.arange(1, 20).__repr__()
        answers = [
            "array([1.1000000000000001 2.2999999999999998 5])",
            "array([1.1 2.3 5])",
            "array([1.1000000000000001 2.2999999999999998 5.00000000000000000])",
        ]
        assert ak.array([1.1, 2.3, 5]).__repr__() in answers

        answers = [
            "array([0 0.52631578947368418 1.0526315789473684 ..."
            " 8.9473684210526319 9.473684210526315 10])",
            "array([0 0.5 1.1 ... 8.9 9.5 10])",
            "array([0.00000000000000000 0.52631578947368418 1.0526315789473684 ..."
            " 8.9473684210526319 9.473684210526315 10.00000000000000000])",
        ]
        assert ak.linspace(0, 10, 20).__repr__() in answers
        assert "array([False False False])" == ak.isnan(ak.array([1.1, 2.3, 5])).__repr__()
        assert (
            "array([False False False ... False False False])"
            == ak.isnan(ak.linspace(0, 10, 20)).__repr__()
        )

        # Don't forget to set this back for other tests
        ak.client.pdarrayIterThresh = ak.client.pdarrayIterThreshDefVal

    def test_bigint_binops(self):
        # test bigint array with max_bits=64 against an equivalent uint64
        u = ak.array([0, 1, 2, 2**64 - 3, 2**64 - 2, 2**64 - 1], dtype=ak.uint64)
        bi = ak.array([0, 1, 2, 2**64 - 3, 2**64 - 2, 2**64 - 1], dtype=ak.bigint, max_bits=64)
        mod_by = 2**64

        bi_range = ak.arange(6, dtype=ak.bigint)
        u_range = ak.arange(6, dtype=ak.uint64)
        i_range = ak.arange(6, dtype=ak.int64)
        neg_range = -i_range
        b = u_range % 2 == 0
        bi_scalar = 2**100
        i_scalar = -10
        u_scalar = 10

        # logical bit ops: only work if both arguments are bigint
        assert (u & u_range).tolist() == (bi & bi_range).tolist()
        assert [(bi[i] & bi_scalar) % mod_by for i in range(bi.size)] == (bi & bi_scalar).tolist()
        assert [(bi_scalar & bi[i]) % mod_by for i in range(bi.size)] == (bi_scalar & bi).tolist()

        assert (u | u_range).tolist() == (bi | bi_range).tolist()
        assert [(bi[i] | bi_scalar) % mod_by for i in range(bi.size)] == (bi | bi_scalar).tolist()
        assert [(bi_scalar | bi[i]) % mod_by for i in range(bi.size)] == (bi_scalar | bi).tolist()

        assert (u ^ u_range).tolist() == (bi ^ bi_range).tolist()
        assert [(bi[i] ^ bi_scalar) % mod_by for i in range(bi.size)] == (bi ^ bi_scalar).tolist()
        assert [(bi_scalar ^ bi[i]) % mod_by for i in range(bi.size)] == (bi_scalar ^ bi).tolist()

        # bit shifts: left side must be bigint, right side must be int/uint
        ans = u << u_range
        assert ans.tolist() == (bi << u_range).tolist()
        assert ans.tolist() == (bi << i_range).tolist()

        ans = u >> u_range
        assert ans.tolist() == (bi >> u_range).tolist()
        assert ans.tolist() == (bi >> i_range).tolist()

        ans = u.rotl(u_range)
        assert ans.tolist() == bi.rotl(u_range).tolist()
        assert ans.tolist() == bi.rotl(i_range).tolist()
        ans = u.rotr(u_range)
        assert ans.tolist() == bi.rotr(u_range).tolist()
        assert ans.tolist() == bi.rotr(i_range).tolist()

        # ops where left side has to bigint
        ans = u // u_range
        for ran in bi_range, u_range, i_range:
            assert ans.tolist() == (bi // ran).tolist()

        ans = u % u_range
        for ran in bi_range, u_range, i_range:
            assert ans.tolist() == (bi % ran).tolist()

        ans = u**u_range
        for ran in bi_range, u_range, i_range:
            assert ans.tolist() == (bi**ran).tolist()

        # ops where either side can any of bigint, int, uint, bool
        ans = u + u_range
        for ran in bi_range, u_range, i_range:
            assert ans.tolist() == (bi + ran).tolist()
            assert ans.tolist() == (ran + bi).tolist()

        ans = u + b
        assert ans.tolist() == (bi + b).tolist()
        assert ans.tolist() == (b + bi).tolist()
        for s in [i_scalar, u_scalar, bi_scalar]:
            assert [(bi[i] + s) % mod_by for i in range(bi.size)] == (bi + s).tolist()
            assert [(s + bi[i]) % mod_by for i in range(bi.size)] == (s + bi).tolist()

        ans = u - u_range
        for ran in bi_range, u_range, i_range:
            assert ans.tolist() == (bi - ran).tolist()
        assert (u - b).tolist() == (bi - b).tolist()
        assert (b - u).tolist() == (b - bi).tolist()

        for s in [i_scalar, u_scalar, bi_scalar]:
            assert [(bi[i] - s) % mod_by for i in range(bi.size)] == (bi - s).tolist()
            assert [(s - bi[i]) % mod_by for i in range(bi.size)] == (s - bi).tolist()

        assert (bi - neg_range).tolist() == (bi + u_range).tolist()

        ans = u * u_range
        for ran in bi_range, u_range, i_range:
            assert ans.tolist() == (bi * ran).tolist()
        ans = u * b
        assert ans.tolist() == (bi * b).tolist()
        assert ans.tolist() == (b * bi).tolist()

        for s in [i_scalar, u_scalar, bi_scalar]:
            assert [(bi[i] * s) % mod_by for i in range(bi.size)] == (bi * s).tolist()
            assert [(s * bi[i]) % mod_by for i in range(bi.size)] == (s * bi).tolist()

    def test_bigint_rotate(self):
        # see issue #2214
        # verify bigint pdarray correctly rotate when shift_amount exceeds max_bits
        # in this test we are rotating 10 with max_bits=4, so even rotations will equal 10
        # and odd rotations will equal 5. We test rotations up to 10 (which is > 4)

        # rotate by scalar
        for i in range(10):
            assert ak.array([10], dtype=ak.bigint, max_bits=4).rotl(i) == 10 if i % 2 == 0 else 5
            assert ak.array([10], dtype=ak.bigint, max_bits=4).rotr(i) == 10 if i % 2 == 0 else 5

        # rotate by array
        left_rot = ak.bigint_from_uint_arrays([ak.full(10, 10, ak.uint64)], max_bits=4).rotl(
            ak.arange(10)
        )
        right_rot = ak.bigint_from_uint_arrays([ak.full(10, 10, ak.uint64)], max_bits=4).rotr(
            ak.arange(10)
        )
        ans = [10 if i % 2 == 0 else 5 for i in range(10)]
        assert left_rot.tolist() == ans
        assert right_rot.tolist() == ans

    def test_float_mods(self):
        edge_cases = [
            np.nan,
            -np.inf,
            -7.0,
            -3.14,
            -0.0,
            0.0,
            3.14,
            7.0,
            np.inf,
            np.nan,
        ]

        # get 2 random permutations of edgecases
        rand_edge_cases1 = np.random.permutation(edge_cases)
        rand_edge_cases2 = np.random.permutation(edge_cases)
        # floats containing negatives and repeating decimals
        float_arr = np.linspace(-3.5, 3.5, 10)
        # ints containing negatives and 0
        int_arr = np.arange(-5, 5)
        i_scal = -17
        # uints > 2**63
        uint_arr = np.arange(2**64 - 10, 2**64, dtype=np.uint64)
        u_scal = np.uint(2**63 + 1)

        args = [
            rand_edge_cases1,
            rand_edge_cases2,
            float_arr,
            int_arr,
            uint_arr,
            i_scal,
            u_scal,
        ]
        # add all the float edge cases as scalars
        args.extend(edge_cases)

        def type_helper(x):
            return ak.resolve_scalar_dtype(x) if ak.isSupportedNumber(x) else x.dtype.name

        # take the product of args (i.e. every possible combination)
        for a, b in product(args, args):
            if all(ak.isSupportedNumber(arg) for arg in [a, b]):
                # we don't support scalar scalar
                continue
            if not any(type_helper(arg) == "float64" for arg in [a, b]):
                # at least one must be float to do fmod
                continue

            # convert ndarrays to pdarray and leave scalars as is
            ak_a = a if ak.isSupportedNumber(a) else ak.array(a)
            ak_b = b if ak.isSupportedNumber(b) else ak.array(b)

            # verify mod and fmod match numpy
            assert np.allclose(ak.mod(ak_a, ak_b).to_ndarray(), np.mod(a, b), equal_nan=True)
            assert np.allclose(ak.fmod(ak_a, ak_b).to_ndarray(), np.fmod(a, b), equal_nan=True)

        npf = np.array([2.23, 3.14, 3.08, 5.7])
        npf2 = np.array([3.14, 2.23, 1.1, 4.1])
        npi = np.array([1, 4, 1, 5])

        akf2 = ak.array(npf2)
        aki = ak.array(npi)

        # opequal
        npf_copy = npf
        akf_copy = ak.array(npf_copy)
        npf_copy %= npf2
        akf_copy %= akf2
        assert np.allclose(akf_copy.to_ndarray(), npf_copy, equal_nan=True)

        npf_copy = npf
        akf_copy = ak.array(npf_copy)
        npf_copy %= npi
        akf_copy %= aki
        assert np.allclose(akf_copy.to_ndarray(), npf_copy, equal_nan=True)

        npf_copy = npf
        akf_copy = ak.array(npf_copy)
        npf_copy %= 2
        akf_copy %= 2
        assert np.allclose(akf_copy.to_ndarray(), npf_copy, equal_nan=True)

        npf_copy = npf
        akf_copy = ak.array(npf_copy)
        npf_copy %= 2.14
        akf_copy %= 2.14
        assert np.allclose(akf_copy.to_ndarray(), npf_copy, equal_nan=True)

    def test_equals(self):
        size = 10
        a1 = ak.arange(size)
        a1_cpy = ak.arange(size)
        a2 = 2 * ak.arange(size)
        a3 = ak.arange(size + 1)

        assert a1.equals(a1_cpy)
        assert not a1.equals(a2)
        assert not a1.equals(a3)
