import warnings
from itertools import product

import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

SIZE = 10
verbose = ArkoudaTest.verbose


def run_tests(verbose):
    # ignore numpy warnings like divide by 0
    np.seterr(all="ignore")
    global pdarrays
    pdarrays = {
        "int64": ak.arange(0, SIZE, 1),
        "uint64": ak.array(np.arange(2**64 - SIZE, 2**64, 1, dtype=np.uint64)),
        "float64": ak.linspace(0, 2, SIZE),
        "bool": (ak.arange(0, SIZE, 1) % 2) == 0,
    }
    global ndarrays
    ndarrays = {
        "int64": np.arange(0, SIZE, 1),
        "uint64": np.arange(2**64 - SIZE, 2**64, 1, dtype=np.uint64),
        "float64": np.linspace(0, 2, SIZE),
        "bool": (np.arange(0, SIZE, 1) % 2) == 0,
    }
    global scalars
    scalars = {"int64": 5, "uint64": np.uint64(2**63 + 1), "float64": 3.14159, "bool": True}
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
            evalstr += 'scalars["{}"]'.format(lt)
        else:
            evalstr += '{}["{}"]'.format(("ndarrays", "pdarrays")[isarkouda], lt)
        evalstr += " {} ".format(oper)
        if rs:
            evalstr += 'scalars["{}"]'.format(rt)
        else:
            evalstr += '{}["{}"]'.format(("ndarrays", "pdarrays")[isarkouda], rt)
        # print(evalstr)
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
            expression = "{}({}) {} {}({})".format(
                ltype, ("array", "scalar")[lscalar], op, rtype, ("array", "scalar")[rscalar]
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
                    "Cannot detect return dtype of ak result: {} (np result: {})".format(akres, npres)
                )
                results["both_implement"].append((expression, str(akres), False, True, False))
                continue

            if akrestype != npres.dtype:
                restypes = "{}(np) vs. {}(ak)".format(npres.dtype, akrestype)
                # warnings.warn("dtype mismatch: {} = {}".format(expression, restypes))
                results["both_implement"].append((expression, restypes, False, True, False))
                continue
            try:
                akasnp = akres.to_ndarray()
            except Exception:
                warnings.warn("Could not convert to ndarray: {}".format(akres))
                results["both_implement"].append((expression, str(akres), True, False, False))
                continue
            if not np.allclose(akasnp, npres, equal_nan=True):
                res = "np: {}\nak: {}".format(npres, akasnp)
                # warnings.warn("result mismatch: {} =\n{}".format(expression, res))
                results["both_implement"].append((expression, res, False, False, True))
                continue
            # Finally, both numpy and arkouda agree on result
            results["both_implement"].append((expression, "", False, False, False))

    print("# ops not implemented by numpy or arkouda: {}".format(len(results["neither_implement"])))
    if verbose:
        for expression, err in results["neither_implement"]:
            print(expression)
    print("# ops implemented by numpy but not arkouda: {}".format(len(results["numpy_minus_arkouda"])))
    if verbose:
        for expression, err, flag in results["numpy_minus_arkouda"]:
            print(expression)
    print("# ops implemented by arkouda but not numpy: {}".format(len(results["arkouda_minus_numpy"])))
    if verbose:
        for expression, res, flag in results["arkouda_minus_numpy"]:
            print(expression, " -> ", res)
    nboth = len(results["both_implement"])
    print("# ops implemented by both: {}".format(nboth))
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
    print("  Matching results:         {} / {}".format(matches, nboth))
    print("  Arkouda execution errors: {} / {}".format(len(execerrors), nboth))
    if verbose:
        print("\n".join(map(": ".join, execerrors)))
    print("  Dtype mismatches:         {} / {}".format(len(dtypeerrors), nboth))
    if verbose:
        print("\n".join(map(": ".join, dtypeerrors)))
    print("  Value mismatches:         {} / {}".format(len(valueerrors), nboth))
    if verbose:
        print("\n".join(map(": ".join, valueerrors)))
    return matches == nboth


"""
Encapsulates test cases that invoke the run_tests method.
"""


class OperatorsTest(ArkoudaTest):
    def testPdArrayAddInt(self):
        aArray = ak.ones(100)
        addArray = aArray + 1
        self.assertIsInstance(addArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(2), addArray[0])

        addArray = 1 + aArray
        self.assertIsInstance(addArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(2), addArray[0])

    def testPdArrayAddNumpyInt(self):
        aArray = ak.ones(100)
        addArray = aArray + np.int64(1)
        self.assertIsInstance(addArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(2), addArray[0])

        addArray = np.int64(1) + aArray
        self.assertIsInstance(addArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(2), addArray[0])

    def testPdArraySubtractInt(self):
        aArray = ak.ones(100)
        subArray = aArray - 2
        self.assertIsInstance(subArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(-1), subArray[0])

        subArray = 2 - aArray
        self.assertIsInstance(subArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(1), subArray[0])

    def testPdArraySubtractNumpyInt(self):
        aArray = ak.ones(100)
        subArray = aArray - np.int64(2)
        self.assertIsInstance(subArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(-1), subArray[0])

        subArray = np.int64(2) - aArray
        self.assertIsInstance(subArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(1), subArray[0])

    def testPdArrayMultInt(self):
        aArray = ak.ones(100)
        mArray = aArray * 5
        self.assertIsInstance(mArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), mArray[0])

        mArray = 5 * aArray
        self.assertIsInstance(mArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), mArray[0])

    def testPdArrayMultNumpyInt(self):
        aArray = ak.ones(100)
        mArray = aArray * np.int64(5)
        self.assertIsInstance(mArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), mArray[0])

        mArray = np.int64(5) * aArray
        self.assertIsInstance(mArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), mArray[0])

    def testPdArrayDivideInt(self):
        aArray = ak.ones(100)
        dArray = aArray * 15 / 3
        self.assertIsInstance(dArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), dArray[0])

        dArray = 15 * aArray / 3
        self.assertIsInstance(dArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), dArray[0])

    def testPdArrayDivideNumpyInt(self):
        aArray = ak.ones(100)
        dArray = aArray * np.int64(15) / 3
        self.assertIsInstance(dArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), dArray[0])

        dArray = np.int64(15) * aArray / 3
        self.assertIsInstance(dArray, ak.pdarrayclass.pdarray)
        self.assertEqual(np.float64(5), dArray[0])

    def testPdArrayConcatenation(self):
        onesOne = ak.randint(0, 100, 100)
        onesTwo = ak.randint(0, 100, 100)

        result = ak.concatenate([onesOne, onesTwo])
        self.assertEqual(200, len(result))
        self.assertEqual(np.int64, result.dtype)

    def testConcatenate(self):
        pdaOne = ak.arange(1, 4)
        pdaTwo = ak.arange(4, 7)

        self.assertListEqual(
            [1, 2, 3, 4, 5, 6],
            ak.concatenate([pdaOne, pdaTwo]).to_list(),
        )
        self.assertListEqual(
            [4, 5, 6, 1, 2, 3],
            ak.concatenate([pdaTwo, pdaOne]).to_list(),
        )

        pdaOne = ak.linspace(start=1, stop=3, length=3)
        pdaTwo = ak.linspace(start=4, stop=6, length=3)

        self.assertListEqual(
            [1, 2, 3, 4, 5, 6],
            ak.concatenate([pdaOne, pdaTwo]).to_list(),
        )
        self.assertListEqual(
            [4, 5, 6, 1, 2, 3],
            ak.concatenate([pdaTwo, pdaOne]).to_list(),
        )

        pdaOne = ak.array([True, False, True])
        pdaTwo = ak.array([False, True, True])

        self.assertListEqual(
            [True, False, True, False, True, True],
            ak.concatenate([pdaOne, pdaTwo]).to_list(),
        )

        pdaOne = ak.arange(5, max_bits=3)
        pdaTwo = ak.arange(2**200 - 1, 2**200 + 4)
        concatenated = ak.concatenate([pdaOne, pdaTwo])
        self.assertEqual(concatenated.max_bits, 3)
        self.assertListEqual([0, 1, 2, 3, 4, 7, 0, 1, 2, 3], concatenated.to_list())

    def test_invert(self):
        ak_uint = ak.arange(10, dtype=ak.uint64)
        inverted = ~ak_uint
        np_uint_inv = ~np.arange(10, dtype=np.uint)
        self.assertListEqual(np_uint_inv.tolist(), inverted.to_list())

    def test_bool_bool_addition_binop(self):
        np_x = np.array([True, True, False, False])
        np_y = np.array([True, False, True, False])
        ak_x = ak.array(np_x)
        ak_y = ak.array(np_y)
        # Vector-Vector Case
        self.assertListEqual((np_x + np_y).tolist(), (ak_x + ak_y).to_list())
        # Scalar-Vector Case
        self.assertListEqual((np_x[0] + np_y).tolist(), (ak_x[0] + ak_y).to_list())
        self.assertListEqual((np_x[-1] + np_y).tolist(), (ak_x[-1] + ak_y).to_list())
        # Vector-Scalar Case
        self.assertListEqual((np_x + np_y[0]).tolist(), (ak_x + ak_y[0]).to_list())
        self.assertListEqual((np_x + np_y[-1]).tolist(), (ak_x + ak_y[-1]).to_list())

    def test_bool_bool_addition_opeq(self):
        np_x = np.array([True, True, False, False])
        np_y = np.array([True, False, True, False])
        ak_x = ak.array(np_x)
        ak_y = ak.array(np_y)
        np_x += np_y
        ak_x += ak_y
        # Vector-Vector Case
        self.assertListEqual(np_x.tolist(), ak_x.to_list())
        # Scalar-Vector Case
        # True
        np_true = np_x[0]
        ak_true = ak_x[0]
        np_true += np_y
        ak_true += ak_y
        self.assertListEqual(np_x.tolist(), ak_x.to_list())
        # False
        np_false = np_x[-1]
        ak_false = ak_x[-1]
        np_false += np_y
        ak_false += ak_y
        self.assertListEqual(np_x.tolist(), ak_x.to_list())

    def test_uint_bool_binops(self):
        # Test fix for issue #1932
        # Adding support to binopvv to correctly handle uint and bool types
        ak_uint = ak.arange(10, dtype=ak.uint64)
        ak_bool = ak_uint % 2 == 0
        self.assertListEqual((ak_uint + ak_bool).to_list(), (ak.arange(10) + ak_bool).to_list())

    def test_int_uint_binops(self):
        np_int = np.arange(-5, 5)
        ak_int = ak.array(np_int)

        np_uint = np.arange(2**64 - 10, 2**64, dtype=np.uint64)
        ak_uint = ak.array(np_uint)

        # Vector-Vector Case (Division and Floor Division)
        self.assertTrue(np.allclose((ak_uint / ak_uint).to_ndarray(), np_uint / np_uint, equal_nan=True))
        self.assertTrue(np.allclose((ak_int / ak_uint).to_ndarray(), np_int / np_uint, equal_nan=True))
        self.assertTrue(np.allclose((ak_uint / ak_int).to_ndarray(), np_uint / np_int, equal_nan=True))
        self.assertTrue(
            np.allclose((ak_uint // ak_uint).to_ndarray(), np_uint // np_uint, equal_nan=True)
        )
        self.assertTrue(np.allclose((ak_int // ak_uint).to_ndarray(), np_int // np_uint, equal_nan=True))
        self.assertTrue(np.allclose((ak_uint // ak_int).to_ndarray(), np_uint // np_int, equal_nan=True))

        # Scalar-Vector Case (Division and Floor Division)
        self.assertTrue(
            np.allclose((ak_uint[0] / ak_uint).to_ndarray(), np_uint[0] / np_uint, equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_int[0] / ak_uint).to_ndarray(), np_int[0] / np_uint, equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_uint[0] / ak_int).to_ndarray(), np_uint[0] / np_int, equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_uint[0] // ak_uint).to_ndarray(), np_uint[0] // np_uint, equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_int[0] // ak_uint).to_ndarray(), np_int[0] // np_uint, equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_uint[0] // ak_int).to_ndarray(), np_uint[0] // np_int, equal_nan=True)
        )

        # Vector-Scalar Case (Division and Floor Division)
        self.assertTrue(
            np.allclose((ak_uint / ak_uint[0]).to_ndarray(), np_uint / np_uint[0], equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_int / ak_int[0]).to_ndarray(), np_int / np_int[0], equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_uint / ak_uint[0]).to_ndarray(), np_uint / np_uint[0], equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_uint // ak_uint[0]).to_ndarray(), np_uint // np_uint[0], equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_int // ak_uint[0]).to_ndarray(), np_int // np_uint[0], equal_nan=True)
        )
        self.assertTrue(
            np.allclose((ak_uint // ak_int[0]).to_ndarray(), np_uint // np_int[0], equal_nan=True)
        )

    def test_float_uint_binops(self):
        # Test fix for issue #1620
        ak_uint = ak.array([5], dtype=ak.uint64)
        np_uint = np.array([5], dtype=np.uint64)
        scalar_uint = np.uint64(5)

        ak_float = ak.array([3.01], dtype=ak.float64)
        np_float = np.array([3.01], dtype=np.float_)
        scalar_float = 3.01

        ak_uints = [ak_uint, scalar_uint]
        np_uints = [np_uint, scalar_uint]
        ak_floats = [ak_float, scalar_float]
        np_floats = [np_float, scalar_float]
        for aku, akf, npu, npf in zip(ak_uints, ak_floats, np_uints, np_floats):
            self.assertTrue(np.allclose((ak_uint + akf).to_ndarray(), np_uint + npf, equal_nan=True))
            self.assertTrue(np.allclose((akf + ak_uint).to_ndarray(), npf + np_uint, equal_nan=True))
            self.assertTrue(np.allclose((ak_float + aku).to_ndarray(), np_float + npu, equal_nan=True))
            self.assertTrue(np.allclose((aku + ak_float).to_ndarray(), npu + np_float, equal_nan=True))

            self.assertTrue(np.allclose((ak_uint - akf).to_ndarray(), np_uint - npf, equal_nan=True))
            self.assertTrue(np.allclose((akf - ak_uint).to_ndarray(), npf - np_uint, equal_nan=True))
            self.assertTrue(np.allclose((ak_float - aku).to_ndarray(), np_float - npu, equal_nan=True))
            self.assertTrue(np.allclose((aku - ak_float).to_ndarray(), npu - np_float, equal_nan=True))

            self.assertTrue(np.allclose((ak_uint * akf).to_ndarray(), np_uint * npf, equal_nan=True))
            self.assertTrue(np.allclose((akf * ak_uint).to_ndarray(), npf * np_uint, equal_nan=True))
            self.assertTrue(np.allclose((ak_float * aku).to_ndarray(), np_float * npu, equal_nan=True))
            self.assertTrue(np.allclose((aku * ak_float).to_ndarray(), npu * np_float, equal_nan=True))
            self.assertTrue(np.allclose((ak_uint / akf).to_ndarray(), np_uint / npf, equal_nan=True))
            self.assertTrue(np.allclose((akf / ak_uint).to_ndarray(), npf / np_uint, equal_nan=True))
            self.assertTrue(np.allclose((ak_float / aku).to_ndarray(), np_float / npu, equal_nan=True))
            self.assertTrue(np.allclose((aku / ak_float).to_ndarray(), npu / np_float, equal_nan=True))
            self.assertTrue(np.allclose((ak_uint // akf).to_ndarray(), np_uint // npf, equal_nan=True))
            self.assertTrue(np.allclose((akf // ak_uint).to_ndarray(), npf // np_uint, equal_nan=True))
            self.assertTrue(np.allclose((ak_float // aku).to_ndarray(), np_float // npu, equal_nan=True))
            self.assertTrue(np.allclose((aku // ak_float).to_ndarray(), npu // np_float, equal_nan=True))
            self.assertTrue(np.allclose((ak_uint**akf).to_ndarray(), np_uint**npf, equal_nan=True))
            self.assertTrue(np.allclose((akf**ak_uint).to_ndarray(), npf**np_uint, equal_nan=True))
            self.assertTrue(np.allclose((ak_float**aku).to_ndarray(), np_float**npu, equal_nan=True))
            self.assertTrue(np.allclose((aku**ak_float).to_ndarray(), npu**np_float, equal_nan=True))
            self.assertTrue(np.allclose((ak_float % aku).to_ndarray(), np_float % npu, equal_nan=True))
            self.assertTrue(np.allclose((aku % ak_float).to_ndarray(), npu % np_float, equal_nan=True))

    def test_shift_maxbits_binop(self):
        # This tests for a bug when left shifting by a value >=64 bits for int/uint, Issue #2099
        # Max bit value
        maxbits = 2**63 - 1

        # Value arrays
        ak_uint = ak.array([maxbits, maxbits, maxbits, maxbits], dtype=ak.uint64)
        np_uint = np.array([maxbits, maxbits, maxbits, maxbits], dtype=np.uint64)
        ak_int = ak.array([maxbits, maxbits, maxbits, maxbits], dtype=ak.int64)
        np_int = np.array([maxbits, maxbits, maxbits, maxbits], dtype=np.int64)
        ak_bool = ak.array([True, True, False, False], dtype=ak.bool_)
        np_bool = np.array([True, True, False, False], dtype=bool)

        # Shifting value arrays
        ak_uint_array = ak.array([62, 63, 64, 65], dtype=ak.uint64)
        np_uint_array = np.array([62, 63, 64, 65], dtype=np.uint64)
        ak_int_array = ak.array([62, 63, 64, 65], dtype=ak.int64)
        np_int_array = np.array([62, 63, 64, 65], dtype=np.int64)
        ak_bool_array = ak.array([True, False, True, False], dtype=ak.bool_)
        np_bool_array = np.array([True, False, True, False], dtype=bool)

        # Binopvs case
        for i in range(62, 66):
            # Left shift
            self.assertTrue(np.allclose((ak_uint << i).to_ndarray(), np_uint << i))
            self.assertTrue(np.allclose((ak_int << i).to_ndarray(), np_int << i))
            # Right shift
            self.assertTrue(np.allclose((ak_uint >> i).to_ndarray(), np_uint >> i))
            self.assertTrue(np.allclose((ak_int >> i).to_ndarray(), np_int >> i))

        self.assertListEqual((ak_bool_array << True).to_list(), (np_bool_array << True).tolist())
        self.assertListEqual((ak_bool_array << False).to_list(), (np_bool_array << False).tolist())
        self.assertListEqual((ak_bool_array >> True).to_list(), (np_bool_array >> True).tolist())
        self.assertListEqual((ak_bool_array >> False).to_list(), (np_bool_array >> False).tolist())

        # Binopsv case
        # Left Shift
        self.assertListEqual((maxbits << ak_uint_array).to_list(), (maxbits << np_uint_array).tolist())
        self.assertListEqual((maxbits << ak_int_array).to_list(), (maxbits << np_int_array).tolist())
        self.assertListEqual((True << ak_bool_array).to_list(), (True << np_bool_array).tolist())
        self.assertListEqual((False << ak_bool_array).to_list(), (False << np_bool_array).tolist())

        # Right Shift
        self.assertListEqual((maxbits >> ak_uint_array).to_list(), (maxbits >> np_uint_array).tolist())
        self.assertListEqual((maxbits >> ak_int_array).to_list(), (maxbits >> np_int_array).tolist())
        self.assertListEqual((True >> ak_bool_array).to_list(), (True >> np_bool_array).tolist())
        self.assertListEqual((False >> ak_bool_array).to_list(), (False >> np_bool_array).tolist())

        # Binopvv case, Same type
        # Left Shift
        self.assertListEqual((ak_uint << ak_uint_array).to_list(), (np_uint << np_uint_array).tolist())
        self.assertListEqual((ak_int << ak_int_array).to_list(), (np_int << np_int_array).tolist())
        self.assertListEqual((ak_bool << ak_bool_array).to_list(), (np_bool << np_bool_array).tolist())
        # Right Shift
        self.assertListEqual((ak_uint >> ak_uint_array).to_list(), (np_uint >> np_uint_array).tolist())
        self.assertListEqual((ak_int >> ak_int_array).to_list(), (np_int >> np_int_array).tolist())
        self.assertListEqual((ak_bool >> ak_bool_array).to_list(), (np_bool >> np_bool_array).tolist())

        # Binopvv case, Mixed type
        # Left Shift
        self.assertListEqual((ak_uint << ak_int_array).to_list(), (np_uint << np_uint_array).tolist())
        self.assertListEqual((ak_int << ak_uint_array).to_list(), (np_int << np_int_array).tolist())

        # Right shift
        self.assertListEqual((ak_uint >> ak_int_array).to_list(), (np_uint >> np_uint_array).tolist())
        self.assertListEqual((ak_int >> ak_uint_array).to_list(), (np_int >> np_int_array).tolist())

    def test_shift_bool_int64_binop(self):
        # This tests for a missing implementation of bit shifting booleans and ints, Issue #2945
        np_int = np.arange(5)
        ak_int = ak.array(np_int)
        np_bool = np.array([True, False, True, False, True])
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

    def test_shift_equals_scalar_binops(self):
        vector_pairs = [
            (ak.arange(0, 5, dtype=ak.int64), np.arange(5, dtype=np.int64)),
            (ak.arange(0, 5, dtype=ak.uint64), np.arange(5, dtype=np.uint64)),
        ]
        shift_scalars = [np.int64(1), np.int64(5), np.uint64(1), np.uint64(5), True, False]

        for ak_vector, np_vector in vector_pairs:
            for x in shift_scalars:
                self.assertListEqual(ak_vector.to_list(), np_vector.tolist())

                ak_vector <<= x
                np_vector <<= x
                self.assertListEqual(ak_vector.to_list(), np_vector.tolist())

                ak_vector >>= x
                np_vector >>= x
                self.assertListEqual(ak_vector.to_list(), np_vector.tolist())

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

                self.assertListEqual(ak_vector.to_list(), np_vector.tolist())

                ak_vector <<= v
                np_vector <<= v.to_ndarray()
                self.assertListEqual(ak_vector.to_list(), np_vector.tolist())

                ak_vector >>= v
                np_vector >>= v.to_ndarray()
                self.assertListEqual(ak_vector.to_list(), np_vector.tolist())

    def test_concatenate_type_preservation(self):
        # Test that concatenate preserves special pdarray types (IPv4, Datetime, BitVector, ...)
        from arkouda.numpy.util import generic_concat as akuconcat

        pda_one = ak.arange(1, 4)
        pda_two = ak.arange(4, 7)
        pda_concat = ak.concatenate([pda_one, pda_two])

        # IPv4 test
        ipv4_one = ak.IPv4(pda_one)
        ipv4_two = ak.IPv4(pda_two)
        ipv4_concat = ak.concatenate([ipv4_one, ipv4_two])
        self.assertEqual(type(ipv4_concat), ak.IPv4)
        self.assertListEqual(ak.IPv4(pda_concat).to_list(), ipv4_concat.to_list())
        # test single and empty
        self.assertEqual(type(ak.concatenate([ipv4_one])), ak.IPv4)
        self.assertListEqual(ak.IPv4(pda_one).to_list(), ak.concatenate([ipv4_one]).to_list())
        self.assertEqual(type(ak.concatenate([ak.IPv4(ak.array([], dtype=ak.int64))])), ak.IPv4)

        # Datetime test
        datetime_one = ak.Datetime(pda_one)
        datetime_two = ak.Datetime(pda_two)
        datetime_concat = ak.concatenate([datetime_one, datetime_two])
        self.assertEqual(type(datetime_concat), ak.Datetime)
        self.assertListEqual(ak.Datetime(pda_concat).to_list(), datetime_concat.to_list())
        # test single and empty
        self.assertEqual(type(ak.concatenate([datetime_one])), ak.Datetime)
        self.assertListEqual(ak.Datetime(pda_one).to_list(), ak.concatenate([datetime_one]).to_list())
        self.assertEqual(type(ak.concatenate([ak.Datetime(ak.array([], dtype=ak.int64))])), ak.Datetime)

        # Timedelta test
        timedelta_one = ak.Timedelta(pda_one)
        timedelta_two = ak.Timedelta(pda_two)
        timedelta_concat = ak.concatenate([timedelta_one, timedelta_two])
        self.assertEqual(type(timedelta_concat), ak.Timedelta)
        self.assertListEqual(ak.Timedelta(pda_concat).to_list(), timedelta_concat.to_list())
        # test single and empty
        self.assertEqual(type(ak.concatenate([timedelta_one])), ak.Timedelta)
        self.assertListEqual(ak.Timedelta(pda_one).to_list(), ak.concatenate([timedelta_one]).to_list())
        self.assertEqual(
            type(ak.concatenate([ak.Timedelta(ak.array([], dtype=ak.int64))])), ak.Timedelta
        )

        # BitVector test
        bitvector_one = ak.BitVector(pda_one)
        bitvector_two = ak.BitVector(pda_two)
        bitvector_concat = ak.concatenate([bitvector_one, bitvector_two])
        self.assertEqual(type(bitvector_concat), ak.BitVector)
        self.assertListEqual(ak.BitVector(pda_concat).to_list(), bitvector_concat.to_list())
        # test single and empty
        self.assertEqual(type(ak.concatenate([bitvector_one])), ak.BitVector)
        self.assertListEqual(ak.BitVector(pda_one).to_list(), ak.concatenate([bitvector_one]).to_list())
        self.assertEqual(
            type(ak.concatenate([ak.BitVector(ak.array([], dtype=ak.int64))])), ak.BitVector
        )

        # Test failure with mixed types
        with self.assertRaises(TypeError):
            ak.concatenate(datetime_one, bitvector_two)

        # verify ak.util.concatenate still works
        ipv4_akuconcat = akuconcat([ipv4_one, ipv4_two])
        self.assertEqual(type(ipv4_akuconcat), ak.IPv4)
        self.assertListEqual(ak.IPv4(pda_concat).to_list(), ipv4_akuconcat.to_list())

        datetime_akuconcat = akuconcat([datetime_one, datetime_two])
        self.assertEqual(type(datetime_akuconcat), ak.Datetime)
        self.assertListEqual(ak.Datetime(pda_concat).to_list(), datetime_akuconcat.to_list())

        timedelta_akuconcat = akuconcat([timedelta_one, timedelta_two])
        self.assertEqual(type(timedelta_akuconcat), ak.Timedelta)
        self.assertListEqual(ak.Timedelta(pda_concat).to_list(), timedelta_akuconcat.to_list())

        bitvector_akuconcat = akuconcat([bitvector_one, bitvector_two])
        self.assertEqual(type(bitvector_akuconcat), ak.BitVector)
        self.assertListEqual(ak.BitVector(pda_concat).to_list(), bitvector_akuconcat.to_list())

    def test_floor_div_edge_cases(self):
        scalar_edge_cases = [-np.inf, -7.0, -0.0, np.nan, 0.0, 7.0, np.inf]
        np_edge_cases = np.array(scalar_edge_cases)
        ak_edge_cases = ak.array(np_edge_cases)

        for s in scalar_edge_cases:
            # test vector // scalar
            self.assertTrue(
                np.allclose((ak_edge_cases // s).to_ndarray(), np_edge_cases // s, equal_nan=True)
            )

            # test scalar // vector
            self.assertTrue(
                np.allclose((s // ak_edge_cases).to_ndarray(), s // np_edge_cases, equal_nan=True)
            )

            # test both vector // vector
            n_vect = np.full(len(scalar_edge_cases), s)
            a_vect = ak.array(n_vect)
            self.assertTrue(
                np.allclose(
                    (ak_edge_cases // a_vect).to_ndarray(), np_edge_cases // n_vect, equal_nan=True
                )
            )
            self.assertTrue(
                np.allclose(
                    (a_vect // ak_edge_cases).to_ndarray(), n_vect // np_edge_cases, equal_nan=True
                )
            )

    def test_pda_power(self):
        n = np.array([10, 5, 2])
        a = ak.array(n)

        p = ak.power(a, 2)
        ex = np.power(n, 2)
        self.assertListEqual(p.to_list(), ex.tolist())

        p = ak.power(a, ak.array([2, 3, 4]))
        ex = np.power(n, [2, 3, 4])
        self.assertListEqual(p.to_list(), ex.tolist())

        # Test a singleton with and without a Boolean argument
        a = ak.array([7])
        self.assertListEqual(ak.power(a, 3, True).to_list(), ak.power(a, 3).to_list())
        self.assertListEqual(ak.power(a, 3, False).to_list(), a.to_list())

        # Test an with and without a Boolean argument, all the same
        a = ak.array([0, 0.0, 1, 7.0, 10])
        self.assertListEqual(ak.power(a, 3, ak.ones(5, bool)).to_list(), ak.power(a, 3).to_list())
        self.assertListEqual(ak.power(a, 3, ak.zeros(5, bool)).to_list(), a.to_list())

        # Test a singleton with a mixed Boolean argument
        a = ak.arange(10)
        self.assertListEqual(
            [i if i % 2 else i**2 for i in range(10)], ak.power(a, 2, a % 2 == 0).to_list()
        )

        # Test invalid input, negative
        n = np.array([-1.0, -3.0])
        a = ak.array(n)

        p = ak.power(a, 0.5)
        ex = np.power(n, 0.5)
        self.assertTrue(np.allclose(p.to_ndarray(), ex, equal_nan=True))

        # Test edge case input, inf
        n = np.array([np.inf, -np.inf])
        a = ak.array([np.inf, -np.inf])
        self.assertListEqual(np.power(n, 2).tolist(), ak.power(a, 2).to_list())

    def test_pda_sqrt(self):
        # Base cases and edge cases
        # Most cases are taken care of in the test_pda_power tests
        n = np.array([4, 16.0, -1, 0, np.inf])
        a = ak.array(n)
        self.assertTrue(np.allclose(ak.sqrt(a).to_ndarray(), np.sqrt(n), equal_nan=True))

        # Test with a mixed Boolean array
        a = ak.arange(5)
        self.assertListEqual(
            [i if i % 2 else i**0.5 for i in range(5)], ak.sqrt(a, a % 2 == 0).to_list()
        )

    def test_uint_and_bigint_operation_equals(self):
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
        i_arr = ak.arange(10)
        f_arr = ak.linspace(1, 5, 10)
        b_arr = i_arr % 2 == 0
        u = np.uint(7)
        i = 7
        f = 3.14
        b = True

        # test uint opequals uint functionality against numpy
        u_arr += u
        bi_arr += u
        np_arr += u
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())
        u_arr += u_arr
        bi_arr += bi_arr
        np_arr += np_arr
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())
        u_arr -= u
        bi_arr -= u
        np_arr -= u
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())
        u_arr -= u_arr
        bi_arr -= bi_arr
        np_arr -= np_arr
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())

        # redeclare minus self zeroed out
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
        u_arr *= u
        bi_arr *= u
        np_arr *= u
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())
        u_arr *= u_arr
        bi_arr *= bi_arr
        np_arr *= np_arr
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())
        u_arr **= u
        bi_arr **= u
        np_arr **= u
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())
        u_arr **= u_arr
        bi_arr **= bi_arr
        np_arr **= np_arr
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())
        u_arr %= u
        bi_arr %= u
        np_arr %= u
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())
        u_arr //= u
        bi_arr //= u
        np_arr //= u
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())

        # redeclare divide zeroed out
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
        u_arr //= u_arr
        bi_arr //= bi_arr
        np_arr //= np_arr
        self.assertListEqual(u_arr.to_list(), np_arr.tolist())
        self.assertListEqual(bi_arr.to_list(), np_arr.tolist())

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
            self.assertListEqual(u_tmp.to_list(), i_tmp.to_list())
            self.assertListEqual(u_tmp.to_list(), bi_tmp.to_list())

        # adding a float or int inplace could have a result which is not a uint
        for e in [i_arr, f_arr]:
            with self.assertRaises(RuntimeError):
                u_arr += e

        with self.assertRaises(RuntimeError):
            bi_arr += f_arr

        # verify other types can have uint applied to them
        f_arr += u_arr
        f_arr += u

    def testAllOperators(self):
        run_tests(verbose)

    def testErrorHandling(self):
        # Test NotImplmentedError that prevents pddarray iteration
        with self.assertRaises(NotImplementedError):
            iter(ak.ones(100))

        # Test NotImplmentedError that prevents Strings iteration
        with self.assertRaises(NotImplementedError):
            iter(ak.array(["String {}".format(i) for i in range(0, 10)]))

        # Test ak,histogram against unsupported dtype
        # with self.assertRaises(ValueError) as cm:
        #    ak.histogram((ak.randint(0, 1, 100, dtype=ak.bool_)))

        with self.assertRaises(TypeError):
            ak.ones(100).any([0])

        with self.assertRaises(AttributeError):
            ak.unique(list(range(0, 10)))

        with self.assertRaises(ValueError):
            ak.concatenate([ak.ones(100), ak.array([True])])

    def test_str_repr(self):
        """
        Test 3 different types: int, float, bool with lengths under/over threshold
        Do this for both __str__() and __repr__()
        """
        ak.client.pdarrayIterThresh = 5
        # Test __str__()
        self.assertEqual("[1 2 3]", ak.array([1, 2, 3]).__str__())
        self.assertEqual("[1 2 3 ... 17 18 19]", ak.arange(1, 20).__str__())
        # Printing floating point values changed precision in Chapel 1.32
        answers = ["[1.100000e+00 2.300000e+00 5.000000e+00]", "[1.1 2.3 5.0]"]
        self.assertTrue(ak.array([1.1, 2.3, 5]).__str__() in answers)
        answers = [
            "[0.000000e+00 5.263158e-01 1.052632e+00 ... 8.947368e+00 9.473684e+00 1.000000e+01]",
            "[0.0 0.526316 1.05263 ... 8.94737 9.47368 10.0]",
        ]
        self.assertTrue(ak.linspace(0, 10, 20).__str__() in answers)
        self.assertEqual("[False False False]", ak.isnan(ak.array([1.1, 2.3, 5])).__str__())
        self.assertEqual(
            "[False False False ... False False False]", ak.isnan(ak.linspace(0, 10, 20)).__str__()
        )

        # Test __repr__()
        self.assertEqual("array([1 2 3])", ak.array([1, 2, 3]).__repr__())
        self.assertEqual("array([1 2 3 ... 17 18 19])", ak.arange(1, 20).__repr__())
        answers = [
            "array([1.1000000000000001 2.2999999999999998 5])",
            "array([1.1 2.3 5])",
            "array([1.1000000000000001 2.2999999999999998 5.00000000000000000])",
        ]
        self.assertTrue(ak.array([1.1, 2.3, 5]).__repr__() in answers)

        answers = [
            "array([0 0.52631578947368418 1.0526315789473684 ... "
            "8.9473684210526319 9.473684210526315 10])",
            "array([0 0.5 1.1 ... 8.9 9.5 10])",
            "array([0.00000000000000000 0.52631578947368418 1.0526315789473684 "
            "... 8.9473684210526319 9.473684210526315 10.00000000000000000])",
        ]
        self.assertTrue(ak.linspace(0, 10, 20).__repr__() in answers)
        self.assertEqual("array([False False False])", ak.isnan(ak.array([1.1, 2.3, 5])).__repr__())
        self.assertEqual(
            "array([False False False ... False False False])",
            ak.isnan(ak.linspace(0, 10, 20)).__repr__(),
        )
        ak.client.pdarrayIterThresh = (
            ak.client.pdarrayIterThreshDefVal
        )  # Don't forget to set this back for other tests.

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
        self.assertListEqual((u & u_range).to_list(), (bi & bi_range).to_list())
        self.assertListEqual(
            [(bi[i] & bi_scalar) % mod_by for i in range(bi.size)], (bi & bi_scalar).to_list()
        )
        self.assertListEqual(
            [(bi_scalar & bi[i]) % mod_by for i in range(bi.size)], (bi_scalar & bi).to_list()
        )

        self.assertListEqual((u | u_range).to_list(), (bi | bi_range).to_list())
        self.assertListEqual(
            [(bi[i] | bi_scalar) % mod_by for i in range(bi.size)], (bi | bi_scalar).to_list()
        )
        self.assertListEqual(
            [(bi_scalar | bi[i]) % mod_by for i in range(bi.size)], (bi_scalar | bi).to_list()
        )

        self.assertListEqual((u ^ u_range).to_list(), (bi ^ bi_range).to_list())
        self.assertListEqual(
            [(bi[i] ^ bi_scalar) % mod_by for i in range(bi.size)], (bi ^ bi_scalar).to_list()
        )
        self.assertListEqual(
            [(bi_scalar ^ bi[i]) % mod_by for i in range(bi.size)], (bi_scalar ^ bi).to_list()
        )

        # bit shifts: left side must be bigint, right side must be int/uint
        ans = u << u_range
        self.assertListEqual(ans.to_list(), (bi << u_range).to_list())
        self.assertListEqual(ans.to_list(), (bi << i_range).to_list())

        ans = u >> u_range
        self.assertListEqual(ans.to_list(), (bi >> u_range).to_list())
        self.assertListEqual(ans.to_list(), (bi >> i_range).to_list())

        ans = u.rotl(u_range)
        self.assertListEqual(ans.to_list(), bi.rotl(u_range).to_list())
        self.assertListEqual(ans.to_list(), bi.rotl(i_range).to_list())
        ans = u.rotr(u_range)
        self.assertListEqual(ans.to_list(), bi.rotr(u_range).to_list())
        self.assertListEqual(ans.to_list(), bi.rotr(i_range).to_list())

        # ops where left side has to bigint
        ans = u // u_range
        self.assertListEqual(ans.to_list(), (bi // bi_range).to_list())
        self.assertListEqual(ans.to_list(), (bi // u_range).to_list())
        self.assertListEqual(ans.to_list(), (bi // i_range).to_list())

        ans = u % u_range
        self.assertListEqual(ans.to_list(), (bi % bi_range).to_list())
        self.assertListEqual(ans.to_list(), (bi % u_range).to_list())
        self.assertListEqual(ans.to_list(), (bi % i_range).to_list())

        ans = u**u_range
        self.assertListEqual(ans.to_list(), (bi**bi_range).to_list())
        self.assertListEqual(ans.to_list(), (bi**u_range).to_list())
        self.assertListEqual(ans.to_list(), (bi**i_range).to_list())

        # ops where either side can any of bigint, int, uint, bool
        ans = u + u_range
        self.assertListEqual(ans.to_list(), (bi + bi_range).to_list())
        self.assertListEqual(ans.to_list(), (bi + u_range).to_list())
        self.assertListEqual(ans.to_list(), (bi + i_range).to_list())
        self.assertListEqual(ans.to_list(), (i_range + bi).to_list())
        self.assertListEqual(ans.to_list(), (u_range + bi).to_list())
        ans = u + b
        self.assertListEqual(ans.to_list(), (bi + b).to_list())
        self.assertListEqual(ans.to_list(), (b + bi).to_list())
        for s in [i_scalar, u_scalar, bi_scalar]:
            self.assertListEqual([(bi[i] + s) % mod_by for i in range(bi.size)], (bi + s).to_list())
            self.assertListEqual([(s + bi[i]) % mod_by for i in range(bi.size)], (s + bi).to_list())

        ans = u - u_range
        self.assertListEqual(ans.to_list(), (bi - bi_range).to_list())
        self.assertListEqual(ans.to_list(), (bi - u_range).to_list())
        self.assertListEqual(ans.to_list(), (bi - i_range).to_list())
        self.assertListEqual((u - b).to_list(), (bi - b).to_list())
        self.assertListEqual((b - u).to_list(), (b - bi).to_list())

        for s in [i_scalar, u_scalar, bi_scalar]:
            self.assertListEqual([(bi[i] - s) % mod_by for i in range(bi.size)], (bi - s).to_list())
            self.assertListEqual([(s - bi[i]) % mod_by for i in range(bi.size)], (s - bi).to_list())

        self.assertListEqual((bi - neg_range).to_list(), (bi + u_range).to_list())

        ans = u * u_range
        self.assertListEqual(ans.to_list(), (bi * bi_range).to_list())
        self.assertListEqual(ans.to_list(), (bi * u_range).to_list())
        self.assertListEqual(ans.to_list(), (bi * i_range).to_list())
        ans = u * b
        self.assertListEqual(ans.to_list(), (bi * b).to_list())
        self.assertListEqual(ans.to_list(), (b * bi).to_list())

        for s in [i_scalar, u_scalar, bi_scalar]:
            self.assertListEqual([(bi[i] * s) % mod_by for i in range(bi.size)], (bi * s).to_list())
            self.assertListEqual([(s * bi[i]) % mod_by for i in range(bi.size)], (s * bi).to_list())

    def test_bigint_rotate(self):
        # see issue #2214
        # verify bigint pdarray correctly rotate when shift_amount exceeds max_bits
        # in this test we are rotating 10 with max_bits=4, so even rotations will equal 10
        # and odd rotations will equal 5. We test rotations up to 10 (which is > 4)

        # rotate by scalar
        for i in range(10):
            self.assertEqual(
                ak.array([10], dtype=ak.bigint, max_bits=4).rotl(i), 10 if i % 2 == 0 else 5
            )
            self.assertEqual(
                ak.array([10], dtype=ak.bigint, max_bits=4).rotr(i), 10 if i % 2 == 0 else 5
            )

        # rotate by array
        left_rot = ak.bigint_from_uint_arrays([ak.full(10, 10, ak.uint64)], max_bits=4).rotl(
            ak.arange(10)
        )
        right_rot = ak.bigint_from_uint_arrays([ak.full(10, 10, ak.uint64)], max_bits=4).rotr(
            ak.arange(10)
        )
        ans = [10 if i % 2 == 0 else 5 for i in range(10)]
        self.assertListEqual(left_rot.to_list(), ans)
        self.assertListEqual(right_rot.to_list(), ans)

    def test_float_mods(self):
        edge_cases = [np.nan, -np.inf, -7.0, -3.14, -0.0, 0.0, 3.14, 7.0, np.inf, np.nan]

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

        args = [rand_edge_cases1, rand_edge_cases2, float_arr, int_arr, uint_arr, i_scal, u_scal]
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
            self.assertTrue(np.allclose(ak.mod(ak_a, ak_b).to_ndarray(), np.mod(a, b), equal_nan=True))
            self.assertTrue(np.allclose(ak.fmod(ak_a, ak_b).to_ndarray(), np.fmod(a, b), equal_nan=True))

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
        self.assertTrue(np.allclose(akf_copy.to_ndarray(), npf_copy, equal_nan=True))

        npf_copy = npf
        akf_copy = ak.array(npf_copy)
        npf_copy %= npi
        akf_copy %= aki
        self.assertTrue(np.allclose(akf_copy.to_ndarray(), npf_copy, equal_nan=True))

        npf_copy = npf
        akf_copy = ak.array(npf_copy)
        npf_copy %= 2
        akf_copy %= 2
        self.assertTrue(np.allclose(akf_copy.to_ndarray(), npf_copy, equal_nan=True))

        npf_copy = npf
        akf_copy = ak.array(npf_copy)
        npf_copy %= 2.14
        akf_copy %= 2.14
        self.assertTrue(np.allclose(akf_copy.to_ndarray(), npf_copy, equal_nan=True))

    def test_equals(self):
        size = 10
        a1 = ak.arange(size)
        a1_cpy = ak.arange(size)
        a2 = 2 * ak.arange(size)
        a3 = ak.arange(size + 1)

        self.assertTrue(a1.equals(a1_cpy))
        self.assertFalse(a1.equals(a2))
        self.assertFalse(a1.equals(a3))


if __name__ == "__main__":
    """
    Enables invocation of operator tests outside of pytest test harness
    """
    import sys

    if len(sys.argv) not in (3, 4):
        print(f"Usage: {sys.argv[0]} <server_name> <port> [<verbose>=(0|1)]")
    verbose = False
    if len(sys.argv) == 4 and sys.argv[3] == "1":
        verbose = True
    ak.connect(server=sys.argv[1], port=int(sys.argv[2]))
    success = run_tests(verbose)
    ak.disconnect()
    sys.exit((1, 0)[success])
