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
        "uint64": ak.array(np.arange(0, SIZE, 1, dtype=np.uint64)),
        "float64": ak.linspace(0, 2, SIZE),
        "bool": (ak.arange(0, SIZE, 1) % 2) == 0,
    }
    global ndarrays
    ndarrays = {
        "int64": np.arange(0, SIZE, 1),
        "uint64": np.arange(0, SIZE, 1, dtype=np.uint64),
        "float64": np.linspace(0, 2, SIZE),
        "bool": (np.arange(0, SIZE, 1) % 2) == 0,
    }
    global scalars
    # scalars = {k: v[SIZE//2] for k, v in ndarrays.items()}
    scalars = {"int64": 5, "uint64": np.uint64(5), "float64": 3.14159, "bool": True}
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
    for (expression, res, ex, dt, val) in results["both_implement"]:
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

    def test_invert(self):
        ak_uint = ak.arange(10, dtype=ak.uint64)
        inverted = ~ak_uint
        np_uint_inv = ~np.arange(10, dtype=np.uint)
        self.assertListEqual(np_uint_inv.tolist(), inverted.to_list())

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
            self.assertEqual(ak_uint + akf, np_uint + npf)
            self.assertEqual(akf + ak_uint, npf + np_uint)
            self.assertEqual(ak_float + aku, np_float + npu)
            self.assertEqual(aku + ak_float, npu + np_float)

            self.assertEqual(ak_uint - akf, np_uint - npf)
            self.assertEqual(akf - ak_uint, npf - np_uint)
            self.assertEqual(ak_float - aku, np_float - npu)
            self.assertEqual(aku - ak_float, npu - np_float)

            self.assertEqual(ak_uint * akf, np_uint * npf)
            self.assertEqual(akf * ak_uint, npf * np_uint)
            self.assertEqual(ak_float * aku, np_float * npu)
            self.assertEqual(aku * ak_float, npu * np_float)

            self.assertEqual(ak_uint / akf, np_uint / npf)
            self.assertEqual(akf / ak_uint, npf / np_uint)
            self.assertEqual(ak_float / aku, np_float / npu)
            self.assertEqual(aku / ak_float, npu / np_float)

            self.assertEqual(ak_uint // akf, np_uint // npf)
            self.assertEqual(akf // ak_uint, npf // np_uint)
            self.assertEqual(ak_float // aku, np_float // npu)
            self.assertEqual(aku // ak_float, npu // np_float)

            self.assertEqual(ak_uint**akf, np_uint**npf)
            self.assertEqual(akf**ak_uint, npf**np_uint)
            self.assertEqual(ak_float**aku, np_float**npu)
            self.assertEqual(aku**ak_float, npu**np_float)

    def test_concatenate_type_preservation(self):
        # Test that concatenate preserves special pdarray types (IPv4, Datetime, BitVector, ...)
        from arkouda.util import generic_concat as akuconcat

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
        self.assertListEqual(
            ak.Datetime(pda_one).to_list(),
            ak.concatenate([datetime_one]).to_list(),
        )
        self.assertEqual(type(ak.concatenate([ak.Datetime(ak.array([], dtype=ak.int64))])), ak.Datetime)

        # Timedelta test
        timedelta_one = ak.Timedelta(pda_one)
        timedelta_two = ak.Timedelta(pda_two)
        timedelta_concat = ak.concatenate([timedelta_one, timedelta_two])
        self.assertEqual(type(timedelta_concat), ak.Timedelta)
        self.assertListEqual(ak.Timedelta(pda_concat).to_list(), timedelta_concat.to_list())
        # test single and empty
        self.assertEqual(type(ak.concatenate([timedelta_one])), ak.Timedelta)
        self.assertListEqual(
            ak.Timedelta(pda_one).to_list(),
            ak.concatenate([timedelta_one]).to_list(),
        )
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
        self.assertListEqual(
            ak.BitVector(pda_one).to_list(),
            ak.concatenate([bitvector_one]).to_list(),
        )
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
        self.assertListEqual([i if i % 2 else i**2 for i in range(10)],
                             ak.power(a, 2, a % 2 == 0).to_list())

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
        self.assertListEqual([i if i % 2 else i**.5 for i in range(5)], ak.sqrt(a, a % 2 == 0).to_list())

    def test_uint_operation_equals(self):
        u_arr = ak.arange(10, dtype=ak.uint64)
        i_arr = ak.arange(10)
        f_arr = ak.linspace(1, 5, 10)
        b_arr = i_arr % 2 == 0
        u = np.uint(7)
        i = 7
        f = 3.14
        b = True

        # test uint opequals uint functionality against numpy
        np_arr = np.arange(10, dtype=np.uint)
        u_tmp = u_arr[:]
        u_tmp += u
        np_arr += u
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp += u_tmp
        np_arr += np_arr
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp -= u
        np_arr -= u
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp -= u_tmp
        np_arr -= np_arr
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp *= u
        np_arr *= u
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp *= u_tmp
        np_arr *= np_arr
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp **= u
        np_arr **= u
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp **= u_tmp
        np_arr **= np_arr
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp %= u
        np_arr %= u
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp //= u
        np_arr //= u
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())
        u_tmp //= u_tmp
        np_arr //= np_arr
        self.assertListEqual(u_tmp.to_list(), np_arr.tolist())

        # the only arrays that can be added in place are uint and bool
        # scalars are cast to same type if possible
        for v in [b_arr, u, b, i, f]:
            u_tmp = u_arr[:]
            i_tmp = i_arr[:]
            u_tmp += v
            i_tmp += v
            self.assertListEqual(u_tmp.to_list(), i_tmp.to_list())

        # adding a float or int inplace could have a result which is not a uint
        for e in [i_arr, f_arr]:
            with self.assertRaises(RuntimeError):
                u_arr += e

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
        #    ak.histogram((ak.randint(0, 1, 100, dtype=ak.bool)))
        with self.assertRaises(RuntimeError) as cm:
            ak.concatenate([ak.array([True]), ak.array([True])]).is_sorted()

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
        self.assertEqual("[1.100000e+00 2.300000e+00 5.000000e+00]", ak.array([1.1, 2.3, 5]).__str__())
        self.assertEqual(
            "[0.000000e+00 5.263158e-01 1.052632e+00 ... 8.947368e+00 9.473684e+00 1.000000e+01]",
            ak.linspace(0, 10, 20).__str__(),
        )
        self.assertEqual("[False False False]", ak.isnan(ak.array([1.1, 2.3, 5])).__str__())
        self.assertEqual(
            "[False False False ... False False False]", ak.isnan(ak.linspace(0, 10, 20)).__str__()
        )

        # Test __repr__()
        self.assertEqual("array([1 2 3])", ak.array([1, 2, 3]).__repr__())
        self.assertEqual("array([1 2 3 ... 17 18 19])", ak.arange(1, 20).__repr__())
        self.assertEqual(
            "array([1.1000000000000001 2.2999999999999998 5])", ak.array([1.1, 2.3, 5]).__repr__()
        )
        self.assertEqual(
            "array([0 0.52631578947368418 1.0526315789473684 ... "
            "8.9473684210526319 9.473684210526315 10])",
            ak.linspace(0, 10, 20).__repr__(),
        )
        self.assertEqual("array([False False False])", ak.isnan(ak.array([1.1, 2.3, 5])).__repr__())
        self.assertEqual(
            "array([False False False ... False False False])",
            ak.isnan(ak.linspace(0, 10, 20)).__repr__(),
        )
        ak.client.pdarrayIterThresh = (
            ak.client.pdarrayIterThreshDefVal
        )  # Don't forget to set this back for other tests.


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
