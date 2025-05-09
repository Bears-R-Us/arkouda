import os
import tempfile

import numpy as np
from base_test import ArkoudaTest
from context import arkouda as ak

from arkouda.pandas import io_util


class SegArrayTest(ArkoudaTest):
    @classmethod
    def setUpClass(cls):
        super(SegArrayTest, cls).setUpClass()
        SegArrayTest.seg_test_base_tmp = "{}/seg_test".format(os.getcwd())
        io_util.get_directory(SegArrayTest.seg_test_base_tmp)

    def test_creation(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.SegArray(segments, akflat)

        self.assertIsInstance(segarr, ak.SegArray)
        self.assertListEqual(segarr.lengths.to_list(), [6, 2, 4])
        self.assertEqual(segarr.__str__(), f"SegArray([\n{a}\n{b}\n{c}\n])".replace(",", ""))
        self.assertEqual(segarr.__getitem__(1).__str__(), str(b).replace(",", ""))
        self.assertEqual(
            segarr.__getitem__(ak.array([1, 2])).__str__(), f"SegArray([\n{b}\n{c}\n])".replace(",", "")
        )
        self.assertEqual(segarr.__eq__(ak.array([1])), NotImplemented)
        self.assertTrue(segarr.__eq__(segarr).all())

        multi_pd = ak.SegArray.from_multi_array(
            [ak.array([10, 11, 12]), ak.array([20, 21, 22]), ak.array([30, 31, 32])]
        )
        self.assertIsInstance(multi_pd, ak.SegArray)
        self.assertEqual(multi_pd.__str__(), "SegArray([\n[10 11 12]\n[20 21 22]\n[30 31 32]\n])")
        with self.assertRaises(TypeError):
            segarr.__getitem__("a")

    def test_creation_empty_segment(self):
        a = [10, 11]
        b = [20, 21, 22]
        c = [30]

        # test empty as first elements
        flat = ak.array(b + c)
        segs = ak.array([0, 0, len(b)])
        segarr = ak.SegArray(segs, flat)
        self.assertIsInstance(segarr, ak.SegArray)
        self.assertListEqual(segarr.lengths.to_list(), [0, 3, 1])

        # test empty as middle element
        flat = ak.array(a + c)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        self.assertIsInstance(segarr, ak.SegArray)
        self.assertListEqual(segarr.lengths.to_list(), [2, 0, 1])

        # test empty as last
        flat = ak.array(a + b + c)
        segs = ak.array([0, len(a), len(a) + len(b), len(a) + len(b) + len(c)])
        segarr = ak.SegArray(segs, flat)
        self.assertIsInstance(segarr, ak.SegArray)
        self.assertListEqual(segarr.lengths.to_list(), [2, 3, 1, 0])

    def test_concat(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b
        akflat = ak.array(flat)
        segments = ak.array([0, len(a)])

        segarr = ak.SegArray(segments, akflat)
        segarr_2 = ak.SegArray(ak.array([0]), ak.array(c))

        concated = ak.SegArray.concat([segarr, segarr_2])
        self.assertEqual(concated.__str__(), f"SegArray([\n{a}\n{b}\n{c}\n])".replace(",", ""))

        # test concat with empty segments
        empty_segs = ak.SegArray(ak.array([0, 2, 2]), ak.array(b + c))
        concated = ak.SegArray.concat([segarr, empty_segs])
        self.assertEqual(concated.__str__(), f"SegArray([\n{a}\n{b}\n{b}\n[]\n{c}\n])".replace(",", ""))

        flat = ak.array(a)
        segs = ak.array([0, len(a)])
        segarr = ak.SegArray(segs, flat)
        a2 = [10]
        b2 = [20]
        flat2 = ak.array(a2 + b2)
        segments2 = ak.array([0, 1])
        segarr2 = ak.SegArray(segments2, flat2)
        concated = ak.SegArray.concat([segarr, segarr2], axis=1)
        self.assertListEqual(concated[0].to_list(), [10, 11, 12, 13, 14, 15, 10])
        self.assertListEqual(concated[1].to_list(), [20])

        with self.assertRaises(ValueError):
            concated = ak.SegArray.concat([segarr, segarr_2], ordered=False)

        with self.assertRaises(ValueError):
            concated = ak.SegArray.concat([])

        self.assertEqual(ak.SegArray.concat([ak.array([1, 2])]), NotImplemented)

        with self.assertRaises(ValueError):
            concated = ak.SegArray.concat([segarr, segarr_2], axis=1)

        with self.assertRaises(ValueError):
            concated = ak.SegArray.concat([segarr, segarr_2], axis=5)

        multi_pd = ak.SegArray.from_multi_array(
            [ak.array([10, 20, 30]), ak.array([11, 21, 31]), ak.array([12, 22, 32])]
        )
        multi_pd2 = ak.SegArray.from_multi_array(
            [ak.array([13, 23, 33]), ak.array([14, 24, 34]), ak.array([15, 25, 35])]
        )
        concated = ak.SegArray.concat([multi_pd, multi_pd2], axis=0)

        test = ak.SegArray.from_multi_array(
            [
                ak.array([10, 20, 30]),
                ak.array([11, 21, 31]),
                ak.array([12, 22, 32]),
                ak.array([13, 23, 33]),
                ak.array([14, 24, 34]),
                ak.array([15, 25, 35]),
            ]
        )
        self.assertEqual(concated.size, test.size)
        for i in range(test.size):
            self.assertListEqual(concated[i].to_list(), test[i].to_list())

        concated = ak.SegArray.concat([multi_pd, multi_pd2], axis=1)

        test = ak.SegArray.from_multi_array(
            [
                ak.array([10, 20, 30, 13, 23, 33]),
                ak.array([11, 21, 31, 14, 24, 34]),
                ak.array([12, 22, 32, 15, 25, 35]),
            ]
        )
        self.assertEqual(concated.size, test.size)
        for i in range(test.size):
            self.assertListEqual(concated[i].to_list(), test[i].to_list())

    def test_suffixes(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)

        suffix, origin = segarr.get_suffixes(1)
        self.assertTrue(origin.all())
        self.assertListEqual(suffix[0].to_list(), [15, 21, 33])

        suffix, origin = segarr.get_suffixes(2)
        self.assertListEqual(suffix[0].to_list(), [14, 32])
        self.assertListEqual(suffix[1].to_list(), [15, 33])
        self.assertTrue(origin[0])
        self.assertFalse(origin[1])
        self.assertTrue(origin[2])

        suffix, origin = segarr.get_suffixes(2, proper=False)
        self.assertListEqual(suffix[0].to_list(), [14, 20, 32])
        self.assertListEqual(suffix[1].to_list(), [15, 21, 33])
        self.assertTrue(origin.all())

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        suffix, origin = segarr.get_suffixes(1)
        self.assertListEqual(suffix[0].to_list(), [15, 21])
        self.assertListEqual(origin.to_list(), [True, False, True])

    def test_prefixes(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)
        prefix, origin = segarr.get_prefixes(1)

        self.assertListEqual(prefix[0].to_list(), [10, 20, 30])
        self.assertTrue(origin.all())

        prefix, origin = segarr.get_prefixes(2)
        self.assertListEqual(prefix[0].to_list(), [10, 30])
        self.assertListEqual(prefix[1].to_list(), [11, 31])
        self.assertTrue(origin[0])
        self.assertFalse(origin[1])
        self.assertTrue(origin[2])

        prefix, origin = segarr.get_prefixes(2, proper=False)
        self.assertListEqual(prefix[0].to_list(), [10, 20, 30])
        self.assertListEqual(prefix[1].to_list(), [11, 21, 31])
        self.assertTrue(origin.all())

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        prefix, origin = segarr.get_prefixes(1)
        self.assertListEqual(prefix[0].to_list(), [10, 20])
        self.assertListEqual(origin.to_list(), [True, False, True])

    def test_ngram(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)
        ngram, origin = segarr.get_ngrams(2)
        self.assertListEqual(ngram[0].to_list(), [10, 11, 12, 13, 14, 20, 30, 31, 32])
        self.assertListEqual(ngram[1].to_list(), [11, 12, 13, 14, 15, 21, 31, 32, 33])
        self.assertListEqual(origin.to_list(), [0, 0, 0, 0, 0, 1, 2, 2, 2])

        ngram, origin = segarr.get_ngrams(5)
        self.assertListEqual(ngram[0].to_list(), [10, 11])
        self.assertListEqual(ngram[1].to_list(), [11, 12])
        self.assertListEqual(ngram[2].to_list(), [12, 13])
        self.assertListEqual(ngram[3].to_list(), [13, 14])
        self.assertListEqual(ngram[4].to_list(), [14, 15])
        self.assertListEqual(origin.to_list(), [0, 0])

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        ngram, origin = segarr.get_ngrams(2)
        self.assertListEqual(ngram[0].to_list(), [10, 11, 12, 13, 14, 20])
        self.assertListEqual(ngram[1].to_list(), [11, 12, 13, 14, 15, 21])
        self.assertListEqual(origin.to_list(), [0, 0, 0, 0, 0, 2])

        with self.assertRaises(ValueError):
            ngram, origin = segarr.get_ngrams(7)

    def test_get_jth(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)

        res, origins = segarr.get_jth(1)
        self.assertListEqual(res.to_list(), [11, 21, 31])
        res, origins = segarr.get_jth(5)
        self.assertListEqual(res.to_list(), [15, 0, 0])
        res, origins = segarr.get_jth(5, compressed=True)
        self.assertListEqual(res.to_list(), [15])

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        res, origin = segarr.get_jth(2)
        self.assertListEqual(res.to_list(), [12, 0, 0])
        self.assertListEqual(origin.to_list(), [True, False, False])

        # verify that segarr.get_jth works with bool vals
        a = [True] * 10
        b = [False] * 10
        segments = ak.array([0, len(a), len(a), len(a), len(a) + len(b)])
        flat = ak.array(a + b)
        sa = ak.SegArray(segments, flat)
        origins_ans = [True, False, False, True, False]

        res, origin = sa.get_jth(1, compressed=True)
        self.assertListEqual(res.to_list(), [True, False])
        self.assertListEqual(origin.to_list(), origins_ans)

        res, origin = sa.get_jth(1)
        self.assertListEqual(res.to_list(), [True, False, False, False, False])
        self.assertListEqual(origin.to_list(), origins_ans)

        res, origin = sa.get_jth(1, default=True)
        self.assertListEqual(res.to_list(), [True, True, True, False, True])
        self.assertListEqual(origin.to_list(), origins_ans)

    def test_set_jth(self):
        """
        No testing for empty segments. Function not designed to add values to segments at
        non-existing indexes
        """
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)

        segarr.set_jth(0, 1, 99)
        self.assertEqual(segarr[0].__str__(), f"{a}".replace(",", "").replace("11", "99"))

        segarr.set_jth(ak.array([0, 1, 2]), 0, 99)
        self.assertEqual(
            segarr[0].__str__(), f"{a}".replace(",", "").replace("10", "99").replace("11", "99")
        )
        self.assertEqual(segarr[1].__str__(), f"{b}".replace(",", "").replace("20", "99"))
        self.assertEqual(segarr[2].__str__(), f"{c}".replace(",", "").replace("30", "99"))

        with self.assertRaises(ValueError):
            segarr.set_jth(1, 4, 999)

    def test_get_length_n(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)

        elem, origin = segarr.get_length_n(2)
        self.assertListEqual(elem[0].to_list(), [20])
        self.assertListEqual(elem[1].to_list(), [21])

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        elem, origin = segarr.get_length_n(2)
        self.assertListEqual(elem[0].to_list(), [20])
        self.assertListEqual(elem[1].to_list(), [21])
        self.assertListEqual(origin.to_list(), [False, False, True])

    def test_append(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)

        self.assertEqual(segarr.append(ak.array([1, 2, 3])), NotImplemented)

        a2 = [0.5, 5.1, 2.3]
        b2 = [1.1, 0.7]
        flat2 = ak.array(a2 + b2)
        segments2 = ak.array([0, len(a2)])
        float_segarr = ak.SegArray(segments2, flat2)

        with self.assertRaises(TypeError):
            segarr.append(float_segarr)

        a2 = [1, 2, 3, 4]
        b2 = [22, 23]
        flat2 = ak.array(a2 + b2)
        segments2 = ak.array([0, len(a2)])
        segarr2 = ak.SegArray(segments2, flat2)

        appended = segarr.append(segarr2)
        self.assertEqual(appended.segments.size, 5)
        self.assertListEqual(appended[3].to_list(), [1, 2, 3, 4])
        self.assertListEqual(appended[4].to_list(), [22, 23])

        a2 = [1, 2]
        b2 = [3]
        segments2 = ak.array([0, len(a2)])
        segarr2 = ak.SegArray(segments2, ak.array(a2 + b2))

        with self.assertRaises(ValueError):
            appended = segarr.append(segarr2, axis=1)

        a = [1, 2]
        b = [3, 4]
        flat = a + b
        akflat = ak.array(flat)
        segments = ak.array([0, len(a)])
        segarr = ak.SegArray(segments, akflat)
        a2 = [10]
        b2 = [20]
        flat2 = ak.array(a2 + b2)
        segments2 = ak.array([0, 1])
        segarr2 = ak.SegArray(segments2, flat2)
        appended = segarr.append(segarr2, axis=1)

        self.assertListEqual(appended.lengths.to_list(), [3, 3])
        self.assertListEqual(appended[0].to_list(), [1, 2, 10])
        self.assertListEqual(appended[1].to_list(), [3, 4, 20])

        # Test with empty segments
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.SegArray(segs, flat)
        appended = segarr.append(segarr2)
        self.assertEqual(appended.segments.size, 5)
        self.assertListEqual(appended[3].to_list(), [10])
        self.assertListEqual(appended[4].to_list(), [20])

        flat = ak.array(a)
        segs = ak.array([0, len(a)])
        segarr = ak.SegArray(segs, flat)
        concated = segarr.append(segarr2, axis=1)
        self.assertListEqual(concated[0].to_list(), [1, 2, 10])
        self.assertListEqual(concated[1].to_list(), [20])

    def test_single_append(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)
        to_append = ak.array([99, 98, 97])

        appended = segarr.append_single(to_append)
        self.assertListEqual(appended.lengths.to_list(), [7, 3, 5])
        self.assertListEqual(appended[0].to_list(), a + [99])
        self.assertListEqual(appended[1].to_list(), b + [98])
        self.assertListEqual(appended[2].to_list(), c + [97])

        to_append = ak.array([99, 99])
        with self.assertRaises(ValueError):
            appended = segarr.append_single(to_append)

        to_append = ak.array([99.99, 1.1, 2.2])
        with self.assertRaises(TypeError):
            appended = segarr.append_single(to_append)

        to_append = 99
        appended = segarr.append_single(to_append)
        self.assertListEqual(appended.lengths.to_list(), [7, 3, 5])
        self.assertListEqual(appended[0].to_list(), a + [99])
        self.assertListEqual(appended[1].to_list(), b + [99])
        self.assertListEqual(appended[2].to_list(), c + [99])

        appended = segarr.prepend_single(to_append)
        self.assertListEqual(appended.lengths.to_list(), [7, 3, 5])
        self.assertListEqual(appended[0].to_list(), [99] + a)
        self.assertListEqual(appended[1].to_list(), [99] + b)
        self.assertListEqual(appended[2].to_list(), [99] + c)

        # test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.SegArray(segs, flat)
        appended = segarr.append_single(99)
        self.assertListEqual(appended[0].to_list(), a + [99])
        self.assertListEqual(appended[1].to_list(), b + [99])
        self.assertListEqual(appended[2].to_list(), [99])

        appended = segarr.prepend_single(99)
        self.assertListEqual(appended[0].to_list(), [99] + a)
        self.assertListEqual(appended[1].to_list(), [99] + b)
        self.assertListEqual(appended[2].to_list(), [99])

        a = [1, 2, 1, 1, 3, 3, 5, 4, 6, 2]
        b = [10, 11, 11, 12, 13, 10, 4, 6, 1, 12]
        segments = ak.array([0, len(a), len(a), len(a), len(a) + len(b)])
        flat = ak.array(a + b)
        sa = ak.SegArray(segments, flat)

        appended = sa.append_single(99)
        self.assertListEqual(appended[0].to_list(), a + [99])
        self.assertListEqual(appended[1].to_list(), [99])
        self.assertListEqual(appended[2].to_list(), [99])
        self.assertListEqual(appended[3].to_list(), b + [99])
        self.assertListEqual(appended[4].to_list(), [99])

        arange = ak.arange(5, 10)
        appended = sa.append_single(arange)
        self.assertListEqual(appended[0].to_list(), a + [arange[0]])
        self.assertListEqual(appended[1].to_list(), [arange[1]])
        self.assertListEqual(appended[2].to_list(), [arange[2]])
        self.assertListEqual(appended[3].to_list(), b + [arange[3]])
        self.assertListEqual(appended[4].to_list(), [arange[4]])

    def test_remove_repeats(self):
        a = [1, 1, 1, 2, 3]
        b = [10, 11, 11, 12]

        flat = ak.array(a + b)
        segments = ak.array([0, len(a)])

        segarr = ak.SegArray(segments, flat)
        dedup = segarr.remove_repeats()
        self.assertListEqual(dedup.lengths.to_list(), [3, 3])
        self.assertListEqual(dedup[0].to_list(), list(set(a)))
        self.assertListEqual(dedup[1].to_list(), list(set(b)))

        # test with empty segments
        segments = ak.array([0, len(a), len(a), len(a) + len(b)])
        segarr = ak.SegArray(segments, flat)
        dedup = segarr.remove_repeats()
        self.assertListEqual(dedup.lengths.to_list(), [3, 0, 3, 0])
        self.assertListEqual(dedup[0].to_list(), list(set(a)))
        self.assertListEqual(dedup[1].to_list(), [])
        self.assertListEqual(dedup[2].to_list(), list(set(b)))
        self.assertListEqual(dedup[3].to_list(), [])

        segments = ak.array([0, len(a), len(a), len(a), len(a) + len(b)])
        segarr = ak.SegArray(segments, flat)
        dedup = segarr.remove_repeats()
        self.assertListEqual(dedup.lengths.to_list(), [3, 0, 0, 3, 0])
        self.assertListEqual(dedup[0].to_list(), list(set(a)))
        self.assertListEqual(dedup[1].to_list(), [])
        self.assertListEqual(dedup[2].to_list(), [])
        self.assertListEqual(dedup[3].to_list(), list(set(b)))
        self.assertListEqual(dedup[4].to_list(), [])

        # reproducer for #2661
        a = [1, 2, 1, 1, 3, 3, 5, 4, 6, 2]
        a_ans = [1, 2, 1, 3, 5, 4, 6, 2]
        a_mult = [1, 1, 2, 2, 1, 1, 1, 1]
        b = [10, 11, 11, 12, 13, 10, 4, 6, 1, 12]
        b_ans = [10, 11, 12, 13, 10, 4, 6, 1, 12]
        b_mult = [1, 2, 1, 1, 1, 1, 1, 1, 1]
        segments = ak.array([0, 0, len(a), len(a), len(a), len(a) + len(b)])
        flat = ak.array(a + b)
        sa = ak.SegArray(segments, flat)
        no_repeats, multiplicity = sa.remove_repeats(return_multiplicity=True)
        self.assertListEqual(no_repeats.non_empty.to_list(), [False, True, False, False, True, False])
        self.assertListEqual(multiplicity.non_empty.to_list(), [False, True, False, False, True, False])
        self.assertListEqual(no_repeats[1].to_list(), a_ans)
        self.assertListEqual(multiplicity[1].to_list(), a_mult)
        self.assertListEqual(no_repeats[4].to_list(), b_ans)
        self.assertListEqual(multiplicity[4].to_list(), b_mult)

    def test_intersection(self):
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8]
        c = [1, 2, 4]
        d = [8]
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        segarr_2 = ak.SegArray(ak.array([0, len(c)]), ak.array(c + d))

        intx = segarr.intersect(segarr_2)

        self.assertEqual(intx.size, 2)
        self.assertListEqual(intx[0].to_list(), [1, 2, 4])
        self.assertListEqual(intx[1].to_list(), [8])

        # test with empty Segments
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        intx = segarr.intersect(segarr_2)
        self.assertListEqual(intx.lengths.to_list(), [3, 0])
        self.assertListEqual(intx[0].to_list(), [1, 2, 4])
        self.assertListEqual(intx[1].to_list(), [])

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + c))
        segarr_2 = ak.SegArray(ak.array([0, len(d)]), ak.array(d + c))
        intx = segarr.intersect(segarr_2)
        self.assertListEqual(intx.lengths.to_list(), [0, 3])
        self.assertListEqual(intx[0].to_list(), [])
        self.assertListEqual(intx[1].to_list(), [1, 2, 4])

    def test_union(self):
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8]
        c = [1, 2, 4]
        d = [8]

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        segarr_2 = ak.SegArray(ak.array([0, len(c)]), ak.array(c + d))

        un = segarr.union(segarr_2)
        self.assertEqual(un.size, 2)
        self.assertListEqual(un[0].to_list(), [1, 2, 3, 4, 5])
        self.assertListEqual(un[1].to_list(), [6, 7, 8])

        # test with empty segments
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        un = segarr.union(segarr_2)
        self.assertListEqual(un.lengths.to_list(), [5, 1])
        self.assertListEqual(un[0].to_list(), [1, 2, 3, 4, 5])
        self.assertListEqual(un[1].to_list(), [8])

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        segarr_2 = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        un = segarr.union(segarr_2)
        self.assertListEqual(un.lengths.to_list(), [5, 0])
        self.assertListEqual(un[0].to_list(), [1, 2, 3, 4, 5])
        self.assertListEqual(un[1].to_list(), [])

    def test_setdiff(self):
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8]
        c = [1, 2, 4]
        d = [8]

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        segarr_2 = ak.SegArray(ak.array([0, len(c)]), ak.array(c + d))

        diff = segarr.setdiff(segarr_2)
        self.assertEqual(diff.size, 2)
        self.assertListEqual(diff[0].to_list(), [3, 5])
        self.assertListEqual(diff[1].to_list(), [6, 7])

        # test with empty segments
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        diff = segarr.setdiff(segarr_2)
        self.assertListEqual(diff.lengths.to_list(), [2, 0])
        self.assertListEqual(diff[0].to_list(), [3, 5])
        self.assertListEqual(diff[1].to_list(), [])

        segarr = ak.SegArray(ak.array([0, len(a), len(a)]), ak.array(a + a))
        segarr_2 = ak.SegArray(ak.array([0, len(c), len(c + c)]), ak.array(c + c + c))
        diff = segarr_2.setdiff(segarr)
        self.assertListEqual(diff.lengths.to_list(), [0, 3, 0])
        self.assertListEqual(diff[0].to_list(), [])
        self.assertListEqual(diff[1].to_list(), [1, 2, 4])
        self.assertListEqual(diff[2].to_list(), [])

    def test_setxor(self):
        a = [1, 2, 3]
        b = [6, 7, 8]
        c = [1, 2, 4]
        d = [8, 12, 13]

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        segarr_2 = ak.SegArray(ak.array([0, len(c)]), ak.array(c + d))
        xor = segarr.setxor(segarr_2)

        self.assertEqual(xor.size, 2)
        self.assertListEqual(xor[0].to_list(), [3, 4])
        self.assertListEqual(xor[1].to_list(), [6, 7, 12, 13])

        # test with empty segment
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        xor = segarr.setxor(segarr_2)
        self.assertListEqual(xor.lengths.to_list(), [2, 3])
        self.assertListEqual(xor[0].to_list(), [3, 4])
        self.assertListEqual(xor[1].to_list(), [8, 12, 13])

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + a))
        segarr_2 = ak.SegArray(ak.array([0, len(a)]), ak.array(a + c))
        xor = segarr.setxor(segarr_2)
        self.assertListEqual(xor.lengths.to_list(), [0, 2])
        self.assertListEqual(xor[0].to_list(), [])
        self.assertListEqual(xor[1].to_list(), [3, 4])

    def test_segarray_load(self):
        segarr = ak.SegArray(ak.array([0, 9, 14]), ak.arange(20))
        with tempfile.TemporaryDirectory(dir=SegArrayTest.seg_test_base_tmp) as tmp_dirname:
            segarr.to_hdf(f"{tmp_dirname}/seg_test.h5")

            seg_load = ak.SegArray.read_hdf(f"{tmp_dirname}/seg_test*").popitem()[1]
            self.assertTrue(ak.all(segarr == seg_load))

    def test_bigint(self):
        a = [2**80, 2**81]
        b = [2**82, 2**83]
        c = [2**84]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.SegArray(segments, akflat)

        self.assertIsInstance(segarr, ak.SegArray)
        self.assertListEqual(segarr.lengths.to_list(), [2, 2, 1])
        self.assertListEqual(segarr[0].to_list(), a)
        self.assertListEqual(segarr[1].to_list(), b)
        self.assertListEqual(segarr[2].to_list(), c)
        self.assertListEqual(segarr[ak.array([1, 2])].values.to_list(), b + c)
        self.assertEqual(segarr.__eq__(ak.array([1])), NotImplemented)
        self.assertTrue(segarr.__eq__(segarr).all())
        self.assertTrue(segarr._non_empty_count == 3)

    def test_filter(self):
        v = ak.randint(0, 5, 100)
        s = ak.arange(0, 100, 2)
        sa = ak.SegArray(s, v)

        # test filtering single value retain empties
        filter_result = sa.filter(2, discard_empty=False)
        self.assertEqual(sa.size, filter_result.size)
        # ensure 2 does not exist in return values
        self.assertTrue((filter_result.values != 2).all())
        for i in range(sa.size):
            self.assertListEqual(sa[i][(sa[i] != 2)].to_list(), filter_result[i].to_list())

        # test list filter
        filter_result = sa.filter([1, 2], discard_empty=False)
        self.assertEqual(sa.size, filter_result.size)
        # ensure 1 & 2 do not exist in return values
        self.assertTrue((filter_result.values != 1).all())
        self.assertTrue((filter_result.values != 2).all())
        for i in range(sa.size):
            x = ak.in1d(ak.array(sa[i]), ak.array([1, 2]), invert=True)
            v = ak.array(sa[i])[x]
            self.assertListEqual(v.to_list(), filter_result[i].to_list())

        # test pdarray filter
        filter_result = sa.filter(ak.array([1, 2]), discard_empty=False)
        self.assertEqual(sa.size, filter_result.size)
        # ensure 1 & 2 do not exist in return values
        self.assertTrue((filter_result.values != 1).all())
        self.assertTrue((filter_result.values != 2).all())
        for i in range(sa.size):
            x = ak.in1d(ak.array(sa[i]), ak.array([1, 2]), invert=True)
            v = ak.array(sa[i])[x]
            self.assertListEqual(v.to_list(), filter_result[i].to_list())

        # test dropping empty segments
        filter_result = sa.filter(ak.array([1, 2]), discard_empty=True)
        # ensure no empty segments
        self.assertTrue((filter_result.lengths != 0).all())
        # ensure 2 does not exist in return values
        self.assertTrue((filter_result.values != 2).all())
        offset = 0
        for i in range(sa.size):
            x = ak.in1d(ak.array(sa[i]), ak.array([1, 2]), invert=True)
            v = ak.array(sa[i])[x]
            if v.size != 0:
                self.assertListEqual(v.to_list(), filter_result[i - offset].to_list())
            else:
                offset += 1

        # reproducer for issue #2666 verify correct results with empty segs
        a = [1, 2, 1, 1, 3, 3, 5, 4, 6, 2]
        a_ans = [1, 2, 1, 1, 5, 4, 6, 2]
        b = [10, 11, 11, 12, 13, 10, 4, 6, 1, 12]
        segments = ak.array([0, len(a), len(a), len(a), len(a) + len(b)])
        flat = ak.array(a + b)
        sa = ak.SegArray(segments, flat)
        filtered = sa.filter(3)
        self.assertListEqual(filtered.non_empty.to_list(), [True, False, False, True, False])
        self.assertListEqual(filtered[0].to_list(), a_ans)
        self.assertListEqual(filtered[3].to_list(), b)

        no_empty_filtered = sa.filter(3, discard_empty=True)
        self.assertListEqual(no_empty_filtered.non_empty.to_list(), [True, True])
        self.assertListEqual(no_empty_filtered[0].to_list(), a_ans)
        self.assertListEqual(no_empty_filtered[1].to_list(), b)

    def test_equality(self):
        # reproducer for issue #2617
        # verify equality no matter position of empty seg
        for has_empty_seg in [0, 0, 9, 14], [0, 9, 9, 14, 14], [0, 0, 7, 9, 14, 14, 17, 20]:
            sa = ak.SegArray(ak.array(has_empty_seg), ak.arange(-10, 10))
            self.assertTrue((sa == sa).all())

        s1 = ak.SegArray(ak.array([0, 4, 14, 14]), ak.arange(-10, 10))
        s2 = ak.SegArray(ak.array([0, 9, 14, 14]), ak.arange(-10, 10))
        self.assertTrue((s1 == s2).to_list() == [False, False, True, True])

        # test segarrays with empty segments, multiple types, and edge cases
        df = ak.DataFrame(
            {
                "c_1": ak.SegArray(ak.array([0, 0, 9, 14]), ak.arange(-10, 10)),
                "c_2": ak.SegArray(
                    ak.array([0, 5, 10, 10]), ak.arange(2**63, 2**63 + 15, dtype=ak.uint64)
                ),
                "c_3": ak.SegArray(ak.array([0, 0, 5, 10]), ak.randint(0, 1, 15, dtype=ak.bool_)),
                "c_4": ak.SegArray(
                    ak.array([0, 9, 14, 14]),
                    ak.array(
                        [
                            np.nan,
                            np.finfo(np.float64).min,
                            -np.inf,
                            -7.0,
                            -3.14,
                            -0.0,
                            0.0,
                            3.14,
                            7.0,
                            np.finfo(np.float64).max,
                            np.inf,
                            np.nan,
                            np.nan,
                            np.nan,
                        ]
                    ),
                ),
                "c_5": ak.SegArray(
                    ak.array([0, 2, 5, 5]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
                ),
                "c_6": ak.SegArray(
                    ak.array([0, 2, 2, 2]), ak.array(["a", "b", "", "c", "d", "e", "f", "g", "h", "i"])
                ),
                "c_7": ak.SegArray(
                    ak.array([0, 0, 2, 2]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
                ),
                "c_8": ak.SegArray(
                    ak.array([0, 2, 3, 3]), ak.array(["", "'", " ", "test", "", "'", "", " ", ""])
                ),
                "c_9": ak.SegArray(
                    ak.array([0, 5, 5, 8]), ak.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"])
                ),
                "c_10": ak.SegArray(
                    ak.array([0, 5, 8, 8]),
                    ak.array(["abc", "123", "xyz", "l", "m", "n", "o", "p", "arkouda"]),
                ),
            }
        )

        for col in df.columns:
            a = df[col]
            if a.dtype == ak.float64:
                a = a.to_ndarray()
                if isinstance(a[0], np.ndarray):
                    self.assertTrue(all(np.allclose(a1, b1, equal_nan=True) for a1, b1 in zip(a, a)))
                else:
                    self.assertTrue(np.allclose(a, a, equal_nan=True))
            else:
                self.assertTrue((a == a).all())
