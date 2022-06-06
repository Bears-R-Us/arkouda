from base_test import ArkoudaTest
from context import arkouda as ak


class SegArrayTest(ArkoudaTest):
    def test_creation(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.SegArray(segments, akflat)

        self.assertIsInstance(segarr, ak.SegArray)
        self.assertTrue((segarr._get_lengths() == ak.array([6, 2, 4])).all())
        self.assertEqual(segarr.__str__(), f"SegArray([\n{a}\n{b}\n{c}\n])".replace(",", ""))
        self.assertEqual(segarr.__getitem__(1).__str__(), str(b).replace(",", ""))
        self.assertEqual(
            segarr.__getitem__(ak.array([1, 2])).__str__(), f"SegArray([\n{b}\n{c}\n])".replace(",", "")
        )
        self.assertEqual(segarr.__eq__(ak.array([1])), NotImplemented)
        self.assertTrue(segarr.__eq__(segarr).all())

        multi_pd = ak.SegArray.from_multi_array(
            [ak.array([10, 20, 30]), ak.array([11, 21, 31]), ak.array([12, 22, 32])]
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
        self.assertListEqual(segarr.lengths.to_ndarray().tolist(), [0, 3, 1])

        # test empty as middle element
        flat = ak.array(a + c)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        self.assertIsInstance(segarr, ak.SegArray)
        self.assertListEqual(segarr.lengths.to_ndarray().tolist(), [2, 0, 1])

        # test empty as last
        flat = ak.array(a + b + c)
        segs = ak.array([0, len(a), len(a) + len(b), len(a) + len(b) + len(c)])
        segarr = ak.SegArray(segs, flat)
        self.assertIsInstance(segarr, ak.SegArray)
        self.assertListEqual(segarr.lengths.to_ndarray().tolist(), [2, 3, 1, 0])

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
        self.assertListEqual(concated[0].tolist(), [10, 11, 12, 13, 14, 15, 10])
        self.assertListEqual(concated[1].tolist(), [20])

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
        concated = ak.SegArray.concat([multi_pd, multi_pd2], axis=1)

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

        self.assertEqual(concated.__str__(), test.__str__())

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
        self.assertTrue((suffix[0] == ak.array([15, 21, 33])).all())

        suffix, origin = segarr.get_suffixes(2)
        self.assertTrue((suffix[0] == ak.array([14, 32])).all())
        self.assertTrue((suffix[1] == ak.array([15, 33])).all())
        self.assertTrue(origin[0])
        self.assertFalse(origin[1])
        self.assertTrue(origin[2])

        suffix, origin = segarr.get_suffixes(2, proper=False)
        self.assertTrue((suffix[0] == ak.array([14, 20, 32])).all())
        self.assertTrue((suffix[1] == ak.array([15, 21, 33])).all())
        self.assertTrue(origin.all())

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        suffix, origin = segarr.get_suffixes(1)
        self.assertListEqual(suffix[0].to_ndarray().tolist(), [15, 21])
        self.assertListEqual(origin.to_ndarray().tolist(), [True, False, True])

    def test_prefixes(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)
        prefix, origin = segarr.get_prefixes(1)

        self.assertTrue((prefix[0] == ak.array([10, 20, 30])).all())
        self.assertTrue(origin.all())

        prefix, origin = segarr.get_prefixes(2)
        self.assertTrue((prefix[0] == ak.array([10, 30])).all())
        self.assertTrue((prefix[1] == ak.array([11, 31])).all())
        self.assertTrue(origin[0])
        self.assertFalse(origin[1])
        self.assertTrue(origin[2])

        prefix, origin = segarr.get_prefixes(2, proper=False)
        self.assertTrue((prefix[0] == ak.array([10, 20, 30])).all())
        self.assertTrue((prefix[1] == ak.array([11, 21, 31])).all())
        self.assertTrue(origin.all())

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        prefix, origin = segarr.get_prefixes(1)
        self.assertListEqual(prefix[0].to_ndarray().tolist(), [10, 20])
        self.assertListEqual(origin.to_ndarray().tolist(), [True, False, True])

    def test_ngram(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)
        ngram, origin = segarr.get_ngrams(2)
        self.assertTrue((ngram[0] == ak.array([10, 11, 12, 13, 14, 20, 30, 31, 32])).all())
        self.assertTrue((ngram[1] == ak.array([11, 12, 13, 14, 15, 21, 31, 32, 33])).all())
        self.assertTrue((origin == ak.array([0, 0, 0, 0, 0, 1, 2, 2, 2])).all())

        ngram, origin = segarr.get_ngrams(5)
        self.assertTrue((ngram[0] == ak.array([10, 11])).all())
        self.assertTrue((ngram[1] == ak.array([11, 12])).all())
        self.assertTrue((ngram[2] == ak.array([12, 13])).all())
        self.assertTrue((ngram[3] == ak.array([13, 14])).all())
        self.assertTrue((ngram[4] == ak.array([14, 15])).all())
        self.assertTrue((origin == ak.array([0, 0])).all())

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        ngram, origin = segarr.get_ngrams(2)
        self.assertListEqual(ngram[0].to_ndarray().tolist(), [10, 11, 12, 13, 14, 20])
        self.assertListEqual(ngram[1].to_ndarray().tolist(), [11, 12, 13, 14, 15, 21])
        self.assertListEqual(origin.to_ndarray().tolist(), [0, 0, 0, 0, 0, 2])

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
        self.assertTrue((res == ak.array([11, 21, 31])).all())
        res, origins = segarr.get_jth(5)
        self.assertTrue((res == ak.array([15, 0, 0])).all())
        res, origins = segarr.get_jth(5, compressed=True)
        self.assertTrue((res == ak.array([15])).all())

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        res, origin = segarr.get_jth(2)
        self.assertListEqual(res.to_ndarray().tolist(), [12, 0, 0])
        self.assertListEqual(origin.to_ndarray().tolist(), [True, False, False])

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

        s = ak.array(["abc", "123"])
        s_segments = ak.array([0])
        segarr = ak.SegArray(s_segments, s)

        with self.assertRaises(TypeError):
            segarr.set_jth(0, 0, "test")

    def test_get_length_n(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])

        segarr = ak.SegArray(segments, akflat)

        elem, origin = segarr.get_length_n(2)
        self.assertTrue((elem[0] == ak.array([20])).all())
        self.assertTrue((elem[1] == ak.array([21])).all())

        # Test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a)])
        segarr = ak.SegArray(segs, flat)
        elem, origin = segarr.get_length_n(2)
        self.assertListEqual(elem[0].to_ndarray().tolist(), [20])
        self.assertListEqual(elem[1].to_ndarray().tolist(), [21])
        self.assertListEqual(origin.to_ndarray().tolist(), [False, False, True])

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
        self.assertListEqual(appended[3].tolist(), [1, 2, 3, 4])
        self.assertListEqual(appended[4].tolist(), [22, 23])

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

        self.assertListEqual(appended.lengths.to_ndarray().tolist(), [3, 3])
        self.assertListEqual(appended[0].tolist(), [1, 2, 10])
        self.assertListEqual(appended[1].tolist(), [3, 4, 20])

        # Test with empty segments
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.SegArray(segs, flat)
        appended = segarr.append(segarr2)
        self.assertEqual(appended.segments.size, 5)
        self.assertListEqual(appended[3].tolist(), [10])
        self.assertListEqual(appended[4].tolist(), [20])

        flat = ak.array(a)
        segs = ak.array([0, len(a)])
        segarr = ak.SegArray(segs, flat)
        concated = segarr.append(segarr2, axis=1)
        self.assertListEqual(concated[0].tolist(), [1, 2, 10])
        self.assertListEqual(concated[1].tolist(), [20])

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
        self.assertListEqual(appended.lengths.to_ndarray().tolist(), [7, 3, 5])
        self.assertListEqual(appended[0].tolist(), a + [99])
        self.assertListEqual(appended[1].tolist(), b + [98])
        self.assertListEqual(appended[2].tolist(), c + [97])

        to_append = ak.array([99, 99])
        with self.assertRaises(ValueError):
            appended = segarr.append_single(to_append)

        to_append = ak.array([99.99, 1.1, 2.2])
        with self.assertRaises(TypeError):
            appended = segarr.append_single(to_append)

        to_append = 99
        appended = segarr.append_single(to_append)
        self.assertListEqual(appended.lengths.to_ndarray().tolist(), [7, 3, 5])
        self.assertListEqual(appended[0].tolist(), a + [99])
        self.assertListEqual(appended[1].tolist(), b + [99])
        self.assertListEqual(appended[2].tolist(), c + [99])

        appended = segarr.prepend_single(to_append)
        self.assertListEqual(appended.lengths.to_ndarray().tolist(), [7, 3, 5])
        self.assertListEqual(appended[0].tolist(), [99] + a)
        self.assertListEqual(appended[1].tolist(), [99] + b)
        self.assertListEqual(appended[2].tolist(), [99] + c)

        # test with empty segment
        flat = ak.array(a + b)
        segs = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.SegArray(segs, flat)
        appended = segarr.append_single(99)
        self.assertListEqual(appended[0].tolist(), a + [99])
        self.assertListEqual(appended[1].tolist(), b + [99])
        self.assertListEqual(appended[2].tolist(), [99])

        appended = segarr.prepend_single(99)
        self.assertListEqual(appended[0].tolist(), [99] + a)
        self.assertListEqual(appended[1].tolist(), [99] + b)
        self.assertListEqual(appended[2].tolist(), [99])

    def test_remove_repeats(self):
        a = [1, 1, 1, 2, 3]
        b = [10, 11, 11, 12]

        flat = ak.array(a + b)
        segments = ak.array([0, len(a)])

        segarr = ak.SegArray(segments, flat)
        dedup = segarr.remove_repeats()
        self.assertListEqual(dedup.lengths.to_ndarray().tolist(), [3, 3])
        self.assertListEqual(dedup[0].tolist(), list(set(a)))
        self.assertListEqual(dedup[1].tolist(), list(set(b)))

        # test with empty segments
        segments = ak.array([0, len(a), len(a), len(a) + len(b)])
        segarr = ak.SegArray(segments, flat)
        dedup = segarr.remove_repeats()
        print(dedup.lengths)
        self.assertListEqual(dedup.lengths.to_ndarray().tolist(), [3, 0, 3, 0])
        self.assertListEqual(dedup[0].tolist(), list(set(a)))
        self.assertListEqual(dedup[1].tolist(), [])
        self.assertListEqual(dedup[2].tolist(), list(set(b)))
        self.assertListEqual(dedup[3].tolist(), [])

        segments = ak.array([0, len(a), len(a), len(a), len(a) + len(b)])
        segarr = ak.SegArray(segments, flat)
        dedup = segarr.remove_repeats()
        self.assertListEqual(dedup.lengths.to_ndarray().tolist(), [3, 0, 0, 3, 0])
        self.assertListEqual(dedup[0].tolist(), list(set(a)))
        self.assertListEqual(dedup[1].tolist(), [])
        self.assertListEqual(dedup[2].tolist(), [])
        self.assertListEqual(dedup[3].tolist(), list(set(b)))
        self.assertListEqual(dedup[4].tolist(), [])

    def test_intersection(self):
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8]
        c = [1, 2, 4]
        d = [8]
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        segarr_2 = ak.SegArray(ak.array([0, len(c)]), ak.array(c + d))

        intx = segarr.intersect(segarr_2)

        self.assertEqual(intx.size, 2)
        self.assertListEqual(intx[0].tolist(), [1, 2, 4])
        self.assertListEqual(intx[1].tolist(), [8])

        # test with empty Segments
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        intx = segarr.intersect(segarr_2)
        self.assertListEqual(intx.lengths.to_ndarray().tolist(), [3, 0])
        self.assertListEqual(intx[0].tolist(), [1, 2, 4])
        self.assertListEqual(intx[1].tolist(), [])

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + c))
        segarr_2 = ak.SegArray(ak.array([0, len(d)]), ak.array(d + c))
        intx = segarr.intersect(segarr_2)
        self.assertListEqual(intx.lengths.to_ndarray().tolist(), [0, 3])
        self.assertListEqual(intx[0].tolist(), [])
        self.assertListEqual(intx[1].tolist(), [1, 2, 4])

    def test_union(self):
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8]
        c = [1, 2, 4]
        d = [8]

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        segarr_2 = ak.SegArray(ak.array([0, len(c)]), ak.array(c + d))

        un = segarr.union(segarr_2)
        self.assertEqual(un.size, 2)
        self.assertListEqual(un[0].tolist(), [1, 2, 3, 4, 5])
        self.assertListEqual(un[1].tolist(), [6, 7, 8])

        # test with empty segments
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        un = segarr.union(segarr_2)
        self.assertListEqual(un.lengths.to_ndarray().tolist(), [5, 1])
        self.assertListEqual(un[0].tolist(), [1, 2, 3, 4, 5])
        self.assertListEqual(un[1].tolist(), [8])

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        segarr_2 = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        un = segarr.union(segarr_2)
        self.assertListEqual(un.lengths.to_ndarray().tolist(), [5, 0])
        self.assertListEqual(un[0].tolist(), [1, 2, 3, 4, 5])
        self.assertListEqual(un[1].tolist(), [])

    def test_setdiff(self):
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8]
        c = [1, 2, 4]
        d = [8]

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        segarr_2 = ak.SegArray(ak.array([0, len(c)]), ak.array(c + d))

        diff = segarr.setdiff(segarr_2)
        self.assertEqual(diff.size, 2)
        self.assertListEqual(diff[0].tolist(), [3, 5])
        self.assertListEqual(diff[1].tolist(), [6, 7])

        # test with empty segments
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        diff = segarr.setdiff(segarr_2)
        self.assertListEqual(diff.lengths.to_ndarray().tolist(), [2, 0])
        self.assertListEqual(diff[0].tolist(), [3, 5])
        self.assertListEqual(diff[1].tolist(), [])

        segarr = ak.SegArray(ak.array([0, len(a), len(a)]), ak.array(a + a))
        segarr_2 = ak.SegArray(ak.array([0, len(c), len(c + c)]), ak.array(c + c + c))
        diff = segarr_2.setdiff(segarr)
        self.assertListEqual(diff.lengths.to_ndarray().tolist(), [0, 3, 0])
        self.assertListEqual(diff[0].tolist(), [])
        self.assertListEqual(diff[1].tolist(), [1, 2, 4])
        self.assertListEqual(diff[2].tolist(), [])

    def test_setxor(self):
        a = [1, 2, 3]
        b = [6, 7, 8]
        c = [1, 2, 4]
        d = [8, 12, 13]

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        segarr_2 = ak.SegArray(ak.array([0, len(c)]), ak.array(c + d))
        xor = segarr.setxor(segarr_2)

        self.assertEqual(xor.size, 2)
        self.assertListEqual(xor[0].tolist(), [3, 4])
        self.assertListEqual(xor[1].tolist(), [6, 7, 12, 13])

        # test with empty segment
        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a))
        xor = segarr.setxor(segarr_2)
        self.assertListEqual(xor.lengths.to_ndarray().tolist(), [2, 3])
        self.assertListEqual(xor[0].tolist(), [3, 4])
        self.assertListEqual(xor[1].tolist(), [8, 12, 13])

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + a))
        segarr_2 = ak.SegArray(ak.array([0, len(a)]), ak.array(a + c))
        xor = segarr.setxor(segarr_2)
        self.assertListEqual(xor.lengths.to_ndarray().tolist(), [0, 2])
        self.assertListEqual(xor[0].tolist(), [])
        self.assertListEqual(xor[1].tolist(), [3, 4])

    def test_register_attach(self):
        a = [1, 2, 3]
        b = [6, 7, 8]

        segarr = ak.SegArray(ak.array([0, len(a)]), ak.array(a + b))
        # register the seg array
        segarr.register("segarrtest")

        segarr_2 = ak.SegArray.attach("segarrtest")

        self.assertEqual(segarr.size, segarr_2.size)
        self.assertListEqual(
            segarr.lengths.to_ndarray().tolist(), segarr_2.lengths.to_ndarray().tolist()
        )
        self.assertListEqual(
            segarr.segments.to_ndarray().tolist(), segarr_2.segments.to_ndarray().tolist()
        )
        self.assertListEqual(segarr.values.to_ndarray().tolist(), segarr_2.values.to_ndarray().tolist())

        # Verify is_registered
        self.assertTrue(segarr.is_registered())
        # Unregister one component
        segarr.values.unregister()
        with self.assertWarns(UserWarning):
            segarr.is_registered()
        # Unregister all components
        segarr.unregister()
        self.assertFalse(segarr.is_registered())

        self.assertEqual(len(ak.list_registry()), 0)
