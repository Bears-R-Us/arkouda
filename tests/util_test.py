from base_test import ArkoudaTest
from context import arkouda as ak
from arkouda.util import attach

class utilTest(ArkoudaTest):
    def test_simple_attach(self):
        a = ak.array(["abc","123","def"])
        b = ak.arange(10)

        # Attach the Strings array and the pdarray to new objects
        a_attached = attach(a.name)
        a_typed_attach = attach(a.name, "strings")
        b_attached = attach(b.name)
        b_typed_attach = attach(b.name, "pdarray")

        self.assertTrue((a == a_attached).all())
        self.assertIsInstance(a_attached, ak.Strings)

        self.assertTrue((a == a_typed_attach).all())
        self.assertIsInstance(a_typed_attach, ak.Strings)

        self.assertTrue((b == b_attached).all())
        self.assertIsInstance(b_attached, ak.pdarray)

        self.assertTrue((b == b_typed_attach).all())
        self.assertIsInstance(b_typed_attach, ak.pdarray)

    def test_categorical_attach(self):
        strings = ak.array(["hurrah", ",", "hurrah", ",", "one", "by", "one", "marching", "go", "ants", "the"])
        cat = ak.Categorical(strings)
        cat.register("catTest")

        attached = attach("catTest")
        self.assertTrue((cat == attached).all())
        self.assertIsInstance(attached, ak.Categorical)

        attached_typed = attach("catTest", "Categorical")
        self.assertTrue((cat == attached_typed).all())
        self.assertIsInstance(attached_typed, ak.Categorical)

    def test_segArray_attach(self):
        a = [10, 11, 12, 13, 14, 15]
        b = [20, 21]
        c = [30, 31, 32, 33]

        flat = a + b + c
        akflat = ak.array(flat)
        segments = ak.array([0, len(a), len(a) + len(b)])
        segarr = ak.SegArray(segments, akflat)
        segarr.register("segTest")

        attached = attach("segTest")
        self.assertTrue((segarr == attached).all())
        self.assertIsInstance(attached, ak.SegArray)

        attached_typed = attach("segTest", "SegArray")
        self.assertTrue((segarr == attached_typed).all())
        self.assertIsInstance(attached_typed, ak.SegArray)