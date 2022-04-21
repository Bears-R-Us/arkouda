from base_test import ArkoudaTest
from context import arkouda as ak
from arkouda.util import attach

class utilTest(ArkoudaTest):
    def test_attach(self):
        a = ak.array(["abc","123","def"])
        b = ak.arange(10)

        #Attach the Strings array and the pdarray to new objects
        a_attached = attach(a.entry.name) #This should be updated after Strings' init of the name property is changed
        b_attached = attach(b.name)

        self.assertTrue((a == a_attached).all())
        self.assertIsInstance(a_attached, ak.Strings)
        self.assertTrue((b == b_attached).all())
        self.assertIsInstance(b_attached, ak.pdarray)