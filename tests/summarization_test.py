import numpy as np

import arkouda as ak


"""
Encapsulates unit tests for the pdarrayclass module that provide
summarized values via reduction methods
"""


class TestSummarization:
    @classmethod
    def setup_class(cls):
        cls.na = np.linspace(1, 10, 10)
        cls.pda = ak.array(cls.na)

    def testStd(self):
        assert self.na.std() == self.pda.std()

    def testMin(self):
        assert self.na.min() == self.pda.min()

    def testMax(self):
        assert self.na.max() == self.pda.max()

    def testMean(self):
        assert self.na.mean() == self.pda.mean()

    def testVar(self):
        assert self.na.var() == self.pda.var()

    def testAny(self):
        assert self.na.any() == self.pda.any()

    def testAll(self):
        assert self.na.all() == self.pda.all()
