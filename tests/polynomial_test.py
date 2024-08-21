from base_test import ArkoudaTest
from context import arkouda as ak

import pytest
import numpy as np

# Code for Polynomial tests is taken from
#    numpy/polynomial/tests/test_classes.py
# and modified to use Arkouda arrays instead, as allowed under the
# following license:
#
# Copyright (c) 2005-2024, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# fixtures (this does not currently work b/c ArkoudaTest derives from
# unittest.TestCase which it shouldn't do when using pytest; however,
# there is currently only one class to test anyway)
#

classes = (
    ak.Polynomial,
    )
classids = tuple(cls.__name__ for cls in classes)

@pytest.fixture(scope="class")#, params=classes)#, ids=classids)
def Poly(request):
    request.cls.Poly = classes[request.param_index]
    #return request.param

#
# message-counting base class
#

class MessageCounterTest(ArkoudaTest):
    """Base class for Arkouda tests with message counting"""

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._wrapped = dict()
        self.reset_counters()

    def _assert_equal(self, res0, res1):
        if isinstance(res0, (ak.pdarray, ak.Strings, np.ndarray)):
            if isinstance(res0, (ak.pdarray, ak.Strings)):
               res0 = res0.to_ndarray()
            if isinstance(res1, (ak.pdarray, ak.Strings)):
               res1 = res1.to_ndarray()
            assert sum(np.round(res0-res1, 11) == 0) == len(res0)
        elif type(res0) == np.float64:
            assert round(res0-res1, 14) == 0
        else:
            assert res0 == res1

    def _wrap(self, forg, label):
        try:
            return self._wrapped[forg]
        except KeyError:
            class Counter:
                counts = self.counts

                @staticmethod
                def call(*args, **kwds):
                    Counter.counts[label] += 1
                    return forg(*args, **kwds)

            self._wrapped[forg] = Counter.call
            self._wrapped[Counter.call] = forg
            return Counter.call

    def _unwrap(self, fwrap):
        return self._wrapped[fwrap]

    def setup_method(self, meth):
        self.reset_counters()
        ak.pdarray._binop      = self._wrap(ak.pdarray._binop,      'binop')
        ak.pdarray._r_binop    = self._wrap(ak.pdarray._r_binop,    'binop')
        ak.pdarray.__getitem__ = self._wrap(ak.pdarray.__getitem__, 'getitem')
        ak.pdarray.__setitem__ = self._wrap(ak.pdarray.__setitem__, 'setitem')

    def teardown_method(self, meth):
        ak.pdarray.__setitem__ = self._unwrap(ak.pdarray.__setitem__)
        ak.pdarray.__getitem__ = self._unwrap(ak.pdarray.__getitem__)
        ak.pdarray._r_binop    = self._unwrap(ak.pdarray._r_binop)
        ak.pdarray._binop      = self._unwrap(ak.pdarray._binop)

    def reset_counters(self):
        self.counts = {
            'binop'   : 0,
            'getitem' : 0,
            'setitem' : 0,
        }

#
# test methods that depend on one class
#

@pytest.mark.usefixtures("Poly")
class PolynomialTest(MessageCounterTest):
    def test_call(self):
        """Polynomial call running server-side"""

        P = ak.Polynomial
        d = self.Poly.domain
        x = ak.linspace(d[0], d[1], 11)

        # check defaults
        p = self.Poly.cast(P([1, 2, 3]))
        res = p(x)

        # note: there are 2 binop calls to map the domain in the generic
        # case, but by default the offset is 0 and scale is 1, which has
        # been elided for performance
        assert self.counts['binop'] == 0

        # verify result
        tgt = 1 + x*(2 + 3*x)
        self._assert_equal(res, tgt)

    def test_specializations(self):
        """Polynomial call specializations"""

        d = self.Poly.domain
        x = ak.linspace(d[0], d[1], 11)
        z = np.linspace(d[0], d[1], 11)

        binops = 0
        for i in range(2, 7):
            ak_p = ak.Polynomial(np.arange(1, i))
            np_p = np.polynomial.Polynomial(np.arange(1, i))

            res = ak_p(x)
            tgt = np_p(z)

            if i == 2: binops += 1     # b/c of the specific optimization
            assert self.counts['binop'] == binops

            self._assert_equal(res, tgt)


