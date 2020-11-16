import numpy as np
from collections import Counter
from context import arkouda as ak
from base_test import ArkoudaTest
import pytest
ak.verbose = False

N = 100
UNIQUE = N//4

# test_strings = np.array(['These are', 'some', 'interesting',
#                          '~!@#$%^&*()_+', 'strings', '8675309.',
#                          'These are', 'some', 'duplicates.',
#                          'hello', 'world'])

def compare_strings(a, b):
    return all(x == y for x, y in zip(a, b))
  
errors = False

def run_test_argsort(strings, test_strings, cat):
    akperm = ak.argsort(strings)
    aksorted = strings[akperm].to_ndarray()
    npsorted = np.sort(test_strings)
    assert((aksorted == npsorted).all())
    catperm = ak.argsort(cat)
    catsorted = cat[catperm].to_ndarray()
    assert((catsorted == npsorted).all())

def run_test_unique(strings, test_strings, cat):
    # unique
    akuniq = ak.unique(strings)
    catuniq = ak.unique(cat)
    akset = set(akuniq.to_ndarray())
    catset = set(catuniq.to_ndarray())
    assert(akset == catset)
    # There should be no duplicates
    assert(akuniq.size == len(akset))
    npset = set(np.unique(test_strings))
    # When converted to a set, should agree with numpy
    assert(akset == npset)
    return akset

def run_test_index(strings, test_strings, cat):
    # int index
    assert(strings[N//3] == test_strings[N//3])
    assert(cat[N//3] == test_strings[N//3])
    print("int index passed")
    
def run_test_slice(strings, test_strings, cat):
    assert(compare_strings(strings[N//4:N//3].to_ndarray(), 
                           test_strings[N//4:N//3]))
    assert(compare_strings(cat[N//4:N//3].to_ndarray(), 
                           test_strings[N//4:N//3]))
    
def run_test_pdarray_index(strings, test_strings, cat):
    inds = ak.arange(0, strings.size, 10)
    assert(compare_strings(strings[inds].to_ndarray(), test_strings[inds.to_ndarray()]))
    assert(compare_strings(cat[inds].to_ndarray(), test_strings[inds.to_ndarray()]))

def run_comparison_test(strings, test_strings, cat):
    akinds = (strings == test_strings[N//4])
    catinds = (cat == test_strings[N//4])
    npinds = (test_strings == test_strings[N//4])
    assert(np.allclose(akinds.to_ndarray(), npinds))

def run_test_in1d(strings, cat, base_words):
    more_choices = ak.randint(0, UNIQUE, 100)
    akwords = base_words[more_choices]
    more_words = akwords.to_ndarray()
    matches = ak.in1d(strings, akwords)
    catmatches = ak.in1d(cat, akwords)
    assert((matches == catmatches).all())
    # Every word in matches should be in the target set
    for word in strings[matches].to_ndarray():
        assert(word in more_words)
    # Exhaustively find all matches to make sure we didn't miss any
    inds = ak.zeros(strings.size, dtype=ak.bool)
    for word in more_words:
        inds |= (strings == word)
    assert((inds == matches).all())

def run_test_groupby(strings, cat, akset):
    g = ak.GroupBy(strings)
    gc = ak.GroupBy(cat)
    # Unique keys should be same result as ak.unique
    assert(akset == set(g.unique_keys.to_ndarray()))
    assert(akset == set(gc.unique_keys.to_ndarray()))
    assert((gc.permutation == g.permutation).all())
    permStrings = strings[g.permutation].to_ndarray()
    # Check each group individually
    lengths = np.diff(np.hstack((g.segments.to_ndarray(), np.array([g.size]))))
    for uk, s, l in zip(g.unique_keys.to_ndarray(),
                        g.segments.to_ndarray(),
                        lengths):
        # All values in group should equal key
        assert((permStrings[s:s+l] == uk).all())
        # Key should not appear anywhere outside of group
        assert(not (permStrings[:s] == uk).any())
        assert(not (permStrings[s+l:] == uk).any())


def run_test_contains(strings, test_strings, delim):
    found = strings.contains(delim).to_ndarray()
    npfound = np.array([s.count(delim) > 0 for s in test_strings])
    assert((found == npfound).all())

def run_test_starts_with(strings, test_strings, delim):
    found = strings.startswith(delim).to_ndarray()
    npfound = np.array([s.startswith(delim) for s in test_strings])
    assert((found == npfound).all())

def run_test_ends_with(strings, test_strings, delim):
    found = strings.endswith(delim).to_ndarray()
    npfound = np.array([s.endswith(delim) for s in test_strings])
    assert((found == npfound).all())

def run_test_peel(strings, test_strings, delim):
    import itertools as it
    tf = (True, False)
    def munge(triple, inc, part):
        ret = []
        for h, s, t in triple:
            if not part and s == '':
                ret.append(('', h))
            else:
                if inc:
                    ret.append((h + s, t))
                else:
                    ret.append((h, t))
        l, r = tuple(zip(*ret))
        return np.array(l), np.array(r)

    def rmunge(triple, inc, part):
        ret = []
        for h, s, t in triple:
            if not part and s == '':
                ret.append((t, ''))
            else:
                if inc:
                    ret.append((h, s + t))
                else:
                    ret.append((h, t))
        l, r = tuple(zip(*ret))
        return np.array(l), np.array(r)

    def slide(triple, delim):
        h, s, t = triple
        h2, s2, t2 = t.partition(delim)
        newh = h + s + h2
        return newh, s2, t2

    def rslide(triple, delim):
        h, s, t = triple
        h2, s2, t2 = h.rpartition(delim)
        newt = t2 + s + t
        return h2, s2, newt
    
    for times, inc, part in it.product(range(1,4), tf, tf):
        ls, rs = strings.peel(delim, times=times, includeDelimiter=inc, keepPartial=part)
        triples = [s.partition(delim) for s in test_strings]
        for i in range(times-1):
            triples = [slide(t, delim) for t in triples]
        ltest, rtest = munge(triples, inc, part)
        assert((ltest == ls.to_ndarray()).all() and (rtest == rs.to_ndarray()).all())

    for times, inc, part in it.product(range(1,4), tf, tf):
        ls, rs = strings.rpeel(delim, times=times, includeDelimiter=inc, keepPartial=part)
        triples = [s.rpartition(delim) for s in test_strings]
        for i in range(times-1):
            triples = [rslide(t, delim) for t in triples]
        ltest, rtest = rmunge(triples, inc, part)
        assert((ltest == ls.to_ndarray()).all() and (rtest == rs.to_ndarray()).all())

def run_test_stick(strings, test_strings, base_words, delim):
    test_strings2 = np.random.choice(base_words.to_ndarray(), N, replace=True)
    strings2 = ak.array(test_strings2)
    stuck = strings.stick(strings2, delimiter=delim).to_ndarray()
    tstuck = np.array([delim.join((a, b)) for a, b in zip(test_strings, test_strings2)])
    assert ((stuck == tstuck).all())
    assert ((strings + strings2) == strings.stick(strings2, delimiter="")).all()

    lstuck = strings.lstick(strings2, delimiter=delim).to_ndarray()
    tlstuck = np.array([delim.join((b, a)) for a, b in zip(test_strings, test_strings2)])
    assert ((lstuck == tlstuck).all())
    assert ((strings2 + strings) == strings.lstick(strings2, delimiter="")).all()

        
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        ak.connect(server=sys.argv[1], port=sys.argv[2])
    else:
        ak.connect()

    # with open(__file__, 'r') as f:
    #     base_words = np.array(f.read().split())
    # test_strings = np.random.choice(base_words, N, replace=True)
    # strings = ak.array(test_strings)

    base_words1 = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
    base_words2 = ak.random_strings_lognormal(2, 0.25, UNIQUE, characters='printable')
    gremlins = ak.array([' ', ''])
    base_words = ak.concatenate((base_words1, base_words2, gremlins))
    np_base_words = np.hstack((base_words1.to_ndarray(), base_words2.to_ndarray()))
    assert(compare_strings(base_words.to_ndarray(), np_base_words))
    choices = ak.randint(0, base_words.size, N)
    strings = ak.concatenate((base_words[choices], gremlins))
    test_strings = strings.to_ndarray()
    cat = ak.Categorical(strings)
    print("strings =", strings)
    print("categorical =", cat)
    print("Generation and concatenate passed")
  
    # int index
    run_test_index(strings, test_strings, cat)
    print("int index passed")
  
    # slice
    run_test_slice(strings, test_strings, cat)
    print("slice passed")
    
    # pdarray int index
    run_test_pdarray_index(strings, test_strings, cat)
    print("pdarray int index passed")

    # comparison
    run_comparison_test(strings, test_strings, cat)
    print("comparison passed")

    # pdarray bool index
    run_test_pdarray_index(strings, test_strings, cat)
    print("pdarray bool index passed")

    # in1d and iter
    # more_words = np.random.choice(base_words, 100)
    # akwords = ak.array(more_words)
    run_test_in1d(strings, cat, base_words)
    print("in1d and iter passed")

    # argsort
    run_test_argsort(strings, test_strings, cat)
    
    # unique
    akset = run_test_unique(strings, test_strings, cat)

    # groupby
    run_test_groupby(strings, cat, akset)
    print("groupby passed")

    # substring functions
    x, w = tuple(zip(*Counter(''.join(base_words.to_ndarray())).items()))
    delim = np.random.choice(x, p=(np.array(w)/sum(w)))

    # contains
    run_test_contains(strings, test_strings, delim)
    print("contains passed")

    # startswith
    run_test_starts_with(strings, test_strings, delim)
    print("startswith passed")

    # endswith
    run_test_ends_with(strings, test_strings, delim)
    print("endswith passed")

    # peel
    run_test_peel(strings, test_strings, delim)
    print("peel passed")

    # stick
    run_test_stick(strings, test_strings, base_words, delim)
    print("stick passed")

class StringTest(ArkoudaTest):
  
    def setUp(self):
        ArkoudaTest.setUp(self)
        base_words1 = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        base_words2 = ak.random_strings_lognormal(2, 0.25, UNIQUE, characters='printable')
        gremlins = ak.array([' ', ''])
        self.base_words = ak.concatenate((base_words1, base_words2, gremlins))
        self.np_base_words = np.hstack((base_words1.to_ndarray(), base_words2.to_ndarray()))
        choices = ak.randint(0, self.base_words.size, N)
        self.strings = ak.concatenate((self.base_words[choices], gremlins))
        self.test_strings = self.strings.to_ndarray()
        self.cat = ak.Categorical(self.strings)
        x, w = tuple(zip(*Counter(''.join(self.base_words.to_ndarray())).items()))
        self.delim =  np.random.choice(x, p=(np.array(w)/sum(w)))
        self.akset = set(ak.unique(self.strings).to_ndarray())

    def test_compare_strings(self):
        assert compare_strings(self.base_words.to_ndarray(), self.np_base_words)
    
    def test_argsort(self):
        run_test_argsort(self.strings, self.test_strings, self.cat)

    def test_in1d(self):
        run_test_in1d(self.strings, self.cat, self.base_words)
                      
    def test_unique(self):
        run_test_unique(self.strings, self.test_strings, self.cat)

    def test_groupby(self):
        run_test_groupby(self.strings, self.cat, self.akset)
    
    @pytest.mark.skip(reason="awaiting bug fix.")
    def test_index(self):
        run_test_index(self.strings, self.test_strings, self.cat)
        
    def test_slice(self):
        run_test_slice(self.strings, self.test_strings, self.cat)
        
    def test_pdarray_index(self):
        run_test_pdarray_index(self.strings, self.test_strings, self.cat)

    def test_contains(self):
        run_test_contains(self.strings, self.test_strings, self.delim)
        
    def test_starts_with(self):
        run_test_starts_with(self.strings, self.test_strings, self.delim)

    @pytest.mark.skip(reason="awaiting bug fix.")
    def test_ends_with(self):
        run_test_ends_with(self.strings, self.test_strings, self.delim)
        
    def test_error_handling(self):
        stringsOne = ak.random_strings_uniform(1, 10, UNIQUE, 
                                            characters='printable')
        stringsTwo = ak.random_strings_uniform(1, 10, UNIQUE, 
                                            characters='printable')

        with self.assertRaises(TypeError) as cm:
            stringsOne.lstick(stringsTwo, delimiter=1)
        self.assertEqual('type of argument "delimiter" must be str; got int instead', 
                         cm.exception.args[0])
        
        with self.assertRaises(TypeError) as cm:
            stringsOne.lstick([1], 1)
        self.assertEqual('type of argument "other" must be arkouda.strings.Strings; got list instead', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            stringsOne.startswith(1)
        self.assertEqual('type of argument "substr" must be one of (str, bytes); got int instead', 
                         cm.exception.args[0])    
        
        with self.assertRaises(TypeError) as cm:
            stringsOne.endswith(1)
        self.assertEqual('type of argument "substr" must be one of (str, bytes); got int instead', 
                         cm.exception.args[0])   
        
        with self.assertRaises(TypeError) as cm:
            stringsOne.contains(1)
        self.assertEqual('type of argument "substr" must be one of (str, bytes); got int instead', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            stringsOne.peel(1)
        self.assertEqual('type of argument "delimiter" must be str; got int instead', 
                         cm.exception.args[0])  

        with self.assertRaises(ValueError) as cm:
            stringsOne.peel("",-5)
        self.assertEqual('times must be >= 1', 
                         cm.exception.args[0])  

    @pytest.mark.skip(reason="awaiting bug fix.")
    def test_peel(self):
        run_test_peel(self.strings, self.test_strings, self.delim)

    @pytest.mark.skip(reson="awaiting bug fix.")
    def test_stick(self):
        run_test_stick(self.strings, self.test_strings, self.base_words, self.delim)
