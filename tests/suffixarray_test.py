import numpy as np
from collections import Counter
from context import arkouda as ak
from base_test import ArkoudaTest
import pytest
import random
import string
ak.verbose = False

N = 100
UNIQUE = N//2

# test_strings = np.array(['These are', 'some', 'interesting',
#                          '~!@#$%^&*()_+', 'sarrays', '8675309.',
#                          'These are', 'some', 'duplicates.',
#                          'hello', 'world'])

# test_suffix array = np.array([9, 5, 0, 6, 8, 4, 2, 1, 7, 3],
#                              [4, 3, 2, 1, 0], [11, 3, 5, 10, 8, 0, 9, 1, 4, 6, 2, 7]
#                              [13, 1, 3, 4, 5, 7, 9, 10, 8, 12, 2, 6, 11, 0],
#                              [7, 5, 3, 4, 2, 6, 0, 1],[8, 7, 5, 4, 3, 1, 2, 0, 6],
#                              [9, 5, 0, 6, 8, 4, 2, 1, 7, 3],[4, 3, 2, 1, 0],
#                              [10, 6, 5, 0, 8, 4, 3, 2, 9, 7, 1],[5, 1, 0, 2, 3, 4]
#                              [5, 4, 3, 1, 2, 0]
def compare_sas(a, b):
    return all(x == y for x, y in zip(a, b))
  
errors = False

def run_test_argsort(sarrays, test_sas, cat):
    akperm = ak.argsort(sarrays)
    aksorted = sarrays[akperm].to_ndarray()
    npsorted = np.sort(test_sas)
    assert((aksorted == npsorted).all())
    catperm = ak.argsort(cat)
    catsorted = cat[catperm].to_ndarray()
    assert((catsorted == npsorted).all())

def run_test_unique(sarrays, test_sas, cat):
    # unique
    akuniq = ak.unique(sarrays)
    catuniq = ak.unique(cat)
    akset = set(akuniq.to_ndarray())
    catset = set(catuniq.to_ndarray())
    assert(akset == catset)
    # There should be no duplicates
    assert(akuniq.size == len(akset))
    npset = set(np.unique(test_sas))
    # When converted to a set, should agree with numpy
    assert(akset == npset)
    return akset

def run_test_index(sarrays, test_sas, cat):
    # int index
    assert(sarrays[N//3] == test_sas[N//3])
    #assert(cat[N//3] == test_sas[N//3])
    print("int index passed")
    
def run_test_slice(sarrays, test_sas, cat):
    assert(compare_sas(sarrays[N//4:N//3], 
                           test_sas[N//4:N//3]))
    #assert(compare_sas(cat[N//4:N//3].to_ndarray(), 
    #                       test_sas[N//4:N//3]))
    
def run_test_pdarray_index(sarrays, test_sas, cat):
    inds = ak.arange(0, len(sarrays), 10)
    assert(compare_sas(sarrays[inds].to_ndarray(), test_sas[inds.to_ndarray()]))
    #assert(compare_sas(cat[inds].to_ndarray(), test_sas[inds.to_ndarray()]))

def run_comparison_test(sarrays, test_sas, cat):
    akinds = (sarrays == test_sas[N//4])
    #catinds = (cat == test_sas[N//4])
    npinds = (test_sas == test_sas[N//4])
    assert(np.allclose(akinds, npinds))

def run_test_in1d(sarrays, cat, base_words):
    more_choices = ak.randint(0, UNIQUE, 100)
    #akwords = base_words[more_choices]
    #more_words = akwords.to_ndarray()
    matches = ak.in1d(sarrays, akwords)
    catmatches = ak.in1d(cat, akwords)
    assert((matches == catmatches).all())
    # Every word in matches should be in the target set
    for word in sarrays[matches].to_ndarray():
        assert(word in more_words)
    # Exhaustively find all matches to make sure we didn't miss any
    inds = ak.zeros(sarrays.size, dtype=ak.bool)
    for word in more_words:
        inds |= (sarrays == word)
    assert((inds == matches).all())

def run_test_groupby(sarrays, cat, akset):
    g = ak.GroupBy(sarrays)
    gc = ak.GroupBy(cat)
    # Unique keys should be same result as ak.unique
    assert(akset == set(g.unique_keys.to_ndarray()))
    assert(akset == set(gc.unique_keys.to_ndarray()))
    assert((gc.permutation == g.permutation).all())
    permStrings = sarrays[g.permutation].to_ndarray()
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


def run_test_contains(sarrays, test_sas, delim):
    found = sarrays.contains(delim).to_ndarray()
    npfound = np.array([s.count(delim) > 0 for s in test_sas])
    assert((found == npfound).all())

def run_test_starts_with(sarrays, test_sas, delim):
    found = sarrays.startswith(delim).to_ndarray()
    npfound = np.array([s.startswith(delim) for s in test_sas])
    assert((found == npfound).all())

def run_test_ends_with(sarrays, test_sas, delim):
    found = sarrays.endswith(delim).to_ndarray()
    npfound = np.array([s.endswith(delim) for s in test_sas])
    assert((found == npfound).all())

def run_test_peel(sarrays, test_sas, delim):
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
        ls, rs = sarrays.peel(delim, times=times, includeDelimiter=inc, keepPartial=part)
        triples = [s.partition(delim) for s in test_sas]
        for i in range(times-1):
            triples = [slide(t, delim) for t in triples]
        ltest, rtest = munge(triples, inc, part)
        assert((ltest == ls.to_ndarray()).all() and (rtest == rs.to_ndarray()).all())

    for times, inc, part in it.product(range(1,4), tf, tf):
        ls, rs = sarrays.rpeel(delim, times=times, includeDelimiter=inc, keepPartial=part)
        triples = [s.rpartition(delim) for s in test_sas]
        for i in range(times-1):
            triples = [rslide(t, delim) for t in triples]
        ltest, rtest = rmunge(triples, inc, part)
        assert((ltest == ls.to_ndarray()).all() and (rtest == rs.to_ndarray()).all())

def run_test_stick(sarrays, test_sas, base_words, delim):
    test_sas2 = np.random.choice(base_words.to_ndarray(), N, replace=True)
    sarrays2 = ak.array(test_sas2)
    stuck = sarrays.stick(sarrays2, delimiter=delim).to_ndarray()
    tstuck = np.array([delim.join((a, b)) for a, b in zip(test_sas, test_sas2)])
    assert ((stuck == tstuck).all())
    assert ((sarrays + sarrays2) == sarrays.stick(sarrays2, delimiter="")).all()

    lstuck = sarrays.lstick(sarrays2, delimiter=delim).to_ndarray()
    tlstuck = np.array([delim.join((b, a)) for a, b in zip(test_sas, test_sas2)])
    assert ((lstuck == tlstuck).all())
    assert ((sarrays2 + sarrays) == sarrays.lstick(sarrays2, delimiter="")).all()

def suffixArray(s):
    suffixes = [(s[i:], i) for i in range(len(s))]
    suffixes.sort(key=lambda x: x[0])
    sa= [s[1] for s in suffixes]
    #sa.insert(0,len(sa))
    return sa

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
#    print("Random string of length", length, "is:", result_str)
        
def ascill_to_string(ini_list):
    res=""
    for val in ini_list: 
       res = res + chr(int(val))
    return res


def string_to_int(sa_str):
    ary=[]
    for val in sa_str: 
       ary.append(int(val))
    return ary

def akstrings_to_suffix_array(ak_str):
    ary=[]
    for val in ak_str: 
        x=val.split(" ",1)
        y=x[1]
        z=y.split(" ")
        s=ascill_to_string(z)
        sa=suffixArray(s)
        ary.append(sa)
    return ary

def aksa_to_int_array(ak_str):
    ary=[]
    for val in ak_str: 
        x=val.split(" ",1)
        y=x[1]
        z=y.split(" ")
        intz= [int(z[i]) for i in range(len(z))]
        ary.append(intz)
    return ary
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        ak.connect(server=sys.argv[1], port=sys.argv[2])
    else:
        ak.connect()

    # with open(__file__, 'r') as f:
    #     base_words = np.array(f.read().split())
    # test_sas = np.random.choice(base_words, N, replace=True)
    # sarrays = ak.array(test_sas)
    # generate a Strings object
    base_words1 = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
    # get the real strings
    strings1 = [base_words1[i] for i in range(len(base_words1))]
    # generate  a  Strings object
    base_words2 = ak.random_strings_lognormal(2, 0.25, UNIQUE, characters='printable')
    # get the real strings
    strings2 = [base_words2[i] for i in range(len(base_words2))]
    #Generate suffix array locally
    sa_ori1=akstrings_to_suffix_array(strings1)
    #Generate suffix array locally
    sa_ori2=akstrings_to_suffix_array(strings2)
    #Generate suffix array remotely
    sa1=ak.suffix_array(base_words1)
    #Generate suffix array remotely
    sa2=ak.suffix_array(base_words2)
    #get the suffix array from SArray object
    suffixarray1=[sa1[i] for i in range(len(sa1))]
    #transfer the string suffix array to real int suffix array
    sa_test1=aksa_to_int_array(suffixarray1)
    #get the suffix array from SArray object
    suffixarray2=[sa2[i] for i in range(len(sa2))]
    #transfer the string suffix array to real int suffix array
    sa_test2=aksa_to_int_array(suffixarray2)
    
    cat=0
    # int index
    run_test_index(sa_ori1, sa_test1, cat)
    run_test_index(sa_ori2, sa_test2, cat)
    print("int index passed")
    
    # slice
    run_test_slice(sa_ori1, sa_test1, cat)
    run_test_slice(sa_ori2, sa_test2, cat)
    print("slice passed")
    
    # pdarray int index
    #run_test_pdarray_index(sa_ori1, sa_test1, cat)
    #run_test_pdarray_index(sa_ori2, sa_test2, cat)
    #print("pdarray int index passed")

    # comparison
    run_comparison_test(sa_ori1, sa_test1, cat)
    run_comparison_test(sa_ori2, sa_test2, cat)
    print("comparison passed")

    # pdarray bool index
    #run_test_pdarray_index(sarrays, test_sas, cat)
    #print("pdarray bool index passed")

    # in1d and iter
    # more_words = np.random.choice(base_words, 100)
    # akwords = ak.array(more_words)
    #run_test_in1d(sa_ori1, sa_test1, cat)
    #run_test_in1d(sa_ori2, sa_test2, cat)
    #print("in1d and iter passed")

    # argsort
    #run_test_argsort(sa_ori1, sa_test1, cat)
    
    # unique
    #akset = run_test_unique(sarrays, test_sas, cat)
    '''
    # groupby
    run_test_groupby(sarrays, cat, akset)
    print("groupby passed")
    
    # substring functions
    x, w = tuple(zip(*Counter(''.join(base_words.to_ndarray())).items()))
    delim = np.random.choice(x, p=(np.array(w)/sum(w)))

    # contains
    run_test_contains(sarrays, test_sas, delim)
    print("contains passed")

    # startswith
    run_test_starts_with(sarrays, test_sas, delim)
    print("startswith passed")

    # endswith
    run_test_ends_with(sarrays, test_sas, delim)
    print("endswith passed")

    # peel
    run_test_peel(sarrays, test_sas, delim)
    print("peel passed")

    # stick
    run_test_stick(sarrays, test_sas, base_words, delim)
    print("stick passed")
    '''
class SuffixArrayTest(ArkoudaTest):
  
    def setUp(self):
        ArkoudaTest.setUp(self)
        base_words1 = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        base_words2 = ak.random_strings_lognormal(2, 0.25, UNIQUE, characters='printable')
        base_sas1 = ak.suffix_array(base_words1)
        base_sas2 = ak.suffix_array(base_words2)
        '''
        gremlins = ak.array([' ', ''])
        self.base_words = ak.concatenate((base_words1, base_words2, gremlins))
        self.np_base_words = np.hstack((base_words1.to_ndarray(), base_words2.to_ndarray()))
        choices = ak.randint(0, self.base_words.size, N)
        self.sarrays = ak.concatenate((self.base_words[choices], gremlins))
        self.test_sas = self.sarrays.to_ndarray()
        self.cat = ak.Categorical(self.sarrays)
        x, w = tuple(zip(*Counter(''.join(self.base_words.to_ndarray())).items()))
        self.delim =  np.random.choice(x, p=(np.array(w)/sum(w)))
        self.akset = set(ak.unique(self.sarrays).to_ndarray())
        '''

    def test_compare_sarrays(self):
        assert compare_sarrays(self.base_words.to_ndarray(), self.np_base_words)
    
    def test_argsort(self):
        run_test_argsort(self.sarrays, self.test_sas, self.cat)

    def test_in1d(self):
        run_test_in1d(self.sarrays, self.cat, self.base_words)
                      
    def test_unique(self):
        run_test_unique(self.sarrays, self.test_sas, self.cat)

    def test_groupby(self):
        run_test_groupby(self.sarrays, self.cat, self.akset)
    
    @pytest.mark.skip(reason="awaiting bug fix.")
    def test_index(self):
        run_test_index(self.sarrays, self.test_sas, self.cat)
        
    def test_slice(self):
        run_test_slice(self.sarrays, self.test_sas, self.cat)
        
    def test_pdarray_index(self):
        run_test_pdarray_index(self.sarrays, self.test_sas, self.cat)

    def test_contains(self):
        run_test_contains(self.sarrays, self.test_sas, self.delim)
        
    def test_starts_with(self):
        run_test_starts_with(self.sarrays, self.test_sas, self.delim)

    @pytest.mark.skip(reason="awaiting bug fix.")
    def test_ends_with(self):
        run_test_ends_with(self.sarrays, self.test_sas, self.delim)
        
    def test_error_handling(self):
        sarraysOne = ak.random_sarrays_uniform(1, 10, UNIQUE, 
                                            characters='printable')
        sarraysTwo = ak.random_sarrays_uniform(1, 10, UNIQUE, 
                                            characters='printable')

        with self.assertRaises(TypeError) as cm:
            sarraysOne.lstick(sarraysTwo, delimiter=1)
        self.assertEqual('Delimiter must be a string, not int', 
                         cm.exception.args[0])
        
        with self.assertRaises(TypeError) as cm:
            sarraysOne.lstick([1], 1)
        self.assertEqual('stick: not supported between String and list', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            sarraysOne.startswith(1)
        self.assertEqual('Substring must be a string, not int', 
                         cm.exception.args[0])    
        
        with self.assertRaises(TypeError) as cm:
            sarraysOne.endswith(1)
        self.assertEqual('Substring must be a string, not int', 
                         cm.exception.args[0])   
        
        with self.assertRaises(TypeError) as cm:
            sarraysOne.contains(1)
        self.assertEqual('Substring must be a string, not int', 
                         cm.exception.args[0])  
        
        with self.assertRaises(TypeError) as cm:
            sarraysOne.peel(1)
        self.assertEqual('Delimiter must be a string, not int', 
                         cm.exception.args[0])  

        with self.assertRaises(ValueError) as cm:
            sarraysOne.peel("",-5)
        self.assertEqual('Times must be >= 1', 
                         cm.exception.args[0])  

    @pytest.mark.skip(reason="awaiting bug fix.")
    def test_peel(self):
        run_test_peel(self.sarrays, self.test_sas, self.delim)

    @pytest.mark.skip(reson="awaiting bug fix.")
    def test_stick(self):
        run_test_stick(self.sarrays, self.test_sas, self.base_words, self.delim)
