#!/usr/bin/env python3

import numpy as np
import arkouda as ak
import sys

ak.verbose = False

N = 1000
UNIQUE = N//4

# test_strings = np.array(['These are', 'some', 'interesting',
#                          '~!@#$%^&*()_+', 'strings', '8675309.',
#                          'These are', 'some', 'duplicates.',
#                          'hello', 'world'])

def compare_strings(a, b):
    return all(x == y for x, y in zip(a, b))

errors = False
if __name__ == '__main__':
    if len(sys.argv) > 1:
        ak.connect(server=sys.argv[1], port=sys.argv[2])
    else:
        ak.connect()

    # with open(__file__, 'r') as f:
    #     base_words = np.array(f.read().split())
    # test_strings = np.random.choice(base_words, N, replace=True)
    # strings = ak.array(test_strings)

    base_words = ak.random_strings_lognormal(2, 1, UNIQUE, characters='printable')
    choices = ak.randint(0, UNIQUE, N)
    strings = base_words[choices]
    test_strings = strings.to_ndarray()
    cat = ak.Categorical(strings)
    print("strings =", strings)
    print("categorical =", cat)
    
    # int index
    assert(strings[N//3] == test_strings[N//3])
    assert(cat[N//3] == test_strings[N//3])
    print("int index passed")
    
    # slice
    assert(compare_strings(strings[N//4:N//3].to_ndarray(), test_strings[N//4:N//3]))
    assert(compare_strings(cat[N//4:N//3].to_ndarray(), test_strings[N//4:N//3]))
    print("slice passed")
    
    # pdarray int index
    inds = ak.arange(0, strings.size, 10)
    assert(compare_strings(strings[inds].to_ndarray(), test_strings[inds.to_ndarray()]))
    assert(compare_strings(cat[inds].to_ndarray(), test_strings[inds.to_ndarray()]))
    print("pdarray int index passed")

    # comparison
    akinds = (strings == test_strings[N//4])
    catinds = (cat == test_strings[N//4])
    npinds = (test_strings == test_strings[N//4])
    assert(np.allclose(akinds.to_ndarray(), npinds))
    print("comparison passed")

    # pdarray bool index
    assert(compare_strings(strings[akinds].to_ndarray(), test_strings[npinds]))
    assert(compare_strings(cat[akinds].to_ndarray(), test_strings[npinds]))
    print("pdarray bool index passed")

    # in1d and iter
    # more_words = np.random.choice(base_words, 100)
    # akwords = ak.array(more_words)
    more_choices = ak.randint(0, UNIQUE, 100)
    akwords = base_words[more_choices]
    more_words = akwords.to_ndarray()
    matches = ak.in1d(strings, akwords)
    catmatches = ak.in1d(cat, akwords)
    assert((matches == catmatches).all())
    # Every word in matches should be in the target set
    for word in strings[matches]:
        assert(word in more_words)
    # Exhaustively find all matches to make sure we didn't miss any
    inds = ak.zeros(strings.size, dtype=ak.bool)
    for word in more_words:
        inds |= (strings == word)
    assert((inds == matches).all())
    print("in1d and iter passed")

    # argsort
    akperm = ak.argsort(strings)
    aksorted = strings[akperm].to_ndarray()
    npsorted = np.sort(test_strings)
    assert((aksorted == npsorted).all())
    catperm = ak.argsort(cat)
    catsorted = cat[catperm].to_ndarray()
    assert((catsorted == npsorted).all())
    print("argsort passed")
    
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
    print("unique passed")

    # groupby
    g = ak.GroupBy(strings)
    gc = ak.GroupBy(cat)
    # Unique keys should be same result as ak.unique
    assert(akset == set(g.unique_keys.to_ndarray()))
    assert(akset == set(gc.unique_keys.to_ndarray()))
    assert((gc.permutation == g.permutation).all())
    permStrings = strings[g.permutation]
    # Check each group individually
    lengths = np.diff(np.hstack((g.segments.to_ndarray(), np.array([g.size]))))
    for uk, s, l in zip(g.unique_keys, g.segments, lengths):
        # All values in group should equal key
        assert((permStrings[s:s+l] == uk).all())
        # Key should not appear anywhere outside of group
        assert(not (permStrings[:s] == uk).any())
        assert(not (permStrings[s+l:] == uk).any())
    print("groupby passed")

    # substring functions
    from collections import Counter
    x, w = tuple(zip(*Counter(''.join(base_words)).items()))
    delim = np.random.choice(x, p=(np.array(w)/sum(w)))

    # contains
    found = strings.contains(delim).to_ndarray()
    npfound = np.array([s.count(delim) > 0 for s in test_strings])
    assert((found == npfound).all())
    print("contains passed")

    # startswith
    found = strings.startswith(delim).to_ndarray()
    npfound = np.array([s.startswith(delim) for s in test_strings])
    assert((found == npfound).all())
    print("startswith passed")

    # endswith
    found = strings.endswith(delim).to_ndarray()
    npfound = np.array([s.endswith(delim) for s in test_strings])
    assert((found == npfound).all())
    print("endswith passed")

    # peel
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

    print("peel passed")

    # stick

    test_strings2 = np.random.choice(base_words, N, replace=True)
    strings2 = ak.array(test_strings2)
    stuck = strings.stick(strings2, delimiter=delim).to_ndarray()
    tstuck = np.array([delim.join((a, b)) for a, b in zip(test_strings, test_strings2)])
    assert ((stuck == tstuck).all())
    assert ((strings + strings2) == strings.stick(strings2, delimiter="")).all()

    lstuck = strings.lstick(strings2, delimiter=delim).to_ndarray()
    tlstuck = np.array([delim.join((b, a)) for a, b in zip(test_strings, test_strings2)])
    assert ((lstuck == tlstuck).all())
    assert ((strings2 + strings) == strings.lstick(strings2, delimiter="")).all()
    print("stick passed")
