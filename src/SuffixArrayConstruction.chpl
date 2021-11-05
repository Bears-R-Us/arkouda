module SuffixArrayConstruction {
  // This module contains algorithms to construct suffix arrays

  //  Chapel implementation of the suffix array construction algorithm using skew algorithm from
  // "Simple Linear Work Suffix Array Construction" by Juha Karkkainen and Peter Sanders (2003)
  // Dec.7, 2020

  /*
  Returns a boolean indicating whether the pair (a1, a2) is less than or equal to pair (b1, b2)

  :arg a1, a2, b1, b2: pairs to be compared
  :type: int

  :returns: boolean True iff pair (a1, a2) is less than or equal to pair (b1, b2)
  */
  inline proc leq_pairs(a1:int, a2:int, b1:int, b2:int): bool {
    return (a1 < b1) || ((a1 == b1) && (a2 <= b2));  // lexicographic order for pairs
  } 

  /*
  Returns a boolean indicating whether the triple (a1, a2, a3) is less than or equal to triple (b1, b2, b3)

  :arg a1, a2, a3, b1, b2, b3: triples to be compared
  :type: int

  :returns: boolean True iff triple (a1, a2, a3) is less than or equal to pair (b1, b2, b3)
  */
  inline proc leq_triples(a1:int, a2:int, a3:int, b1:int, b2:int, b3:int): bool {
    return (a1 < b1) || ((a1 == b1) && leq_pairs(a2, a3, b2, b3));  // lexicographic order for triples
  }

  /*
  Using Radix Sort, stably sorts a[0..n-1] according to b[0..n-1] using keys 0..K from r[] 
  Element a[i] is mapping to r[a[i]] where r is the alphabet with K+1 characters.

  :arg a: array to be sorted
  :type: [] int
  :arg b: array used to sort
  :type: [] int
  :arg r: array containing keys
  :type: [] int || uint(8)
  :arg n: bound for indicies of a and b used during this calculation
  :type: int
  :arg K: number of keys
  :type: int
  */
  proc radixPass(a:[] int, b:[] int, r:[] ?t, n:int, K:int) where t == int || t == uint(8) {
    var c: [0..K] t = 0;

    // count the number of occurences of each character in a
    for i in a[0..#n] do c[r[i]] += 1;
    // calculate the presum of c, so c[i] will be the starting position of different characters
    c = (+ scan c) - c;
    // let b[j] store the position of each a[i] based on their order.
    // The same character but following the previous suffix will be put at the next position.
    for i in a[0..#n] {
      b[c[r[i]]] = i;
      c[r[i]] += 1;
    }
  }

  /*
  Finds the suffix array SA of s[0..n-1] in {1..K}^n (n long with alphabet 1..K)
  Pad out with 0s. Require s[n]=s[n+1]=s[n+2]=0, n>=2. So the size of s should be n+3

  :arg s: n+3 long string to be converted into suffix array (where s[n]=s[n+1]=s[n+2]=0)
  :type: [] int
  :arg SA: Array to be populated with the suffix array for s
  :type: [] int
  :arg n: length of s (before padding out with 0s to n+2)
  :type: int
  :arg K: Size of alphabet
  :type: int
  */
  proc SuffixArraySkew(s:[] int, SA:[] int, n:int, K:int) {
    var n0 = (n+2)/3: int;
    var n1 = (n+1)/3: int;
    var n2 = n/3: int;
    var n02 = n0 + n2: int;
    var n12 = n1 + n2: int;
    //number of elements meet i%3 = 0, 1, and 2.
    //s[i] is the ith suffix, i in 0..n-1
    var s12: [0..n02+2] int = 0;
    // Here n02 instead of n12=n1+n2 is used for the later s0 building based on n1 elements
    var SA12: [0..n02+2] int = 0;

    var s0: [0..n0+2] int;
    var SA0: [0..n0+2] int;
    var j = 0: int;

    // generate positions of mod 1 and mod 2 suffixes
    // n0-n1 is used for building s0, s1 has the same number of elements as s0
    for i in 0..n+(n0-n1)-1 {
      if i%3 != 0 {
        s12[j] = i;
        j += 1;
      }
    }
    // lsb radix sort the mod 1 and mod 2 triples
    var tmps: [0..n+2] int;
    forall i in 0..n-2 do tmps[i] = s[i+2];
    radixPass(s12, SA12, tmps, n02, K);
    forall i in 0..n-1 do tmps[i] = s[i+1];
    radixPass(SA12, s12, tmps, n02, K);
    radixPass(s12, SA12, s, n02, K);
    // find lexicographic names of triples
    var name = 0: int, c0 = -1: int, c1 = -1: int, c2 = -1: int;

    for i in SA12[0..#n02] {
      if s[i] != c0 || s[i+1] != c1 || s[i+2] != c2 {
        name += 1;
        c0 = s[i];
        c1 = s[i+1];
        c2 = s[i+2];
      }
      if i % 3 == 1 {
        s12[i/3] = name; // mapping the suffix to small alphabets
      } // left half
      else {
        s12[i/3 + n0] = name;
      } // right half
    }

    // recurse if names are not unique
    if name < n02 {
      SuffixArraySkew(s12, SA12, n02, name);
      // store unique names in s12 using the suffix array
      for i in 0..#n02 do s12[SA12[i]] = i+1;
      //restore the value of s12 since we will change its values during the procedure
    }
    else { // generate the suffix array of s12 directly
      for i in 0..#n02 do SA12[s12[i]-1] = i;
      // here SA12 is in fact the ISA array.
    }
    // stably sort the mod 0 suffixes from SA12 by their first character
    j = 0;
    for i in SA12[0..#n02] {
      // here in fact we take advantage of the sorted SA12 to just sort s0 once to get its sorted array
      // at first we think the postion i%3=1 is the position
      if i < n0 {
        s0[j] = 3*i;
        j += 1;
      }
    }
    radixPass(s0, SA0, s, n0, K);

    // merge sorted SA0 suffixes and sorted SA12 suffixes
    var p = 0: int; // first s0 position
    var t = n0-n1: int; //first s1 position
    var k = 0: int;
    for tmpk in 0..#n {
      proc GetI(): int {
        return if SA12[t] < n0 then SA12[t] * 3 + 1 else (SA12[t]-n0) * 3 + 2;
      }
      var i = GetI(); // pos of current offset 12 suffix
      j = SA0[p]; // pos of current offset 0 suffix
      var flag: bool = if SA12[t] < n0 then leq_pairs(s[i], s12[SA12[t]+n0], s[j], s12[j/3]) else leq_triples(s[i], s[i+1], s12[SA12[t]-n0+1], s[j], s[j+1], s12[j/3+n0]);
      if flag {
        // suffix from SA12 is smaller
        SA[k] = i;
        k += 1;
        t += 1;
        if t == n02 {
          // done --- only SA0 suffixes left
          forall (i1,j1) in zip (k..n-1, p..#(n-k)) do SA[i1] = SA0[j1];
          break;
        }
      }
      else {
        // suffix from SA0 is smaller
        SA[k] = j;
        k += 1;
        p += 1;
        var tmpt = t: int;
        if p == n0 {
          // done --- only SA12 suffixes left
          for i1 in tmpt..n02-1 {
            SA[k] = GetI();
            t += 1;
            k += 1;
          }
          break;
        }
      }
    }
  }
}
