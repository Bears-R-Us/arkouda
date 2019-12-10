module UnitTestSort
{
  use CommDiagnostics;
  use IO;
  use Memory;
  use Random;
  use Time;

  use BlockDist;

  use RadixSortLSD;
  use AryUtil;

  enum testMode { correctness, correctnessFast, performance, commDiags };
  config const mode = testMode.correctness;

  config const elemsPerLocale = -1;
  const numElems = numLocales * elemsPerLocale;
  config const printArrays = false;

  /* Timing and Comm diagnostic reporting helpers */

  var t: Timer;
  inline proc startDiag() {
    select mode {
      when testMode.performance { t.start(); }
      when testMode.commDiags   { startCommDiagnostics(); }
    }
  }
  inline proc endDiag(name, type elemType, nElems, sortDesc) {
    select mode {
      when testMode.performance { t.stop(); }
      when testMode.commDiags { stopCommDiagnostics(); }
    }

    var sortType = try! "%s(A:[] %s)".format(name, elemType:string);
    writef("%33s -- (%s)", sortType, sortDesc);
    select mode {
      when testMode.performance {
        const sec = t.elapsed();
        const mbPerNode = (nElems * numBytes(elemType)):real / (1024.0*1024.0) / numLocales:real;
        writef(" -- %.2dr MB/s per node (%.2drs)", mbPerNode/sec, sec);
        t.clear();
      }
      when testMode.commDiags {
        const d = getCommDiagnostics();
        const GETS = +reduce (d.get + d.get_nb);
        const PUTS = +reduce (d.put + d.put_nb);
        const ONS = +reduce (d.execute_on + d.execute_on_fast + d.execute_on_nb);
        writef(" -- GETS: %i, PUTS: %i, ONS: %i", GETS, PUTS, ONS);
        resetCommDiagnostics();
      }
    }
    writef("\n");
  }


  /* Main sort testing routine */

  proc testSort(A:[?D], type elemType, nElems, sortDesc) {
    {
      startDiag();
      var sortedA = radixSortLSD_keys(A, checkSorted=false);
      endDiag("radixSortLSD_keys", elemType, nElems, sortDesc);
      if printArrays { writeln(A); writeln(sortedA); }
      assert(isSorted(sortedA));
    }

    {
      startDiag();
      var rankSortedA = radixSortLSD_ranks(A, checkSorted=false);
      endDiag("radixSortLSD_ranks", elemType, nElems, sortDesc);
      // TODO need faster rank sort verification. Could use aggregation,
      // but seems like there should be a comm-free way to verify.
      if mode == testMode.correctness {
        var sortedA: [D] elemType = forall i in rankSortedA do A[i];
        if printArrays { writeln(A); writeln(rankSortedA); writeln(sortedA); }
        assert(isSorted(sortedA));
      }
    }
  }
 
 
  /* Correctness Testing */

  inline proc negateEven(val) {
    if val % 2 == 0 then return -val;
                    else return val;
  }

  // Sort permutations of the indices
  proc testSortIndexPerm(type elemType, nElems) {
    const D = newBlockDom({0..#nElems});
    var A: [D] elemType;

    forall i in D { A[i] = i:elemType; }
    testSort(A, elemType, nElems, "indices");

    forall i in D { A[i] = (nElems - 1 - i):elemType; }
    testSort(A, elemType, nElems, "rev indices");

    forall i in D { A[i] = (nElems/2 - i):elemType; }
    testSort(A, elemType, nElems, "rev shifted indices");

    forall i in D { A[i] = max(elemType) - i:elemType; }
    testSort(A, elemType, nElems, "rev max-indices");

    forall i in D { A[i] = negateEven(i):elemType; }
    testSort(A, elemType, nElems, "negate even indices");
  }


  // Fill an array with random values between min(T) and max(T) and sort it
  proc testSortRandVals(A:[?D], type T, nElems) {
    type elemType = A.eltType;

    var B: [D] T;
    fillRandom(B);
    A = B:A.eltType;

    testSort(A, elemType, nElems, "rand "+T:string+" vals");
  }

  // Sort arrays that contain random values between various int/uint sizes
  // (e.g. sort random values that fit in int(32) and uint(32))
  proc testSortMultRandVals(type elemType, nElems) {
    const D = newBlockDom({0..#nElems});
    var A: [D] elemType;

    testSortRandVals(A, int(8),  nElems);
    testSortRandVals(A, int(16), nElems);
    testSortRandVals(A, int(32), nElems);
    testSortRandVals(A, int(64), nElems);

    testSortRandVals(A, uint(8),  nElems);
    testSortRandVals(A, uint(16), nElems);
    testSortRandVals(A, uint(32), nElems);
    testSortRandVals(A, uint(64), nElems);
  }


  proc testSortActiveBitRange(A, type elemType, nElems, activeBits) {
    var mask: int;
    for bit in activeBits {
      if bit < numBits(int) then mask |= (1:uint<<bit):int;
    }

    fillRandom(A);
    A &= mask;

    testSort(A, elemType, nElems, "activeBits="+activeBits:string);
  }

  // Sort random values where various bit pattens are active. Radix sorting
  // treats values as "bags of bits", so here we mask different bit ranges and
  // bit clusters, which can trigger some interesting corner cases
  proc testSortActiveBitRanges(type elemType, nElems) {
    const D = newBlockDom({0..#nElems});
    var A: [D] elemType;

    const intBits = numBits(int);
    // Test sorting positive values with consecutive active bit ranges
    // (bits 0-(maxBits-1) are active)
    for maxBits in 1..intBits do
      testSortActiveBitRange(A, elemType, nElems, 0..#maxBits);

    // Test sorting where clusters of bits are active (bits 0-15, 16-31, ..)
    for startBit in 0..#intBits by 16 do
      testSortActiveBitRange(A, elemType, nElems, startBit..#16);

    // Test sorting negative values for consecutive active bit ranges
    // (startBit-(intBits-1) are active)
    for maxBits in 1..intBits do
      testSortActiveBitRange(A, elemType, nElems, intBits-maxBits..intBits-1);

    // Test sorting values where a strides of bits are active
    for stride in 1..intBits do
      testSortActiveBitRange(A, elemType, nElems, 0..intBits-1 by stride);

    // Test integers that use the full bit range 
    testSortActiveBitRange(A, elemType, nElems, 0..#intBits);
  }


  /* Performance Testing */

  // By default, sort ints with uint(16) values using 1/50 of memory
  config type perfElemType = int,
              perfValRange = uint(16);
  config const perfMemFraction = 50;
  config param perfOnlyCompile = false; // reduces compilation time

  proc testPerformance() {
    param elemSize = numBytes(perfElemType);
    const totMem = here.physicalMemory(unit = MemUnits.Bytes);
    const fraction = totMem / elemSize / perfMemFraction * numLocales;
    const nElems = if numElems > 0 then numElems else fraction;

    const D = newBlockDom({0..#nElems});
    var A: [D] perfElemType;
    var B: [D] perfValRange;
    fillRandom(B);
    A = B:perfElemType;

    // Warmup
    radixSortLSD_keys(A, checkSorted=false);
    radixSortLSD_ranks(A, checkSorted=false);

    testSort(A, perfElemType, nElems, "rand "+perfValRange:string+" vals");
  }
 
  proc main() {
    const correctness = mode == testMode.correctness || mode == testMode.correctnessFast;
    if !perfOnlyCompile && correctness {
      const nElems = if numElems > 0 then numElems else numLocales*10000;
      testSortIndexPerm(int, nElems);
      testSortIndexPerm(real, nElems);
      testSortMultRandVals(int, nElems);
      testSortMultRandVals(real, nElems);
      if mode == testMode.correctness {
        testSortActiveBitRanges(int, nElems);
      }
    } else {
      testPerformance();
    }
  }
}
