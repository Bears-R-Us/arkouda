module SegStringSort {
  use SegmentedArray;
  use Sort;
  use Time;
  use IO;
  private config const v = true;
  private config const numTasks = here.maxTaskPar;
  private config const MINBYTES = 8;
  private config const MEMFACTOR = 5;
  private config const flattenValues = false;
  
  record StringIntComparator {
    proc keyPart(a: (string, int), i: int) {
      var len = a[1].numBytes;
      var section = if i <= len then 0:int(8) else -1:int(8);
      var part = if i <= len then a[1].byte(i) else 0:uint(8);
      return (section, part);
    }
  }
  
  proc twoPhaseStringSort(ss: SegString): [ss.offsets.aD] int throws {
    if v { writeln("Copy mode = %t".format(copyMode)); stdout.flush(); }
    var t = getCurrentTime();
    const lengths = ss.getLengths();
    if v { writeln("Found lengths in %t seconds".format(getCurrentTime() - t)); stdout.flush(); t = getCurrentTime(); }
    // Compute length survival function and choose a pivot length
    const (pivot, nShort) = getPivot(lengths);
    if v { writeln("Computed pivot in %t seconds".format(getCurrentTime() - t)); writeln("Pivot = %t, nShort = %t".format(pivot, nShort)); stdout.flush(); t = getCurrentTime(); }
    const longStart = ss.offsets.aD.low + nShort;
    const isLong = (lengths >= pivot);
    var locs = [i in ss.offsets.aD] i;
    var longLocs = + scan isLong;
    locs -= longLocs;
    var gatherInds: [ss.offsets.aD] int;
    forall (i, l, ll, t) in zip(ss.offsets.aD, locs, longLocs, isLong) {
      if !t {
        unorderedCopy(gatherInds[l], i);
      } else {
        unorderedCopy(gatherInds[longStart+ll-1], i);
      }
    }
    if v { writeln("Partitioned short/long strings in %t seconds".format(getCurrentTime() - t)); stdout.flush(); }
    on Locales[Locales.domain.high] {
      var tl = getCurrentTime();
      const ref highDom = {longStart..ss.offsets.aD.high};
      ref highInds = gatherInds[highDom];
      // Get local copy of the long strings as Chapel strings, and their original indices
      var stringsWithInds = gatherLongStrings(ss, lengths, highInds);
      if v {writeln("Gathered long strings in %t seconds".format(getCurrentTime() - tl)); stdout.flush(); tl = getCurrentTime(); }
      // Sort the strings, but bring the inds along for the ride
      const myComparator = new StringIntComparator();
      sort(stringsWithInds, comparator=myComparator);
      if v { writeln("Sorted long strings in %t seconds".format(getCurrentTime() - tl)); stdout.flush(); tl = getCurrentTime(); }
      [(h, s) in zip(highDom, stringsWithInds.domain)] unorderedCopy(gatherInds[h], stringsWithInds[s][2]);
      if v { writeln("Permuted long inds in %t seconds".format(getCurrentTime() - tl)); stdout.flush(); }
    }
    if flattenValues {
      if v { t = getCurrentTime(); }
      const heads = getHeads(ss, lengths, pivot);
      if v { writeln("Gathered all strings in %t seconds".format(getCurrentTime() - t)); stdout.flush(); t = getCurrentTime(); }
      const ranks = radixSortLSD_heads(heads, gatherInds, pivot);
      if v { writeln("Sorted ranks in %t seconds".format(getCurrentTime() - t)); stdout.flush(); }
      return ranks;
    } else {
      if v { t = getCurrentTime(); }
      const ranks = radixSortLSD_raw(ss.offsets.a, lengths, ss.values.a, gatherInds, pivot);
      if v { writeln("Sorted ranks in %t seconds".format(getCurrentTime() - t)); stdout.flush(); }
      return ranks;
    }
  }
  
  proc getHeads(ss: SegString, lengths: [?D] int, pivot: int) {
    ref va = ss.values.a;
    var heads: [D] [0..#pivot] uint(8);
    forall (o, l, h) in zip(ss.offsets.a, lengths, heads) {
      const len = min(l, pivot);
      for j in 0..#len {
        unorderedCopy(h[j], va[o+j]);
      }
    }
    return heads;
  }
  
  proc getPivot(lengths: [?D] int): 2*int {
    const NBINS = 2**16;
    const BINDOM = {0..#NBINS};
    var pBins: [PrivateSpace][BINDOM] int;
    coforall loc in Locales {
      on loc {
        const lD = D.localSubdomain();
        ref locLengths = lengths.localSlice[lD];
        var locBins: [0..#numTasks][BINDOM] int;
        coforall task in 0..#numTasks {
          const tD = calcBlock(task, lD.low, lD.high);
          for i in tD {
            var bin = min(locLengths[i], NBINS-1);
            // Count number of *bytes* in bin, not the number of strings
            locBins[task][bin] += locLenghts[i];
          }
        }
        pBins[here.id] = + reduce [task in 0..#numTasks] locBins[task];
      }
    }
    const bins = + reduce [loc in PrivateSpace] pBins[loc];
    // Number of bytes in strings longer than or equal to the current bin
    const tailPop = (+ reduce bins) - (+ scan bins) + bins;
    // Find the largest value of "long" such that long strings fit in one local subdomain
    const singleLocale = (tailPop < (MEMFACTOR * D.localSubdomain().size));
    var (dummy, pivot) = maxloc reduce zip(singleLocale, BINDOM);
    // Pivot should be even and not less than MINBYTES
    pivot = max(pivot + (pivot % 2), MINBYTES);
    // How many strings are "short"?
    const nShort = + reduce (lengths < pivot);
    return (pivot, nShort);
  }
}
