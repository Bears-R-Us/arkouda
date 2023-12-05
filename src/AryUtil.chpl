module AryUtil
{
    use Random;
    use Reflection;
    use Logging;
    use ServerConfig;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrors;
    use BitOps;
    use GenSymIO;

    use ArkoudaPOSIXCompat;
    use ArkoudaCTypesCompat;
    use ArkoudaBlockCompat;

    param bitsPerDigit = RSLSD_bitsPerDigit;
    private param numBuckets = 1 << bitsPerDigit; // these need to be const for comms/performance reasons
    private param maskDigit = numBuckets-1; // these need to be const for comms/performance reasons

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const auLogger = new Logger(logLevel, logChannel);

    /*
      Threshold for the amount of data that will be printed.
      Arrays larger than printThresh will print less data.
    */
    var printThresh = 30;

    /*
      Prints the passed array.

      :arg name: name of the array
      :arg A: array to be printed
    */
    proc formatAry(A: [?d]):string throws {
        if d.rank == 1 {
            var s:string = "";
            if (d.size == 0) {
                s =  ""; // Unnecessary, but left for clarity
            } else if (d.size < printThresh || d.size <= 6) {
                for i in 0..(d.size-2) {s += try! "%?".doFormat(A[i]) + " ";}
                s += try! "%?".doFormat(A[d.size-1]);
            } else {
                s = try! "%? %? %? ... %? %? %?".doFormat(A[0], A[1], A[2], A[d.size-3], A[d.size-2], A[d.size-1]);
            }
            return s;
        } else {
            const shape = d.shape;
            var s = "%?\n".doFormat(shape),
                front_indices: d.rank*range,
                back_indices: d.rank*range;

            for param i in 0..<d.rank {
                front_indices[i] = if shape[i] < 3
                    then 0..<shape[i]
                    else 0..2;
                back_indices[i] = if shape[i] < 3
                    then 0..<shape[i]
                    else (shape[i]-3)..<shape[i];
            }

            const frontDom = {(...front_indices)},
                  backDom = {(...back_indices)};

            s += "%? ... %?".doFormat(A[frontDom], A[backDom]);
        }
    }

    proc printAry(name:string, A) {
        try! writeln(name, formatAry(A));
    }

    /* 1.18 version print out localSubdomains

       :arg x: array
       :type x: []
    */
    proc printOwnership(x) {
        for loc in Locales do
            on loc do
                write(x.localSubdomain(), " ");
        writeln();
    }


    /*
      Determines if the passed array is sorted.

      :arg A: array to check

    */
    proc isSorted(A:[?D] ?t): bool {
        var sorted: bool;
        sorted = true;
        forall (a,i) in zip(A,D) with (&& reduce sorted) {
            if i > D.low {
                sorted &&= (A[i-1] <= a);
            }
        }
        return sorted;
    }

    /*
      Returns stats on a given array in form (int,int,real,real,real).

      :arg a: array to produce statistics on
      :type a: [] int

      :returns: a_min, a_max, a_mean, a_variation, a_stdDeviation
    */
    proc aStats(a: [?D] int): (int,int,real,real,real) {
        var a_min:int = min reduce a;
        var a_max:int = max reduce a;
        var a_mean:real = (+ reduce a:real) / a.size:real;
        var a_vari:real = (+ reduce (a:real **2) / a.size:real) - a_mean**2;
        var a_std:real = sqrt(a_vari);
        return (a_min,a_max,a_mean,a_vari,a_std);
    }

    proc fillUniform(A:[?D] int, a_min:int ,a_max:int, seed:int=241) {
        // random numer generator
        var R = new owned RandomStream(real, seed); R.getNext();
        [a in A] a = (R.getNext() * (a_max - a_min) + a_min):int;
    }

    /*
       Concatenate 2 arrays and return the result.
     */
    proc concatArrays(a: [?aD] ?t, b: [?bD] t) throws {
      var ret = makeDistArray((a.size + b.size), t);

      ret[0..#a.size] = a;
      ret[a.size..#b.size] = b;

      return ret;
    }

    /*
      Iterate over indices (range/domain) ``ind`` but in an offset manner based
      on the locale id. Can be used to avoid doing communication in lockstep.
    */
    iter offset(ind) where isRange(ind) || isDomain(ind) {
        for i in ind + (ind.size/numLocales * here.id) do {
            yield i % ind.size + ind.first;
        }
    }

    /*
      Determines if the passed array array maps contiguous indices to
      contiguous memory.

      :arg A: array to check
    */
    proc contiguousIndices(A: []) param {
        use BlockDist;
        return A.isDefaultRectangular() || isSubtype(A.domain.distribution.type, blockDist);
    }

    /*
     * Takes a variable number of array names from a command message and
     * validates them, checking that they all exist and are the same length
     * and returning metadata about them.
     *
     * :arg n: number of arrays
     * :arg fields: the fields derived from the command message
     * :arg st: symbol table
     *
     * :returns: (length, hasStr, names, objtypes)
     */
    proc validateArraysSameLength(n:int, names:[] string, types: [] string, st: borrowed SymTab) throws {
      // Check that fields contains the stated number of arrays
      if (names.size != n) {
          var errorMsg = "Expected %i arrays but got %i".doFormat(n, names.size);
          auLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          throw new owned ErrorWithContext(errorMsg,
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "ArgumentError");
      }
      if (types.size != n) {
          var errorMsg = "Expected %i types but got %i".doFormat(n, types.size);
          auLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          throw new owned ErrorWithContext(errorMsg,
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "ArgumentError");
      }
      /* var arrays: [0..#n] borrowed GenSymEntry; */
      var size: int;
      // Check that all arrays exist in the symbol table and have the same size
      var hasStr = false;
      for (name, objtype, i) in zip(names, types, 1..) {
        var thisSize: int;
        select objtype.toUpper(): ObjType {
          when ObjType.PDARRAY {
            var g = getGenericTypedArrayEntry(name, st);
            thisSize = g.size;
          }
          when ObjType.STRINGS {
            var (myNames, _) = name.splitMsgToTuple('+', 2);
            var g = getSegStringEntry(myNames, st);
            thisSize = g.size;
            hasStr = true;
          }
          when ObjType.CATEGORICAL {
            if st.contains(name) {
              // passed only Categorical.codes.name to be sorted on
              var g = getGenericTypedArrayEntry(name, st);
              thisSize = g.size;
            }
            else {
              var catComps = jsonToMap(name);
              var codesName = catComps["codes"];
              var codes = getGenericTypedArrayEntry(codesName, st);
              thisSize = codes.size;
            }
          }
          when ObjType.SEGARRAY {
            var segComps = jsonToMap(name);
            var (segName, valName) = (segComps["segments"], segComps["values"]);
            var segs = getGenericTypedArrayEntry(segName, st);
            var vals = getGenericTypedArrayEntry(valName, st);
            thisSize = segs.size;
          }
          otherwise {
              var errorMsg = "Unrecognized object type: %s".doFormat(objtype);
              auLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              throw new owned ErrorWithContext(errorMsg,
                                               getLineNumber(),
                                               getRoutineName(),
                                               getModuleName(),
                                               "TypeError");
          }
        }

        if (i == 1) {
            size = thisSize;
        } else {
            if (thisSize != size) {
              var errorMsg = "Arrays must all be same size; expected size %?, got size %?".doFormat(size, thisSize);
                auLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                throw new owned ErrorWithContext(errorMsg,
                                                 getLineNumber(),
                                                 getRoutineName(),
                                                 getModuleName(),
                                                 "ArgumentError");
            }
        }
      }
      return (size, hasStr, names, types);
    }

    inline proc getBitWidth(a: [?aD] int): (int, bool) {
      var aMin = min reduce a;
      var aMax = max reduce a;
      var wPos = if aMax >= 0 then numBits(int) - clz(aMax) else 0;
      var wNeg = if aMin < 0 then numBits(int) - clz((-aMin)-1) else 0;
      const signBit = if aMin < 0 then 1 else 0;
      const bitWidth = max(wPos, wNeg) + signBit;
      const negs = aMin < 0;
      return (bitWidth, negs);
    }

    inline proc getBitWidth(a: [?aD] uint): (int, bool) {
      const negs = false;
      var aMax = max reduce a;
      var bitWidth = numBits(uint) - clz(aMax):int;
      return (bitWidth, negs);
    }

    inline proc getBitWidth(a: [?aD] real): (int, bool) {
      const bitWidth = numBits(real);
      const negs = | reduce signbit(a);
      return (bitWidth, negs);
    }

    inline proc getBitWidth(a: [?aD] bool): (int, bool) {
      return (1, false);
    }

    inline proc getBitWidth(a: [?aD] (uint, uint)): (int, bool) {
      const negs = false;
      var highMax = max reduce [(ai,_) in a] ai;
      var whigh = numBits(uint) - clz(highMax);
      if (whigh == 0) {
        var lowMax = max reduce [(_,ai) in a] ai;
        var wlow = numBits(uint) - clz(lowMax);
        const bitWidth = wlow: int;
        return (bitWidth, negs);
      } else {
        const bitWidth = (whigh + numBits(uint)): int;
        return (bitWidth, negs);
      }
    }

    inline proc getBitWidth(a: [?aD] ?t): (int, bool)
        where isHomogeneousTuple(t) && t == t.size*uint(bitsPerDigit) {
      for digit in 0..t.size-1 {
        const m = max reduce [ai in a] ai(digit);
        if m > 0 then return ((t.size-digit) * bitsPerDigit, false);
      }
      return (t.size * bitsPerDigit, false);
    }

    // Get the digit for the current rshift. In order to correctly sort
    // negatives, we have to invert the signbit if we're looking at the last
    // digit and the array contained negative values.
    inline proc getDigit(key: int, rshift: int, last: bool, negs: bool): int {
      const invertSignBit = last && negs;
      const xor = (invertSignBit:uint << (RSLSD_bitsPerDigit-1));
      const keyu = key:uint;
      return (((keyu >> rshift) & (maskDigit:uint)) ^ xor):int;
    }

    inline proc getDigit(key: uint, rshift: int, last: bool, negs: bool): int {
      return ((key >> rshift) & (maskDigit:uint)):int;
    }

    // Get the digit for the current rshift. In order to correctly sort
    // negatives, we have to invert the entire key if it's negative, and invert
    // just the signbit for positive values when looking at the last digit.
    inline proc getDigit(in key: real, rshift: int, last: bool, negs: bool): int {
      const invertSignBit = last && negs;
      var keyu: uint;
      memcpy(c_ptrTo(keyu), c_ptrTo(key), numBytes(key.type).safeCast(c_size_t));
      var signbitSet = keyu >> (numBits(keyu.type)-1) == 1;
      var xor = 0:uint;
      if signbitSet {
        keyu = ~keyu;
      } else {
        xor = (invertSignBit:uint << (RSLSD_bitsPerDigit-1));
      }
      return (((keyu >> rshift) & (maskDigit:uint)) ^ xor):int;
    }

    inline proc getDigit(key: 2*uint, rshift: int, last: bool, negs: bool): int {
      const (key0,key1) = key;
      if (rshift >= numBits(uint)) {
        return getDigit(key0, rshift - numBits(uint), last, negs);
      } else {
        return getDigit(key1, rshift, last, negs);
      }
    }

    inline proc getDigit(key: _tuple, rshift: int, last: bool, negs: bool): int
        where isHomogeneousTuple(key) && key.type == key.size*uint(bitsPerDigit) {
      const keyHigh = key.size - 1;
      return key[keyHigh - rshift/bitsPerDigit]:int;
    }

    proc getNumDigitsNumericArrays(names, st: borrowed SymTab) throws {
      var bitWidths: [names.domain] int;
      var negs: [names.domain] bool;
      var totalDigits: int;

      for (bitWidth, name, neg) in zip(bitWidths, names, negs) {
        // TODO checkSorted and exclude array if already sorted?
        var g: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        select g.dtype {
          when DType.Int64   { (bitWidth, neg) = getBitWidth(toSymEntry(g, int ).a); }
          when DType.UInt64  { (bitWidth, neg) = getBitWidth(toSymEntry(g, uint).a); }
          when DType.Float64 { (bitWidth, neg) = getBitWidth(toSymEntry(g, real).a); }
          when DType.Bool { (bitWidth, neg) = getBitWidth(toSymEntry(g, bool).a); }
          otherwise {
            throw getErrorWithContext(
                                      msg=dtype2str(g.dtype),
                                      lineNumber=getLineNumber(),
                                      routineName=getRoutineName(),
                                      moduleName=getModuleName(),
                                      errorClass="TypeError"
                                      );
          }
          }
        totalDigits += (bitWidth + (bitsPerDigit-1)) / bitsPerDigit;
      }
      return (totalDigits, bitWidths, negs);
    }

    proc mergeNumericArrays(param numDigits, size, totalDigits, bitWidths, negs, names, st) throws {
      // check mem limit for merged array and sort on merged array
      const itemsize = numDigits * bitsPerDigit / 8;
      overMemLimit(size*itemsize + radixSortLSD_memEst(size, itemsize));

      var ivname = st.nextName();
      var merged = makeDistArray(size, numDigits*uint(bitsPerDigit));
      var curDigit = numDigits - totalDigits;
      for (name, nBits, neg) in zip(names, bitWidths, negs) {
        var g: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        proc mergeArray(type t) {
          var e = toSymEntry(g, t);
          ref A = e.a;

          const r = 0..#nBits by bitsPerDigit;
          for rshift in r {
            const myDigit = (nBits-1 - rshift) / bitsPerDigit;
            const last = myDigit == 0;
            forall (m, a) in zip(merged, A) {
              m[curDigit+myDigit] =  getDigit(a, rshift, last, neg):uint(bitsPerDigit);
            }
          }
          curDigit += r.size;
        }
        select g.dtype {
          when DType.Int64   { mergeArray(int); }
          when DType.UInt64  { mergeArray(uint); }
          when DType.Float64 { mergeArray(real); }
          when DType.Bool { mergeArray(bool); }
          otherwise {
            throw getErrorWithContext(
                                      msg=dtype2str(g.dtype),
                                      lineNumber=getLineNumber(),
                                      routineName=getRoutineName(),
                                      moduleName=getModuleName(),
                                      errorClass="IllegalArgumentError"
                                      );
          }
        }
      }
      return merged;
    }

    /*
       A localized "slice" of an array. This is meant to be a low-level
       alternative to array slice assignment with better performance (fewer
       allocations, especially when the region is local). When the region being
       sliced is local, a borrowed pointer is stored and `isOwned` is set to
       false. When the region is remote or non-contiguous, memory is copied to
       a local buffer and `isOwned` is true.
     */
    record lowLevelLocalizingSlice {
        type t;
        /* Pointer to localized memory */
        var ptr: c_ptr(t) = nil;
        /* Do we own the memory? */
        var isOwned: bool = false;

        proc init(ref A: [] ?t, region: range()) {
            use CommPrimitives;
            use CTypes;

            this.t = t;
            if region.isEmpty() {
                this.ptr = nil;
                this.isOwned = false;
                return;
            }
            ref start = A[region.low];
            ref end = A[region.high];
            const startLocale = start.locale.id;
            const endLocale = end.locale.id;
            const hereLocale = here.id;

            if contiguousIndices(A) && startLocale == endLocale {
                if startLocale == hereLocale {
                    // If data is contiguous and local, return a borrowed c_ptr
                    this.ptr = c_ptrTo(start);
                    this.isOwned = false;
                } else {
                    // If data is contiguous on a single remote node,
                    // alloc+bulk GET and return owned c_ptr
                    this.ptr = allocate(t, region.size);
                    this.isOwned = true;
                    const byteSize = region.size:c_size_t * c_sizeof(t);
                    GET(ptr, getAddr(start), startLocale, byteSize);
                }
            } else {
                // If data is non-contiguous or split across nodes, get element
                // at a time and return owned c_ptr (slow, expected to be rare)
                this.ptr = allocate(t, region.size);
                this.isOwned = true;
                for i in 0..<region.size {
                    this.ptr[i] = A[region.low + i];
                }
            }
        }

        proc deinit() {
            if isOwned {
                deallocate(ptr);
            }
        }
    }


    /*
      Create a rank 'N' array by removing the degenerate ranks from
      'A' and copying it's contents into the new array

      'N' must be equal to 'A's rank minus the number of degenerate ranks;
      halts if this condition isn't met.

      Analogous to matlab's 'squeeze': https://www.mathworks.com/help/matlab/ref/squeeze.html

      Assumes that the degenerate ranks have an index of 0 (i.e., they have a range of 0..0)
    */
    proc removeDegenRanks(A: [?D] ?t, param N: int) throws
      where N <= D.rank
    {
      var degenRanks: D.rank*bool,
          numDegenRanks: int;

      // determine which, and how many, ranks are degenerate
      for param i in 0..<D.rank {
        if D.shape[i] == 1 {
          degenRanks[i] = true;
          numDegenRanks += 1;
        }
      }

      if N != D.rank - numDegenRanks then
        halt("removeDegenRanks: N must be equal to A's rank minus the number of degenerate ranks");

      // compute the shape of the new array and create a mapping from the
      // new array's ranks to the old array's ranks
      var shape: N*range,
          mapping: N*int,
          i = 0;

      for param ii in 0..<D.rank {
        mapping(i) = ii;
        if !degenRanks[ii] {
          shape[i] = D.dim(ii);
          i += 1;
        }
      }

      // used to map indices from the new array to the old array
      // (assumes that the degenerate ranks have an index of 0, s.t.
      //  the default initialization of 'ret' will index into 'A')
      inline proc map(idx: int ...N): D.rank*int {
        var ret: D.rank*int;
        for param i in 0..<N do
          ret[mapping[i]] = idx[i];
        return ret;
      }

      // create the new array
      var AReduced = makeDistArray({(...shape)}, t);

      // copy values from the old array to the new one
      // TODO: Is this being auto-aggregated? If not, add explicit aggregation
      forall idx in AReduced.domain do
        if N == 1
          then AReduced[idx] = A[map(idx)];
          else AReduced[idx] = A[map((...idx))];

      return AReduced;
    }

    /*
      Get a domain that selects out the idx'th set of indices along the specified axes

      :arg D: the domain to slice
      :arg idx: the index to select along the specified axes (must have the same rank as D)
      :arg axes: the axes to slice along (must be a subset of the axes of D)

      For example, if D represents a stack of 10x10 matrices (ex: {1..10, 1..10, 1..1000})
      Then, domOnAxis(D, (1, 1, 25), 0, 1) will return D sliced with {1..10, 1..10, 25..25}
    */
    proc domOnAxis(D: domain, idx: D.rank*int, axes: int ...?NA): domain
      where NA < D.rank
    {
      var outDims: D.rank*range;
      label ranks for i in 0..<D.rank {
        for param j in 0..<NA {
          if i == axes[j] {
            outDims[i] = D.dim(i);
            continue ranks;
          }
        }
        outDims[i] = idx[i]..idx[i];
      }
      return D[{(...outDims)}];
    }

    /*
      Get a domain over the set of indices orthogonal to the specified axes

      :arg D: the domain to slice
      :arg axes: the axes to slice along (must be a subset of the axes of D)

      For example, if D represents a stack of 10x10 matrices (ex: {1..10, 1..10, 1..1000})
      Then, domOffAxis(D, 0, 1) will return D sliced with {0..0, 0..0, 1..1000}
    */
    proc domOffAxis(D: domain, axes: int ...?NA): domain
      where NA < D.rank
    {
      var outDims: D.rank*range;
      label ranks for i in 0..<D.rank {
        for param j in 0..<NA {
          if i == axes[j] {
            outDims[i] = 0..0;
            continue ranks;
          }
        }
        outDims[i] = D.dim(i);
      }
      return D[{(...outDims)}];
    }

    /*
    Algorithm to determine shape of broadcasted PD array given two array shapes

    see: https://data-apis.org/array-api/latest/API_specification/broadcasting.html#algorithm
    */
    proc broadcastShape(sa: ?Na*int, sb: ?Nb*int, param N: int): N*int throws {
      var s: N*int;
      for param i in 0..<N by -1 do {
        const n1 = Na - N + i,
              n2 = Nb - N + i,
              d1 = if n1 < 0 then 1 else sa[n1],
              d2 = if n2 < 0 then 1 else sb[n2];

        if      d1 == 1  then s[i] = d2;
        else if d2 == 1  then s[i] = d1;
        else if d1 == d2 then s[i] = d1;
        else throw new Error("Incompatible shapes for broadcast");
      }
      return s;
    }

    proc broadcastShape(sa: ?N1*int, sb: ?N2*int): N1*int throws
      where N1 >= N2
        do return broadcastShape(sa, sb, N1);

    proc broadcastShape(sa: ?N1*int, sb: ?N2*int): N2*int throws
      where N1 < N2
        do return broadcastShape(sa, sb, N2);

    proc removeAxis(shape: ?N*int, axis: int): (N-1)*int {
      var s: (N-1)*int,
          i = 0;
      for param ii in 0..<N {
        if ii != axis {
          s[i] = shape[ii];
          i += 1;
        }
      }
      return s;
    }

    proc appendAxis(shape: ?N*int, axis: int, param value: int): (N+1)*int throws {
      var s: (N+1)*int,
          i = 0;
      if axis > N then throw new Error("Axis out of bounds");
      for param ii in 0..<N+1 {
        if ii == axis {
          s[ii] = value;
        } else {
          s[ii] = shape[i];
          i += 1;
        }
      }
      return s;
    }

    proc appendAxis(shape: int, axis: int, param value: int): 2*int throws {
      var s: 2*int;
      if axis == 0 {
        s[0] = value;
        s[1] = shape;
      } else if axis == 1 {
        s[0] = shape;
        s[1] = value;
      } else {
        throw new Error("Axis out of bounds");
      }
      return s;
    }
}
