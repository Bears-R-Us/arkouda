/* arg sort algorithm
these pass back an index vector which can be used
to permute the original array into sorted order */

module ArgSortMsg
{
    use ServerConfig;
    
    use CTypes;

    use Time only;
    use Math only;
    private use Sort;
    use Reflection only;
    
    use PrivateDist;

    use CommAggregation;

    use AryUtil;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use RadixSortLSD;
    use SegmentedString;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    private config const logLevel = ServerConfig.logLevel;
    const asLogger = new Logger(logLevel);

    // thresholds for different sized sorts
    var lgSmall = 10;
    var small = 2**lgSmall;
    var lgMedium = 20;
    var medium = 2**lgMedium;
    var lgLarge = 30;
    var large = 2**lgLarge;

    // thresholds for ranges of values in the sorts
    var sBins = 2**10;
    var mBins = 2**25;
    var lBins = 2**25 * numLocales;

    enum SortingAlgorithm {
      RadixSortLSD,
      TwoArrayRadixSort
    };
    config const defaultSortAlgorithm: SortingAlgorithm = SortingAlgorithm.RadixSortLSD;

    // proc DefaultComparator.keyPart(x: _tuple, i:int) where !isHomogeneousTuple(x) &&
    // (isInt(x(0)) || isUint(x(0)) || isReal(x(0))) {
    
    import Reflection.canResolveMethod;
    record ContrivedComparator {
      const dc = new DefaultComparator();
      proc keyPart(a, i: int) {
        if canResolveMethod(dc, "keyPart", a, 0) {
          return dc.keyPart(a, i);
        } else if isTuple(a) {
          return tupleKeyPart(a, i);
        } else {
          compilerError("No keyPart method for eltType ", a.type:string);
        }
      }
      proc tupleKeyPart(x: _tuple, i:int) {
        proc makePart(y): uint(64) {
          var part: uint(64);
          // get the part, ignore the section
          const p = dc.keyPart(y, 0)(1);
          // assuming result of keyPart is int or uint <= 64 bits
          part = p:uint(64); 
          // If the number is signed, invert the top bit, so that
          // the negative numbers sort below the positive numbers
          if isInt(p) {
            const one:uint(64) = 1;
            part = part ^ (one << 63);
          }
          return part;
        }
        var part: uint(64);
        if isTuple(x[0]) && (x.size == 2) {
          for param j in 0..<x[0].size {
            if i == j {
              part = makePart(x[0][j]);
            }
          }
          if i == x[0].size {
            part = makePart(x[1]);
          }
          if i > x[0].size {
            return (-1, 0:uint(64));
          } else {
            return (0, part);
          }
        } else {
          for param j in 0..<x.size {
            if i == j {
              part = makePart(x[j]);
            }
          }
          if i >= x.size {
            return (-1, 0:uint(64));
          } else {
            return (0, part);
          }
        }
      }
    }
    
    const myDefaultComparator = new ContrivedComparator();

    /* Perform one step in a multi-step argsort, starting with an initial 
       permutation vector and further permuting it in the manner required
       to sort an array of keys.
     */
    proc incrementalArgSort(g: GenSymEntry, iv: [?aD] int): [] int throws {
      // Store the incremental permutation to be applied on top of the initial perm
      var deltaIV: [aD] int;
      // Discover the dtype of the entry holding the keys array
      select g.dtype {
          when DType.Int64 {
              var e = toSymEntry(g, int);
              // Permute the keys array with the initial iv
              var newa: [e.aD] int;
              ref olda = e.a;
              // Effectively: newa = olda[iv]
              forall (newai, idx) in zip(newa, iv) with (var agg = newSrcAggregator(int)) {
                  agg.copy(newai, olda[idx]);
              }
              // Generate the next incremental permutation
              deltaIV = argsortDefault(newa);
          }
          when DType.UInt64 {
              var e = toSymEntry(g, uint);
              // Permute the keys array with the initial iv
              var newa: [e.aD] uint;
              ref olda = e.a;
              // Effectively: newa = olda[iv]
              forall (newai, idx) in zip(newa, iv) with (var agg = newSrcAggregator(uint)) {
                  agg.copy(newai, olda[idx]);
              }
              // Generate the next incremental permutation
              deltaIV = argsortDefault(newa);
          }
          when DType.Float64 {
              var e = toSymEntry(g, real);
              var newa: [e.aD] real;
              ref olda = e.a;
              forall (newai, idx) in zip(newa, iv) with (var agg = newSrcAggregator(real)) {
                  agg.copy(newai, olda[idx]);
              }
              deltaIV = argsortDefault(newa);
          }
          otherwise { throw getErrorWithContext(
                                msg="Unsupported DataType: %t".format(dtype2str(g.dtype)),
                                lineNumber=getLineNumber(),
                                routineName=getRoutineName(),
                                moduleName=getModuleName(),
                                errorClass="IllegalArgumentError"
                                ); 
          }
      }
      // The output permutation is the composition of the initial and incremental permutations
      var newIV: [aD] int;
      // Effectively: newIV = iv[deltaIV] 
      forall (newIVi, idx) in zip(newIV, deltaIV) with (var agg = newSrcAggregator(int)) {
        agg.copy(newIVi, iv[idx]);
      }
      return newIV;
    }

    proc incrementalArgSort(s: SegString, iv: [?aD] int): [] int throws {
      var hashes = s.siphash();
      var newHashes: [aD] 2*uint;
      forall (nh, idx) in zip(newHashes, iv) with (var agg = newSrcAggregator((2*uint))) {
        agg.copy(nh, hashes[idx]);
      }
      var deltaIV = argsortDefault(newHashes);
      // var (newOffsets, newVals) = s[iv];
      // var deltaIV = newStr.argGroup();
      var newIV: [aD] int;
      forall (newIVi, idx) in zip(newIV, deltaIV) with (var agg = newSrcAggregator(int)) {
        agg.copy(newIVi, iv[idx]);
      }
      return newIV;
    }

    /* Do a LSD radix sort across multiple arrays, where each array represents a digit.
     */
    /* proc coArgSort(arrays: [?D] GenSymEntry): [] int throws { */
    /*   // Calling function already checked that all arrays have same size */
    /*   const aD = makeDistDom(arrays[D.low].size); */
    /*   // Initialize permutation to the identity */
    /*   var cumulativeIV: [aD] int = aD.low..aD.high; */
    /*   // Starting with the last array, incrementally permute the IV by sorting each array */
    /*   for i in D.low..D.high-1 by -1 { */
    /*         try cumulativeIV = incrementalArgSort(arrays[i], cumulativeIV); */
    /*   } */
    /*   return cumulativeIV; */
    /* } */

    /* Find the permutation that sorts multiple arrays, treating each array as a
       new level of the sorting key.
     */
    proc coargsortMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      param pn = Reflection.getRoutineName();
      var repMsg: string;
      var (algoName, nstr, rest) = payload.splitMsgToTuple(3);
      var algorithm: SortingAlgorithm = defaultSortAlgorithm;
      if algoName != "" {
        try {
          algorithm = algoName: SortingAlgorithm;
        } catch {
          throw getErrorWithContext(
                                    msg="Unrecognized sorting algorithm: %s".format(algoName),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="NotImplementedError"
                                    );
        }
      }
      var n = nstr:int; // number of arrays to sort
      var fields = rest.split();
      asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                  "number of arrays: %i fields: %t".format(n,fields));
      var (size, hasStr, names, types) = validateArraysSameLength(n, fields, st);

      // If there were no string arrays, merge the arrays into a single array and sort
      // that. This eliminates having to merge index vectors, but has a memory overhead
      // and increases the size of the comm we have to do since the KEY is larger). We
      // merge the elements into a `uint(RSLSD_bitsPerDigit)` tuple. This wastes space
      // (e.g. when merging 2 arrays that use 7 and 9 bits), but it allows us to use
      // `getDigit`, which changes the bit patterns to correctly sort negatives. We
      // consider tuple[1] to be the most significant digit.
      //
      // TODO support string? This further increases size (128-bits for each hash), so we
      // need to be OK with memory overhead and comm from the KEY)
      if !hasStr {
        var (totalDigits, bitWidths, negs) = getNumDigitsNumericArrays(names, st);

        // TODO support arbitrary size with array-of-arrays or segmented array
        proc mergedArgsort(param numDigits) throws {

          // check mem limit for merged array and sort on merged array
          const itemsize = numDigits * bitsPerDigit / 8;
          overMemLimit(size*itemsize + radixSortLSD_memEst(size, itemsize));

          var ivname = st.nextName();
          var merged = mergeNumericArrays(numDigits, size, totalDigits, bitWidths, negs, names, st);

          var iv = argsortDefault(merged, algorithm=algorithm);
          st.addEntry(ivname, new shared SymEntry(iv));

          var repMsg = "created " + st.attrib(ivname);
          asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
          return repMsg;
        }

        // Since we're using tuples, we have to stamp out for each size we want to
        // support. For now support 8, 16, and 32 byte sorting.
        if totalDigits <=  4 { return new MsgTuple(mergedArgsort( 4), MsgType.NORMAL); }
        if totalDigits <=  8 { return new MsgTuple(mergedArgsort( 8), MsgType.NORMAL); }
        if totalDigits <= 16 { return new MsgTuple(mergedArgsort(16), MsgType.NORMAL); }
      }

      // check mem limit for permutation vectors and sort
      const itemsize = numBytes(int);
      overMemLimit(2*size*itemsize + radixSortLSD_memEst(size, itemsize));
      
      // Initialize the permutation vector in the symbol table with the identity perm
      var rname = st.nextName();
      st.addEntry(rname, size, int);
      var iv = toSymEntry(getGenericTypedArrayEntry(rname, st), int);
      iv.a = 0..#size;
      // Starting with the last array, incrementally permute the IV by sorting each array
      for (i, j) in zip(names.domain.low..names.domain.high by -1,
                        types.domain.low..types.domain.high by -1) {
        if (types[j] == "str") {
          var strings = getSegString(names[i], st);
          iv.a = incrementalArgSort(strings, iv.a);
        } else {
          var g: borrowed GenSymEntry = getGenericTypedArrayEntry(names[i], st);
          // Perform the coArgSort and store in the new SymEntry
          iv.a = incrementalArgSort(g, iv.a);
        }
      }
      repMsg = "created " + st.attrib(rname);
      asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
    }
    
    proc argsortDefault(A:[?D] ?t, algorithm:SortingAlgorithm=defaultSortAlgorithm):[D] int throws {
      var t1 = Time.getCurrentTime();
      var iv: [D] int;
      select algorithm {
        when SortingAlgorithm.TwoArrayRadixSort {
          var AI = [(a, i) in zip(A, D)] (a, i);
          Sort.TwoArrayRadixSort.twoArrayRadixSort(AI, comparator=myDefaultComparator);
          iv = [(a, i) in AI] i;
        }
        when SortingAlgorithm.RadixSortLSD {
          iv = radixSortLSD_ranks(A);
        }
        otherwise {
          throw getErrorWithContext(
                                    msg="Unrecognized sorting algorithm: %s".format(algorithm:string),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="NotImplementedError"
                  );
        }
      }
      try! asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                             "argsort time = %i".format(Time.getCurrentTime() - t1));
      return iv;
    }
    
    /* argsort takes pdarray and returns an index vector iv which sorts the array */
    proc argsortMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (algoName, objtype, name) = payload.splitMsgToTuple(3);
        var algorithm: SortingAlgorithm = defaultSortAlgorithm;
        if algoName != "" {
          try {
            algorithm = algoName: SortingAlgorithm;
          } catch {
            throw getErrorWithContext(
                                    msg="Unrecognized sorting algorithm: %s".format(algoName),
                                    lineNumber=getLineNumber(),
                                    routineName=getRoutineName(),
                                    moduleName=getModuleName(),
                                    errorClass="NotImplementedError"
                  );
          }
        }
        // get next symbol name
        var ivname = st.nextName();
        asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                              "cmd: %s name: %s ivname: %s".format(cmd, name, ivname));

        select objtype {
          when "pdarray" {
            var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
            // check and throw if over memory limit
            overMemLimit(radixSortLSD_memEst(gEnt.size, gEnt.itemsize));
        
            select (gEnt.dtype) {
                when (DType.Int64) {
                    var e = toSymEntry(gEnt,int);
                    var iv = argsortDefault(e.a, algorithm=algorithm);
                    st.addEntry(ivname, new shared SymEntry(iv));
                }
                when (DType.UInt64) {
                    var e = toSymEntry(gEnt,uint);
                    var iv = argsortDefault(e.a, algorithm=algorithm);
                    st.addEntry(ivname, new shared SymEntry(iv));
                }
                when (DType.Float64) {
                    var e = toSymEntry(gEnt, real);
                    var iv = argsortDefault(e.a);
                    st.addEntry(ivname, new shared SymEntry(iv));
                }
                otherwise {
                    var errorMsg = notImplementedError(pn,gEnt.dtype);
                    asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);               
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
            }
          }
          when "str" {
            var strings = getSegString(name, st);
            // check and throw if over memory limit
            overMemLimit((8 * strings.size * 8)
                         + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
            var iv = strings.argsort();
            st.addEntry(ivname, new shared SymEntry(iv));
          }
          otherwise {
              var errorMsg = notImplementedError(pn, objtype);
              asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                    
              return new MsgTuple(errorMsg, MsgType.ERROR);
          }
        }

        repMsg = "created " + st.attrib(ivname);
        asLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("argsort", argsortMsg, getModuleName());
    registerFunction("coargsort", coargsortMsg, getModuleName());
}
