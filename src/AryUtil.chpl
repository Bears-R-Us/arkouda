module AryUtil
{
    use CPtr;
    use Random;
    use Reflection;
    use Logging;
    use ServerConfig;
    use MultiTypeSymbolTable;
    use ServerErrors;
    
    private config const logLevel = ServerConfig.logLevel;
    const auLogger = new Logger(logLevel);
    
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
    proc formatAry(A):string throws {
        if A.size <= printThresh {
            return "%t".format(A);
        } else {
            return "%t ... %t".format(A[A.domain.low..A.domain.low+2],
                                      A[A.domain.high-2..A.domain.high]);
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
    proc concatArrays(a: [?aD] ?t, b: [?bD] t) {
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
        return A.isDefaultRectangular() || isSubtype(A.domain.dist.type, Block);
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
    proc validateArraysSameLength(n:int, fields:[] string, st: borrowed SymTab) throws {
      // Check that fields contains the stated number of arrays
      if (fields.size != 2*n) { 
          var errorMsg = "Expected %i arrays but got %i".format(n, fields.size/2 - 1);
          auLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          throw new owned ErrorWithContext(errorMsg,
                                           getLineNumber(),
                                           getRoutineName(),
                                           getModuleName(),
                                           "ArgumentError");
      }
      const low = fields.domain.low;
      var names = fields[low..#n];
      var types = fields[low+n..#n];
      /* var arrays: [0..#n] borrowed GenSymEntry; */
      var size: int;
      // Check that all arrays exist in the symbol table and have the same size
      var hasStr = false;
      for (name, objtype, i) in zip(names, types, 1..) {
        var thisSize: int;
        select objtype {
          when "pdarray" {
            var g = getGenericTypedArrayEntry(name, st);
            thisSize = g.size;
          }
          when "str" {
            var (myNames, _) = name.splitMsgToTuple('+', 2);
            var g = getSegStringEntry(myNames, st);
            thisSize = g.size;
            hasStr = true;
          }
          when "category" {
            // passed only Categorical.codes.name to be sorted on
            var g = getGenericTypedArrayEntry(name, st);
            thisSize = g.size;
          }
          otherwise {
              var errorMsg = "Unrecognized object type: %s".format(objtype);
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
              var errorMsg = "Arrays must all be same size; expected size %t, got size %t".format(size, thisSize);
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
        var ptr: c_ptr(t) = c_nil;
        /* Do we own the memory? */
        var isOwned: bool = false;

        proc init(A: [] ?t, region: range(?)) {
            use CommPrimitives;
            use SysCTypes;

            this.t = t;
            if region.isEmpty() {
                this.ptr = c_nil;
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
                    this.ptr = c_malloc(t, region.size);
                    this.isOwned = true;
                    const byteSize = region.size:size_t * c_sizeof(t);
                    GET(ptr, startLocale, getAddr(start), byteSize);
                }
            } else {
                // If data is non-contiguous or split across nodes, get element
                // at a time and return owned c_ptr (slow, expected to be rare)
                this.ptr = c_malloc(t, region.size);
                this.isOwned = true;
                for i in 0..<region.size {
                    this.ptr[i] = A[region.low + i];
                }
            }
        }

        proc deinit() {
            if isOwned {
                c_free(ptr);
            }
        }
    }
}
