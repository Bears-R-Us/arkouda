module AryUtil
{
    use CPtr;
    use Random;
    use Reflection;
    use Logging;
    use ServerConfig;
    
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
      Determines if the passed array array maps contiguous indices to
      contiguous memory.

      :arg A: array to check
    */
    proc contiguousIndices(A: []) param {
        use BlockDist;
        return A.isDefaultRectangular() || isSubtype(A.domain.dist.type, Block);
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
