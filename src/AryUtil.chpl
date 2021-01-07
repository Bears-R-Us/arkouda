module AryUtil
{
    use Random;
    use Reflection;
    use Logging;
    use ServerConfig;
    
    const auLogger = new Logger();
    if v {
        auLogger.level = LogLevel.DEBUG;
    } else {
        auLogger.level = LogLevel.INFO;    
    }
    
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
}
