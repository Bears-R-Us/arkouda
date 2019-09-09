module testSplitRadixSort
{
    use Random;
    use BlockDist;
    use BitOps;
    use Time;

    module AryUtil
    {
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
        proc printAry(name:string, A) {
            if A.size <= printThresh {writeln(name,A);}
            else {writeln(name,[i in A.domain.low..A.domain.low+2] A[i],
                          " ... ", [i in A.domain.high-2..A.domain.high] A[i]);}
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
    }
    
    use AryUtil;
    
    use SplitRadixSort;
    
    config const NVALS = 10;
    config const NRANGE = 10;
    
    proc testIt(nVals: int, nRange:int) {

        var D = newBlockDom({0..#nVals});
        var A: [D] int;

        var R = new owned RandomStream(real, 241); R.getNext();
        for a in A { a = (R.getNext() * nRange):int; }

        printAry("A = ",A);
        
        var nBits = 64 - clz(max reduce A);
        writeln("nBits = ",nBits);
        var timer:Timer;
        timer.start();
        // sort data
        var (sorted, iv) = splitRadixSort(A, nBits);
        timer.stop();
        writeln(">>>Sorted ", nVals, " elements in ", timer.elapsed(), " seconds",
                " (", 8.0*nVals/timer.elapsed()/1024.0/1024.0, " MiB/s)");

        printAry("sorted = ", sorted);
        printAry("iv = ", iv);
        var aiv = A[iv];
        printAry("A[iv] = ", aiv);
        writeln(isSorted(aiv));
    }

    proc testSimple() {
        // test a simple case
        var D = newBlockDom({0..#8});
        var A: [D] int = [5,7,3,1,4,2,7,2];
        
        printAry("A = ",A);
        
        var nBits = 64 - clz(max reduce A);
        
        var (sorted, iv) = splitRadixSort(A, nBits);
        printAry("sorted = ", sorted);
        printAry("iv = ", iv);
        var aiv = A[iv];
        printAry("A[iv] = ", aiv);
        writeln(isSorted(aiv));
    }
    
    proc main() {
        testSimple();
        testIt(NVALS, NRANGE);
    }
}
