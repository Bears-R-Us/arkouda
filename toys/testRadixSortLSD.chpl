module testRadixSortLSD
{
    use BlockDist;
    use BitOps;
    use AryUtil;

    use Time;
    use Random;

    use RadixSortLSD;
    
    config const NVALS = 10;
    config const NRANGE = 10;
    
    proc testIt(nVals: int, nRange: int, posOnly: bool, type t) {

        var D = newBlockDom({0..#nVals});
        var A: [D] t;

        fillRandom(A, 241);
        if isIntegral(t) {
          if posOnly {
            [a in A] a = if a < 0 then -a else a;
          }
          A %= nRange;
        } else if isRealType(t) {
          if !posOnly {
            A = 2*A - 1;
          }
          A *= nRange;
        }
        
        printAry("A = ",A);
        writeln(">> radixSortLSD_ranks");
        var timer:Timer;
        timer.clear();
        timer.start();
        // sort data
        var iv = radixSortLSD_ranks(A);
        timer.stop();
        writeln(">>>Sorted ", nVals, " elements in ", timer.elapsed(), " seconds",
                " (", 8.0*nVals/timer.elapsed()/1024.0/1024.0, " MiB/s)");

        printAry("iv = ", iv);
        var aiv = A[iv];
        printAry("A[iv] = ", aiv);
        writeln(isSorted(aiv));

        writeln(">> radixSortLSD_keys");
        timer.clear();
        timer.start();
        // sort data
        var sorted = radixSortLSD_keys(A);
        timer.stop();
        writeln(">>>Sorted ", nVals, " elements in ", timer.elapsed(), " seconds",
                " (", 8.0*nVals/timer.elapsed()/1024.0/1024.0, " MiB/s)");

        printAry("sorted = ", sorted);
        writeln(isSorted(sorted));

        
    }

    proc testSimple() {
        vv = true;
        
        // test a simple case
        var D = newBlockDom({0..#8});
        var A: [D] int = [5,7,3,1,4,2,7,2];
        
        printAry("A = ",A);
        
        var nBits = 64 - clz(max reduce A);
        
        var iv = radixSortLSD_ranks(A);
        printAry("iv = ", iv);
        var aiv = A[iv];
        printAry("A[iv] = ", aiv);
        writeln(isSorted(aiv));

        var sorted = radixSortLSD_keys(A);
        printAry("sorted = ", sorted);
        writeln(isSorted(sorted));

        vv = RSLSD_vv;
    }
    
    proc main() {
        writeln("RSLSD_numTasks = ",RSLSD_numTasks);
        writeln("numTasks = ",numTasks);
        writeln("Tasks = ",Tasks);
        writeln("RSLSD_bitsPerDigit = ",RSLSD_bitsPerDigit);
        writeln("bitsPerDigit = ",bitsPerDigit);
        writeln("numBuckets = ",numBuckets);
        writeln("maskDigit = ",maskDigit);

        testSimple();
        writeln("Testing positive int");
        testIt(NVALS, NRANGE,true, int);
        writeln("Testing negative int");
        testIt(NVALS, NRANGE,false, int);
        writeln("Testing positive real");
        testIt(NVALS, NRANGE, true, real);
        writeln("Testing negative real");
        testIt(NVALS, NRANGE, false, real);
    }


}

