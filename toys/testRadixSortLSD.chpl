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
    
    proc testIt(nVals: int, nRange:int) {

        var D = newBlockDom({0..#nVals});
        var A: [D] int;

        fillRandom(A, 241);
        [a in A] a = if a<0 then -a else a;
        A %= nRange;
        
        printAry("A = ",A);
        
        var nBits = 64 - clz(max reduce A);
        writeln("nBits = ",nBits);
        var timer:Timer;
        timer.start();
        // sort data
        var iv = radixSortLSD(A);
        timer.stop();
        writeln(">>>Sorted ", nVals, " elements in ", timer.elapsed(), " seconds",
                " (", 8.0*nVals/timer.elapsed()/1024.0/1024.0, " MiB/s)");

        printAry("iv = ", iv);
        var aiv = A[iv];
        printAry("A[iv] = ", aiv);
        writeln(isSorted(aiv));
    }

    proc testSimple() {
        v = true;
        
        // test a simple case
        var D = newBlockDom({0..#8});
        var A: [D] int = [5,7,3,1,4,2,7,2];
        
        printAry("A = ",A);
        
        var nBits = 64 - clz(max reduce A);
        
        var iv = radixSortLSD(A);
        printAry("iv = ", iv);
        var aiv = A[iv];
        printAry("A[iv] = ", aiv);
        writeln(isSorted(aiv));

        v = RSLSD_v;
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
        testIt(NVALS, NRANGE);
    }


}

