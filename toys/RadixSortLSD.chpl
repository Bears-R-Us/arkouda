/* Radix Sort Least Significant Digit */
module RadixSortLSD
{
    config const RSLSD_v = true;
    var v = RSLSD_v;
    
    config const RSLSD_numTasks = 4; // tasks per locale
    const numTasks = RSLSD_numTasks; // tasks per locale
    const Tasks = {0..#numTasks};
    
    config const RSLSD_bitsPerDigit = 1;
    const bitsPerDigit = RSLSD_bitsPerDigit;
    const numBuckets = 1 << bitsPerDigit;
    const maskDigit = numBuckets-1;

    use BlockDist;
    use BitOps;
    use Time;
    use AryUtil;


    inline proc getDigit(v: int, rshift: int): int {
        return ((v >> rshift) & maskDigit);
    }

    // very close but still not quite right when high is less than low
    inline proc calcBlock(task: int, low: int, high: int) {
        var totalsize = high - low + 1;
        var div = totalsize / numTasks;
        var rem = totalsize % numTasks;
        var rlow: int;
        var rhigh: int;
        if (task < rem) {
            rlow = task * (div+1) + low;
            rhigh = rlow + div;
        }
        else {
            rlow = task * div + rem + low;
            rhigh = rlow + div - 1;
        }
        return {rlow .. rhigh};
    }

    // calc global transposed index
    // (bucket,loc,task) = (bucket * numLocales * numTasks) + (loc * numTasks) + task;
    inline proc calcGlobalIndex(bucket: int, loc: int, task: int): int {
        return ((bucket * numLocales * numTasks) + (loc * numTasks) + task);
    }
    
    /* radix sort a block distributed array returning a permutation vector */
    proc radixSortLSD(a: [?aD] int): [aD] int {

        // calc max value in bit position
        var nBits = 64 - clz(max reduce a);
        
        // form (key,rank) vector
        param KEY = 1;
        param RANK = 2;
        var kr0: [aD] (int,int) = [(key,rank) in zip(a,aD)] (key,rank);
        var kr1: [aD] (int,int);

        // create a global count array to scan
        var gD = newBlockDom({0..#(numLocales * numTasks * numBuckets)});
        var globalCounts: [gD] int;
        var globalEnds: [gD] int;
        var globalStarts: [gD] int;
        
        // loop over digits
        for rshift in {0..#nBits by bitsPerDigit} {
            if v {writeln("rshift = ",rshift);}
            // count digits
            coforall loc in Locales {
                on loc {
                    coforall task in Tasks {
                        // bucket domain
                        var bD = {0..#numBuckets};
                        // allocate counts
                        var taskBucketCounts: [bD] int;
                        // get local domain's indices
                        var lD = kr0.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        if v {writeln((loc.id,task,tD));}
                        // count digits in this task's part of the array
                        for i in tD {
                            var bucket = getDigit(kr0[i][KEY], rshift); // calc bucket from key
                            taskBucketCounts[bucket] += 1;
                        }
                        // write counts in to global counts in transposed order
                        for bucket in bD {
                            globalCounts[calcGlobalIndex(bucket, loc.id, task)] = taskBucketCounts[bucket];
                        }
                    }//coforall task
                }//on loc
            }//coforall loc
            
            // scan globalCounts to get bucket ends on each locale/task
            globalEnds = + scan globalCounts;
            globalStarts = globalEnds - globalCounts;
            
            if v {printAry("globalCounts =",globalCounts);try! stdout.flush();}
            if v {printAry("globalEnds =",globalEnds);try! stdout.flush();}
            if v {printAry("globalStarts =",globalStarts);try! stdout.flush();}
            
            // calc new positions and permute
            coforall loc in Locales {
                on loc {
                    coforall task in Tasks {
                        // bucket domain
                        var bD = {0..#numBuckets};
                        // allocate counts
                        var taskBucketPos: [bD] int;
                        // get local domain's indices
                        var lD = kr0.localSubdomain();
                        // calc task's indices from local domain's indices
                        var tD = calcBlock(task, lD.low, lD.high);
                        // read start pos in to globalStarts back from transposed order
                        for bucket in bD {
                            taskBucketPos[bucket] = globalStarts[calcGlobalIndex(bucket, loc.id, task)];
                        }
                        // calc new position and put (key,rank) pair there in kr1
                        for i in tD {
                            var bucket = getDigit(kr0[i][KEY], rshift); // calc bucket from key
                            var pos = taskBucketPos[bucket];
                            taskBucketPos[bucket] += 1;
                            kr1[pos] = kr0[i];
                        }
                    }//coforall task 
                }//on loc
            }//coforall loc

            // copy back to kr0 for next iteration
            kr0 = kr1;
            
        }//for digit

        var ranks: [aD] int = [(key,rank) in kr0] rank;

        return ranks;
        
    }//proc radixSortLSD


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

