/* Radix Sort Least Significant Digit */
module RadixSortLSD
{
    config const RSLSD_v = true;
    var v = RSLSD_v;
    
    config const RSLSD_numTasks = 4; // tasks per locale
    var numTasks = RSLSD_numTasks; // tasks per locale
    var Tasks = {0..#numTasks};
    
    config const RSLSD_bitsPerDigit = 16;
    var bitsPerDigit = RSLSD_bitsPerDigit;
    var numBuckets = 1 << bitsPerDigit;
    var maskDigit = numBuckets-1;

    use BlockDist;
    use BitOps;
    use AryUtil;
    use UnorderedCopy;

    
    inline proc getDigit(v: int, rshift: int): int {
        return ((v >> rshift) & maskDigit);
    }

    // calculate sub-domain for task
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
    
    /* Radix Sort Least Significant Digit
       radix sort a block distributed array
       returning a permutation vector also as a block distributed array */
    proc radixSortLSD(a: [?aD] int): [aD] int {

        // calc max value in bit position
        // *** need to fix this to take into account negative integers
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
                            // kr1[pos] = kr0[i];
                            unorderedCopy(kr1[pos][KEY], kr0[i][KEY]);
                            unorderedCopy(kr1[pos][RANK], kr0[i][RANK]);
                        }
                    }//coforall task 
                }//on loc
            }//coforall loc

            // copy back to kr0 for next iteration
            kr0 = kr1;
            
        }//for rshift

        var ranks: [aD] int;
        var (negVal, firstNegative) = maxloc reduce ([(key, rank) in kr0] ((key < 0), rank));
        if (negVal < 0) {
            [((key, rank), i) in zip(kr0[firstNegative..], aD.low..)] unorderedCopy(ranks[i], rank);
            [((key, rank), i) in zip(kr0[..firstNegative], aD.high-firstNegative+1..)] unorderedCopy(ranks[i], rank);
        } else {
            [((key, rank), i) in zip(kr0, aD)] ranks[i] = rank;
        }

        return ranks;
        
    }//proc radixSortLSD
    
}

