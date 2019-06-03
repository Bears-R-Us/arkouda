// compile line:
// chpl --fast -senableParScan --cache-remote --print-passes CountingSort.chpl   

module CountingSort
{

    config const N = 100_000;
    config const NVALS = 8192;
    
    var v = true;
    
    use Time only;
    use Math only;
    use Random;
    
    use BlockDist;
    use PrivateDist;

    use UnorderedCopy;
    use UnorderedAtomics;

    var printThresh = 50;
    
    proc printAry(name:string, A) {
        if A.size <= printThresh {writeln(name,A);}
        else {writeln(name,[i in A.domain.low..A.domain.low+4] A[i],
                      " ... ", [i in A.domain.high-4..A.domain.high] A[i]);}
    }

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

    proc aStats(a: [?D] int): (int,int,real,real,real) {
        var aMin:int = min reduce a;
        var aMax:int = max reduce a;
        var aMean:real = (+ reduce a:real) / a.size:real;
        var aVari:real = (+ reduce (a:real **2) / a.size:real) - aMean**2;
        var aStd:real = sqrt(aVari);
        return (aMin,aMax,aMean,aVari,aStd);
    }

    // defined for reduction and scan on atomics
    proc +(x: atomic int, y: atomic int) {
        return x.read() + y.read();
    }
    
    // defined for reduction and scan on atomics
    proc +=(X: [?D] int, Y: [D] atomic int) {
        [i in D] {X[i] += Y[i].read();}
    }

    // do a counting sort on a (an array of integers)
    // returns iv an array of indices that would sort the array original array
    proc argCountSortGlobHist(a: [?aD] int, aMin: int, aMax: int): [aD] int {
        // index vector to hold permutation
        var iv: [aD] int;

        // how many bins in histogram
        var bins = aMax-aMin+1;
        if v {try! writeln("bins = %t".format(bins));}

        // histogram domain size should be equal to a_nvals
        var hD = newBlockDom({0..#bins});

        // atomic histogram
        var atomic_hist: [hD] atomic int;

        // normal histogram for + scan
        var hist: [hD] int;

        // count number of each value into atomic histogram
        //[e in a] atomic_hist[e-aMin].add(1);
        [e in a] atomic_hist[e-aMin].unorderedAdd(1);
        
        // copy from atomic histogram to normal histogram
        [(e,ae) in zip(hist, atomic_hist)] e = ae.read();
        if v {printAry("hist =",hist);}

        // calc starts and ends of buckets
        var ends: [hD] int = + scan hist;
        if v {printAry("ends =",ends);}
        var starts: [hD] int = ends - hist;
        if v {printAry("starts =",starts);}

        // atomic position in output array for buckets
        var atomic_pos: [hD] atomic int;
        
        // copy in start positions
        [(ae,e) in zip(atomic_pos, starts)] ae.write(e);

        // permute index vector
        forall (e,i) in zip(a,aD) {
            var pos = atomic_pos[e-aMin].fetchAdd(1);// get position to deposit element
            //iv[pos] = i;
            var idx = i;
            unorderedCopy(iv[pos], idx);
        }
        
        // return the index vector
        return iv;
    }

    // do a counting sort on a (an array of integers)
    // returns iv an array of indices that would sort the array original array
    proc argCountSortLocHistGlobHist(a: [?aD] int, aMin: int, aMax: int): [aD] int {
        // index vector to hold permutation
        var iv: [aD] int;

        // how many bins in histogram
        var bins = aMax-aMin+1;
        if v {try! writeln("bins = %t".format(bins));}

        // create a global count array to scan
        var globalCounts = newBlockArr({0..#(bins * numLocales)}, int);

        coforall loc in Locales {
            on loc {
                // histogram domain size should be equal to bins
                var hD = {0..#bins};

                // atomic histogram
                var atomicHist: [hD] atomic int;

                // count number of each value into local atomic histogram
                [i in a.localSubdomain()] atomicHist[a[i]-aMin].add(1);

                // put counts into globalCounts array
                [i in hD] globalCounts[i * numLocales + here.id] = atomicHist[i].read();
            }
        }

        // scan globalCounts to get bucket ends on each locale
        var globalEnds: [globalCounts.domain] int = + scan globalCounts;
        if v {printAry("globalCounts =",globalCounts);try! stdout.flush();}
        if v {printAry("globalEnds =",globalEnds);try! stdout.flush();}
        
        coforall loc in Locales {
            on loc {
                // histogram domain size should be equal to bins
                var hD = {0..#bins};
                var localCounts: [hD] int;
                [i in hD] localCounts[i] = globalCounts[i * numLocales + here.id];
                var localEnds: [hD] int = + scan localCounts;
                
                // atomic histogram
                var atomicHist: [hD] atomic int;

                // local storage to sort into
                var localBuffer: [0..#(a.localSubdomain().size)] int;

                // put locale-bucket-ends into atomic hist
                [i in hD] atomicHist[i].write(localEnds[i] - localCounts[i]);
                
                // get position in localBuffer of each element and place it there
                // counting up to local-bucket-end
                [idx in a.localSubdomain()] {
                    var pos = atomicHist[a[idx]-aMin].fetchAdd(1); // local pos in localBuffer
                    localBuffer[pos] = idx; // should be local pos and global idx
                }

                // move blocks to output array
                [i in hD] {
                    var gEnd = globalEnds[i * numLocales + here.id];
                    var gHigh = gEnd - 1;
                    var gLow =  gEnd - localCounts[i];
                    var lHigh = localEnds[i] - 1;
                    var lLow = localEnds[i] - localCounts[i];
                    if (gLow..gHigh).size != (lLow..lHigh).size {
                        writeln(gLow..gHigh, " ", lLow..lHigh);
                        writeln((gLow..gHigh).size, " != ", (lLow..lHigh).size);
                        try! stdout.flush();
                        exit(1);
                    }
                    if localCounts[i] > 0 {iv[gLow..gHigh] = localBuffer[lLow..lHigh];}
                }
            }
        }
        
        // return the index vector
        return iv;
    }
    
    // do a counting sort on a (an array of integers)
    // returns iv an array of indices that would sort the array original array
    // PD == PrivateDist
    // IW == Indirect write to local array then block copy to output array
    proc argCountSortLocHistGlobHistPDIW(a: [?aD] int, aMin: int, aMax: int): [aD] int {
        // index vector to hold permutation
        var iv: [aD] int;

        // how many bins in histogram
        var bins = aMax-aMin+1;
        if v {try! writeln("bins = %t".format(bins));}

        // create a global count array to scan
        var globalCounts = newBlockArr({0..#(bins * numLocales)}, int);

        // histogram domain size should be equal to bins
        var hD = {0..#bins};
        
        // atomic histogram
        var atomicHist: [PrivateSpace] [hD] atomic int;
        
        // start timer
        var t1 = Time.getCurrentTime();
        // count number of each value into local atomic histogram
        [val in a] atomicHist[here.id][val-aMin].add(1);
        if v {writeln("done atomicHist time = ",Time.getCurrentTime() - t1);try! stdout.flush();}

        // start timer
        t1 = Time.getCurrentTime();
        coforall loc in Locales {
            on loc {
                // put counts into globalCounts array
                [i in hD] globalCounts[i * numLocales + here.id] = atomicHist[here.id][i].read();
            }
        }
        if v {writeln("done copy to globalCounts time = ",Time.getCurrentTime() - t1);try! stdout.flush();}

        // scan globalCounts to get bucket ends on each locale
        var globalEnds: [globalCounts.domain] int = + scan globalCounts;
        if v {printAry("globalCounts =",globalCounts);try! stdout.flush();}
        if v {printAry("globalEnds =",globalEnds);try! stdout.flush();}

        var localCounts: [PrivateSpace] [hD] int;
        var localEnds: [PrivateSpace] [hD] int;
        
        // start timer
        t1 = Time.getCurrentTime();
        coforall loc in Locales {
            on loc {
                [i in hD] localCounts[here.id][i] = globalCounts[i * numLocales + here.id];
            }
        }
        if v {writeln("done copy back to localCounts time = ",Time.getCurrentTime() - t1);try! stdout.flush();}
        
        // start timer
        t1 = Time.getCurrentTime();
        coforall loc in Locales {
            on loc {
                localEnds[here.id] = + scan localCounts[here.id];

                // put locale-bucket-ends into atomic hist
                [i in hD] atomicHist[here.id][i].write(localEnds[here.id][i] - localCounts[here.id][i]);
                
                // local storage to sort into
                var localBuffer: [0..#(a.localSubdomain().size)] int;

                // get position in localBuffer of each element and place it there
                // counting up to local-bucket-end
                [idx in a.localSubdomain()] {
                    var pos = atomicHist[here.id][a[idx]-aMin].fetchAdd(1); // local pos in localBuffer
                    localBuffer[pos] = idx; // should be local pos and global idx
                }

                // move blocks to output array
                [i in hD] {
                    var gEnd = globalEnds[i * numLocales + here.id];
                    var gHigh = gEnd - 1;
                    var gLow =  gEnd - localCounts[here.id][i];
                    var lHigh = localEnds[here.id][i] - 1;
                    var lLow = localEnds[here.id][i] - localCounts[here.id][i];
                    if (gLow..gHigh).size != (lLow..lHigh).size {
                        writeln(gLow..gHigh, " ", lLow..lHigh);
                        writeln((gLow..gHigh).size, " != ", (lLow..lHigh).size);
                        try! stdout.flush();
                        exit(1);
                    }
                    if localCounts[here.id][i] > 0 {iv[gLow..gHigh] = localBuffer[lLow..lHigh];}
                }
            }
        }
        if v {writeln("done sort locally and move segments time = ",Time.getCurrentTime() - t1);try! stdout.flush();}
        
        // return the index vector
        return iv;
    }
    
    // do a counting sort on a (an array of integers)
    // returns iv an array of indices that would sort the array original array
    // PD = PrivateDist
    // DW = Direct Write into output array
    proc argCountSortLocHistGlobHistPDDW(a: [?aD] int, aMin: int, aMax: int): [aD] int {
        // index vector to hold permutation
        var iv: [aD] int;

        // how many bins in histogram
        var bins = aMax-aMin+1;
        if v {try! writeln("bins = %t".format(bins));}

        // create a global count array to scan
        var globalCounts = newBlockArr({0..#(bins * numLocales)}, int);

        // histogram domain size should be equal to bins
        var hD = {0..#bins};
        
        // atomic histogram
        var atomicHist: [PrivateSpace] [hD] atomic int;
        
        // start timer
        var t1 = Time.getCurrentTime();
        // count number of each value into local atomic histogram
        [val in a] atomicHist[here.id][val-aMin].add(1);
        if v {writeln("done atomicHist time = ",Time.getCurrentTime() - t1);try! stdout.flush();}

        // start timer
        t1 = Time.getCurrentTime();
        coforall loc in Locales {
            on loc {
                // put counts into globalCounts array
                [i in hD] globalCounts[i * numLocales + here.id] = atomicHist[here.id][i].read();
            }
        }
        if v {writeln("done copy to globalCounts time = ",Time.getCurrentTime() - t1);try! stdout.flush();}

        // scan globalCounts to get bucket ends on each locale
        var globalEnds: [globalCounts.domain] int = + scan globalCounts;
        if v {printAry("globalCounts =",globalCounts);try! stdout.flush();}
        if v {printAry("globalEnds =",globalEnds);try! stdout.flush();}

        var localCounts: [PrivateSpace] [hD] int;
        
        // start timer
        t1 = Time.getCurrentTime();
        coforall loc in Locales {
            on loc {
                [i in hD] localCounts[here.id][i] = globalCounts[i * numLocales + here.id];
            }
        }
        if v {writeln("done copy back to localCounts time = ",Time.getCurrentTime() - t1);try! stdout.flush();}
        
        // start timer
        t1 = Time.getCurrentTime();
        coforall loc in Locales {
            on loc {
                // put locale-subbin-starts into atomic hist
                [i in hD] atomicHist[here.id][i].write(globalEnds[i * numLocales + here.id] - localCounts[here.id][i]);
            }
        }
        if v {writeln("done init atomic counts time = ",Time.getCurrentTime() - t1);try! stdout.flush();}
        
        // start timer
        t1 = Time.getCurrentTime();
        coforall loc in Locales {
            on loc {
                // fetch-and-inc to get per-locale-subbin-position
                // and directly write index to output array
                forall i in a.localSubdomain() {
                    var idx = i;
                    var pos = atomicHist[here.id][a[idx]-aMin].fetchAdd(1); // local pos in localBuffer
                    unorderedCopy(iv[pos],idx); // iv[pos] = idx; // should be global pos and global idx
                }
            }
        }
        if v {writeln("done move time = ",Time.getCurrentTime() - t1);try! stdout.flush();}

        // return the index vector
        return iv;
    }

    // fill a with integers from interval aMin..(aMax-1)
    proc fillRandInt(a: [?aD] int, aMin: int, aMax: int) {

        var t1 = Time.getCurrentTime();
        coforall loc in Locales {
            on loc {
                var R = new owned RandomStream(real); R.getNext();
                [i in a.localSubdomain()] a[i] = (R.getNext() * (aMax - aMin) + aMin):int;
            }
        }
        writeln("compute time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();

    }

    proc test_argsort(n: int, nVals: int) {
        const blockDom = newBlockDom({0..#n});
        var a: [blockDom] int;

        var aMin = 0;
        var aMax = nVals-1;
        writeln((n, nVals));

        // fill an array with random ints from a range
        fillRandInt(a, aMin, aMax+1);

        printAry("a = ", a);

        {
            writeln(">>> argCountSortGlobHist");
            var t1 = Time.getCurrentTime();
            // returns permutation vector
            var iv = argCountSortGlobHist(a,aMin,aMax);
            writeln("sort time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
            
            // permute a into sorted order
            var sorted: [a.domain] int = a[iv];
            printAry("sorted = ", sorted);
            
            // check to see if we successfully sorted a
            writeln("<<< isSorted = ", isSorted(sorted));
        }

        
        {
            writeln(">>> argCountSortLocHistGlobHist");
            var t1 = Time.getCurrentTime();
            // returns permutation vector
            var iv = argCountSortLocHistGlobHist(a,aMin,aMax);
            writeln("sort time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
            
            // permute a into sorted order
            var sorted: [a.domain] int = a[iv];
            printAry("sorted = ", sorted);
            
            // check to see if we successfully sorted a
            writeln("<<< isSorted = ", isSorted(sorted));
        }

        
        {
            writeln(">>> argCountSortLocHistGlobHistPDIW");
            var t1 = Time.getCurrentTime();
            // returns permutation vector
            var iv = argCountSortLocHistGlobHistPDIW(a,aMin,aMax);
            writeln("sort time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
            
            // permute a into sorted order
            var sorted: [a.domain] int = a[iv];
            printAry("sorted = ", sorted);
            
            // check to see if we successfully sorted a
            writeln("<<< isSorted = ", isSorted(sorted));
        }

        
        {
            writeln(">>> argCountSortLocHistGlobHistPDDW");
            var t1 = Time.getCurrentTime();
            // returns permutation vector
            var iv = argCountSortLocHistGlobHistPDDW(a,aMin,aMax);
            writeln("sort time = ",Time.getCurrentTime() - t1,"sec"); try! stdout.flush();
            
            // permute a into sorted order
            var sorted: [a.domain] int = a[iv];
            printAry("sorted = ", sorted);
            
            // check to see if we successfully sorted a
            writeln("<<< isSorted = ", isSorted(sorted));
        }

        
    }
    // unit test dirver
    proc main() {

        var n = N;
        var nVals = NVALS;
        test_argsort(n, nVals);

    }

}
