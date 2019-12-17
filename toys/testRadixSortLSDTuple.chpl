module testRadixSortLSDTuple
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
        
        var timer:Timer;
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
    }

    proc makeArray(example, D) {
      var arr: [D] example.type;
      return arr;
    }
    
    proc testTuple(nVals: int, type args...?nt) {
      var D = newBlockDom({0..#nVals});
      var dummy: args;
      var arrays = for i in 0..#nt do makeArray(dummy[i], D);
      for (a, i) in zip(arrays, 1..) {
        fillRandom(a, i);
        printAry("arrays[", i, "] = ", a);
      }
        
        var timer:Timer;
        timer.start();
        // sort data
        var iv = radixSortLSDMulti_ranks(arrays);
        timer.stop();
        writeln(">>>Sorted ", nVals, " elements in ", timer.elapsed(), " seconds",
                " (", 8.0*nVals/timer.elapsed()/1024.0/1024.0, " MiB/s)");

        printAry("iv = ", iv);
        var sorted = for a in arrays do a[iv];
        for (s, i) in zip(sorted, 1..) {
          printAry("arrays[", i, "][iv] = ", s);
        }
        var allSorted = true;
        forall i in D with (&& reduce allSorted) {
          if i < D.high {
            for j in 1..nt {
              if (sorted[j][i] < sorted[j][i+1]) {
                break;
              } else if sorted[j][i] > sorted[j][i+1] {
                allSorted reduce= false;
              }
            }
          }
        }
        writeln("allSorted? ", allSorted);
    }

    /* proc testSimple() { */
    /*     vv = true; */
        
    /*     // test a simple case */
    /*     var D = newBlockDom({0..#8}); */
    /*     var A: [D] int = [5,7,3,1,4,2,7,2]; */
        
    /*     printAry("A = ",A); */
        
    /*     var nBits = 64 - clz(max reduce A); */
        
    /*     var iv = radixSortLSD(A); */
    /*     printAry("iv = ", iv); */
    /*     var aiv = A[iv]; */
    /*     printAry("A[iv] = ", aiv); */
    /*     writeln(isSorted(aiv)); */

    /*     vv = RSLSD_vv; */
    /* } */

    /* proc testDigit() { */
    /*   var e = -0.138365; */
    /*   writeln("e = ", e); */
    /*   try! writeln("shiftDoublt(e, 0) = %xu".format(shiftDouble(e, 0):uint)); */
    /*   var a = getDigit(e, 0); */
    /*   var b = getDigit(e, 16); */
    /*   var c = getDigit(e, 32); */
    /*   var d = getDigit(e, 48); */
    /*   try! writeln("getDigit(e,  0) = %xu".format(a:uint)); */
    /*   try! writeln("getDigit(e, 16) = %xu".format(b:uint)); */
    /*   try! writeln("getDigit(e, 32) = %xu".format(c:uint)); */
    /*   try! writeln("getDigit(e, 48) = %xu".format(d:uint)); */
    /* } */
      
    proc main() {
        writeln("RSLSD_numTasks = ",RSLSD_numTasks);
        writeln("numTasks = ",numTasks);
        writeln("Tasks = ",Tasks);
        writeln("RSLSD_bitsPerDigit = ",RSLSD_bitsPerDigit);
        writeln("bitsPerDigit = ",bitsPerDigit);
        writeln("numBuckets = ",numBuckets);
        writeln("maskDigit = ",maskDigit);

        // testDigit();
        // testSimple();
        testIt(NVALS, NRANGE,true, int);
        testIt(NVALS, NRANGE,false, int);
        testIt(NVALS, NRANGE, true, real);
        testIt(NVALS, NRANGE, false, real);
        // testTuple(NVALS, (NRANGE, NRANGE*NRANGE, (NRANGE**0.5):int), (false, true, false), (int, int, int));
        // testTuple(NVALS, (NRANGE, NRANGE*NRANGE, 1), (false, false, true), (real, int, real));
        testTuple(NVALS, (int, int, int));
        testTuple(NVALS, (real, int, real));
    }


}

