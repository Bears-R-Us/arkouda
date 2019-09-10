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

    proc testTuple(nVals: int, nRange: (int, int, int), posOnly: (bool, bool, bool),
		   type t) {
      var D = newBlockDom({0..#nVals});
      var v: t;
      var x: [D] v[1].type;
      var y: [D] v[2].type;
      var z: [D] v[3].type;
      fillRandom(x, 241);
      fillRandom(y, 242);
      fillRandom(z, 243);
      var A: [D] t = [(a, b, c) in zip(x, y, z)] (a, b, c);
      forall a in A {
	for param i in 1..3 {
	  if isIntegral(a[i].type) {
	    if (posOnly[i]) {
	      a[i] = if a[i] < 0 then -a[i] else a[i];
	    }
	    a[i] %= nRange[i];
	  } else if isRealType(a[i].type) {
	    if !posOnly[i] {
	      a[i] = 2*a[i] - 1;
	    }
	    a[i] *= nRange[i];
	  }
	}
      }
      printAry("A = ",A);
        
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
        vv = true;
        
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

        vv = RSLSD_vv;
    }

    proc testDigit() {
      var e = -0.138365;
      writeln("e = ", e);
      try! writeln("shiftDoublt(e, 0) = %xu".format(shiftDouble(e, 0):uint));
      var a = getDigit(e, 0);
      var b = getDigit(e, 16);
      var c = getDigit(e, 32);
      var d = getDigit(e, 48);
      try! writeln("getDigit(e,  0) = %xu".format(a:uint));
      try! writeln("getDigit(e, 16) = %xu".format(b:uint));
      try! writeln("getDigit(e, 32) = %xu".format(c:uint));
      try! writeln("getDigit(e, 48) = %xu".format(d:uint));
    }
      
    proc main() {
        writeln("RSLSD_numTasks = ",RSLSD_numTasks);
        writeln("numTasks = ",numTasks);
        writeln("Tasks = ",Tasks);
        writeln("RSLSD_bitsPerDigit = ",RSLSD_bitsPerDigit);
        writeln("bitsPerDigit = ",bitsPerDigit);
        writeln("numBuckets = ",numBuckets);
        writeln("maskDigit = ",maskDigit);

	testDigit();
        testSimple();
        testIt(NVALS, NRANGE,true, int);
        testIt(NVALS, NRANGE,false, int);
	testIt(NVALS, NRANGE, true, real);
	testIt(NVALS, NRANGE, false, real);
	testTuple(NVALS, (NRANGE, NRANGE*NRANGE, (NRANGE**0.5):int), (false, true, false), (int, int, int));
	testTuple(NVALS, (NRANGE, NRANGE*NRANGE, 1), (false, false, true), (real, int, real));
    }


}

