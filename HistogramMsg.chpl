module HistogramMsg
{
    
    use ServerConfig;
    
    use Time only;
    use Math only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use PrivateDist;
    
    var sBound = 2**12;
    var mBound = 2**25;

    proc +(x: atomic int, y: atomic int) {
        return x.read() + y.read();
    }
    
    proc +=(X: [?D] int, Y: [D] atomic int) {
        [i in D] {X[i] += Y[i].read();}
    }
    
    /*
    Takes the data in array a, creates an atomic histogram in parallel, 
    and copies the result of the histogram operation into a distributed int array

    Returns the histogram (distributed int array).

    :arg a: array of data to be histogrammed
    :type a: [] ?etype

    :arg aMin: Min value in array a
    :type aMin: etype

    :arg aMax: Max value in array a
    :type aMax: etype

    :arg bins: allocate size of the histogram's distributed domain
    :type bins: int

    :arg binWidth: set value for either 1:1 unique value counts, or multiple unique values per bin.
    :type binWidth: real

    :returns: [] int

    */
    proc histogramGlobalAtomic(a: [?aD] ?etype, aMin: etype, aMax: etype, bins: int, binWidth: real) {

        var hD = makeDistDom(bins);
        var atomicHist: [hD] atomic int;
        
        // count into atomic histogram
        forall v in a {
            var vBin = ((v - aMin) / binWidth):int;
            if v == aMax {vBin = bins-1;}
            //if (v_bin < 0) | (v_bin > (bins-1)) {try! writeln("OOB");try! stdout.flush();}
            atomicHist[vBin].add(1);
        }
        
        var hist = makeDistArray(bins,int);
        // copy from atomic histogram to normal histogram
        [(e,ae) in zip(hist, atomicHist)] e = ae.read();
        //if v {try! writeln("hist =",hist); try! stdout.flush();}

        return hist;
    }

    /*
    Takes the data in array a, creates an atomic histogram in each locale,
    + reduces each locale's histogram computations into a distributed int array

    Returns the histogram (distributed int array).

    :arg a: array of data to be histogrammed
    :type a: [] ?etype

    :arg aMin: Min value in array a
    :type aMin: etype

    :arg aMax: Max value in array a
    :type aMax: etype

    :arg bins: allocate size of the histogram's distributed domain
    :type bins: int

    :arg binWidth: set value for either 1:1 unique value counts, or multiple unique values per bin.
    :type binWidth: real

    :returns: [] int

    */
    proc histogramLocalAtomic(a: [?aD] ?etype, aMin: etype, aMax: etype, bins: int, binWidth: real) {

        // allocate per-locale atomic histogram
        var atomicHist: [PrivateSpace] [0..#bins] atomic int;

        // count into per-locale private atomic histogram
        forall v in a {
            var vBin = ((v - aMin) / binWidth):int;
            if v == aMax {vBin = bins-1;}
            atomicHist[here.id][vBin].add(1);
        }
        
        var hist = makeDistArray(bins,int);

        // +reduce across per-locale histograms to get counts
        hist = + reduce [i in PrivateSpace] atomicHist[i];

        return hist;
    }
    

    /*
    Iterates in parallel over all values of a, histogramming into a new array as each value is processed. 
    This new array is returned as the histogram.
    
    Returns the histogram (distributed int array).

    :arg a: array of data to be histogrammed
    :type a: [] ?etype

    :arg aMin: Min value in array a
    :type aMin: etype

    :arg aMax: Max value in array a
    :type aMax: etype

    :arg bins: allocate size of the histogram's distributed domain
    :type bins: int

    :arg binWidth: set value for either 1:1 unique value counts, or multiple unique values per bin.
    :type binWidth: real

    :returns: [] int

    */
    proc histogramReduceIntent(a: [?aD] ?etype, aMin: etype, aMax: etype, bins: int, binWidth: real) {

        var gHist: [0..#bins] int;
        
        // count into per-task/per-locale histogram and then reduce as tasks complete
        forall v in a with (+ reduce gHist) {
            var vBin = ((v - aMin) / binWidth):int;
            if v == aMax {vBin = bins-1;}
            gHist[vBin] += 1;
        }

        var hist = makeDistArray(bins,int);        
        hist = gHist;
        return hist;
    }
    
    /* histogram takes a pdarray and returns a pdarray with the histogram in it */
    proc histogramMsg(reqMsg: string, st: borrowed SymTab): string {
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var name = fields[2];
        var bins = try! fields[3]:int;
        
        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %s %i : %s".format(cmd, name, bins, rname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);
        if (gEnt == nil) {return unknownSymbolError("histogram",name);}

        // helper nested procedure
        proc histogramHelper(type t) {
          var e = toSymEntry(gEnt,t);
          var aMin = min reduce e.a;
          var aMax = max reduce e.a;
          var binWidth:real = (aMax - aMin):real / bins:real;
          if v {try! writeln("binWidth %r".format(binWidth)); try! stdout.flush();}

          if (bins <= sBound) {
              if v {try! writeln("%t <= %t".format(bins,sBound)); try! stdout.flush();}
              var hist = histogramReduceIntent(e.a, aMin, aMax, bins, binWidth);
              st.addEntry(rname, new shared SymEntry(hist));
          }
          else if (bins <= mBound) {
              if v {try! writeln("%t <= %t".format(bins,mBound)); try! stdout.flush();}
              var hist = histogramLocalAtomic(e.a, aMin, aMax, bins, binWidth);
              st.addEntry(rname, new shared SymEntry(hist));
          }
          else {
              if v {try! writeln("%t > %t".format(bins,mBound)); try! stdout.flush();}
              var hist = histogramGlobalAtomic(e.a, aMin, aMax, bins, binWidth);
              st.addEntry(rname, new shared SymEntry(hist));
          }
        }

        select (gEnt.dtype) {
            when (DType.Int64)   {histogramHelper(int);}
            when (DType.Float64) {histogramHelper(real);}
            otherwise {return notImplementedError("histogram",gEnt.dtype);}
        }
        
        return try! "created " + st.attrib(rname);
    }


}