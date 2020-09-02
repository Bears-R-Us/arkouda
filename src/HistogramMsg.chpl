module HistogramMsg
{
    use ServerConfig;

    use Reflection only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    use Histogram;
    
    private config const sBound = 2**12;
    private config const mBound = 2**25;

    /* histogram takes a pdarray and returns a pdarray with the histogram in it */
    proc histogramMsg(cmd: string, payload: bytes, st: borrowed SymTab): string throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name, binsStr) = payload.decode().splitMsgToTuple(2);
        var bins = try! binsStr:int;
        
        // get next symbol name
        var rname = st.nextName();
        if v {try! writeln("%s %s %i : %s".format(cmd, name, bins, rname));try! stdout.flush();}

        var gEnt: borrowed GenSymEntry = st.lookup(name);

        // helper nested procedure
        proc histogramHelper(type t) throws {
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
            otherwise {return notImplementedError(pn,gEnt.dtype);}
        }
        
        return try! "created " + st.attrib(rname);
    }


}
