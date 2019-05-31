module FindSegmentsMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;

    // experimental
    use UnorderedCopy;

    proc findSegmentsMsg(reqMsg: string, st: borrowed SymTab): string {
        var pn = "findSegments";
        var repMsg: string; // response message
        var fields = reqMsg.split(); // split request into fields
        var cmd = fields[1];
        var kname = fields[2]; // key array
        var pname = fields[3]; // perm array

        // get next symbol name
        var sname = st.nextName(); // segments
        var uname = st.nextName(); // unique keys

        var kEnt: borrowed GenSymEntry = st.lookup(kname);
        if (kEnt == nil) {return unknownSymbolError(pn,kname);}
        var pEnt: borrowed GenSymEntry = st.lookup(pname);
        if (pEnt == nil) {return unknownSymbolError(pn,pname);}

        select (kEnt.dtype, pEnt.dtype) {
            when (DType.Int64, DType.Int64) {
                var k = toSymEntry(kEnt,int); // key array
                var p = toSymEntry(pEnt,int); // perm to sort key array from argsort

                ref ka = k.a; // ref to key array
                ref kad = k.aD; // ref to key array domain
                ref pa = p.a; // ref to permutation array
                
                var sorted: [k.aD] int;
                // permute key array into sorted order
                [(s,idx) in zip(sorted, pa)] unorderedCopy(s,ka[idx]);
                
                var truth: [k.aD] bool;
                // truth array to hold segment break points
                truth[0] = true;
                [(t, s, i) in zip(truth, sorted, kad)] if i > kad.low { t = (sorted[i-1] != s); }

                // +scan to compute segment position... 1-based because of inclusive-scan
                var iv: [truth.domain] int = (+ scan truth);
                // compute how many segments
                var pop = iv[iv.size-1];
                if v {writeln("pop = ",pop,"last-scan = ",iv[iv.size-1]);try! stdout.flush();}

                var segs = makeDistArray(pop, int);
                var ukeys = makeDistArray(pop, int);

                // segment position... 1-based needs to be converted to 0-based because of inclusive-scan
                // where ever a segment break (true value) is... that index is a segment start index
                [i in truth.domain] if (truth[i] == true) {var idx = i; unorderedCopy(segs[iv[i]-1], idx);}
                // pull out the first key in each segment as a unique key
                [i in segs.domain] ukeys[i] = sorted[segs[i]];
                
                st.addEntry(sname, new shared SymEntry(segs));
                st.addEntry(uname, new shared SymEntry(ukeys));
            }
            otherwise {return notImplementedError(pn,"("+dtype2str(kEnt.dtype)+","+dtype2str(pEnt.dtype)+")");}
        }
        
        return try! "created " + st.attrib(sname) + " +created " + st.attrib(uname);
    }
    
}
