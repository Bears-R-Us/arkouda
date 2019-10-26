module ConcatenateMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection only;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    
    use AryUtil;

    /* Concatenate a list of arrays together
       to form one array
     */
    proc concatenateMsg(reqMsg: string, st: borrowed SymTab) {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var fields = reqMsg.split();
        var cmd = fields[1];
        var n = try! fields[2]:int; // number of arrays to sort
        var names = fields[3..];
        // Check that fields contains the stated number of arrays
        if (n != names.size) { return try! incompatibleArgumentsError(pn, "Expected %i arrays but got %i".format(n, names.size)); }
        /* var arrays: [0..#n] borrowed GenSymEntry; */
        var size: int = 0;
        var dtype: DType;
        // Check that all arrays exist in the symbol table and have the same size
        for (name, i) in zip(names, 1..) {
            // arrays[i] = st.lookup(name): borrowed GenSymEntry;
            var g: borrowed GenSymEntry = st.lookup(name);
            if (g == nil) { return unknownSymbolError(pn, name); }
            if (i == 1) {dtype = g.dtype;}
            else {
                if (dtype != g.dtype) {
                    return try! incompatibleArgumentsError(pn, "Expected %s dtype but got %s dtype".format(dtype2str(dtype), dtype2str(g.dtype)));
                }
            }
            // accumulate size from each array size
            size += g.size;
        }
        // allocate a new array in the symboltable
        // and copy in arrays
        var rname = st.nextName();
        select (dtype) {
            when DType.Int64 {
                // create array to copy into
                var e = st.addEntry(rname, size, int);
                var start: int;
                var end: int;
                start = 0;
                for (name, i) in zip(names, 1..) {
                    // lookup and cast operand to copy from
                    var o = toSymEntry(st.lookup(name), int);
                    // calculate end which is inclusive
                    end = start + o.size - 1;
                    // copy array into concatenation array
                    e.a[{start..end}] = o.a;
                    // update new start for next array copy
                    start += o.size;
                }
            }
            when DType.Float64 {
                // create array to copy into
                var e = st.addEntry(rname, size, real);
                var start: int;
                var end: int;
                start = 0;
                for (name, i) in zip(names, 1..) {
                    // lookup and cast operand to copy from
                    var o = toSymEntry(st.lookup(name), real);
                    // calculate end which is inclusive
                    end = start + o.size - 1;
                    // copy array into concatenation array
                    e.a[{start..end}] = o.a;
                    // update new start for next array copy
                    start += o.size;
                }
            }
            when DType.Bool {
                // create array to copy into
                var e = st.addEntry(rname, size, bool);
                var start: int;
                var end: int;
                start = 0;
                for (name, i) in zip(names, 1..) {
                    // lookup and cast operand to copy from
                    var o = toSymEntry(st.lookup(name), bool);
                    // calculate end which is inclusive
                    end = start + o.size - 1;
                    // copy array into concatenation array
                    e.a[{start..end}] = o.a;
                    // update new start for next array copy
                    start += o.size;
                }
            }
            otherwise {return notImplementedError("concatenate",dtype);}
        }

        return try! "created " + st.attrib(rname);
    }
    
}