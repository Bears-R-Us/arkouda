module ConcatenateMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection;
    use Errors;
    use Logging;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use CommAggregation;

    use AryUtil;

    const cmLogger = new Logger();
    if v {
        cmLogger.level = LogLevel.DEBUG;
    } else {
        cmLogger.level = LogLevel.INFO;
    }

    /* Concatenate a list of arrays together
       to form one array
     */
    proc concatenateMsg(cmd: string, payload: bytes, st: borrowed SymTab) throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var (nstr, objtype, rest) = payload.decode().splitMsgToTuple(3);
        var n = try! nstr:int; // number of arrays to sort
        var fields = rest.split();
        const low = fields.domain.low;
        var names = fields[low..];
        
        cmLogger.debug(getModuleName(),getRoutineName(), getLineNumber(), 
              "number of arrays: %i fields: %t low: %t names: %t".format(n,fields,low,names));

        // Check that fields contains the stated number of arrays
        if (n != names.size) { 
            var errorMsg = incompatibleArgumentsError(pn, 
                             "Expected %i arrays but got %i".format(n, names.size)); 
            cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                               
            return errorMsg;
        }
        /* var arrays: [0..#n] borrowed GenSymEntry; */
        var size: int = 0;
        var nbytes: int = 0;          
        var dtype: DType;
        // Check that all arrays exist in the symbol table and have the same size
        for (rawName, i) in zip(names, 1..) {
            // arrays[i] = st.lookup(name): borrowed GenSymEntry;
            var name: string;
            select objtype {
                when "str" {
                    var valName: string;
                    (name, valName) = rawName.splitMsgToTuple('+', 2);
                    var gval = st.lookup(valName);
                    nbytes += gval.size;
                    cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "name: %s valName: %s".format(name,valName));
                }
                when "pdarray" {
                    name = rawName;
                    cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                 "pdarray name %s".format(rawName));
                }
                otherwise { 
                    var errorMsg = notImplementedError(pn, objtype); 
                    cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
                    return errorMsg;                    
                }
            }
            var g: borrowed GenSymEntry = st.lookup(name);
            if (i == 1) {dtype = g.dtype;}
            else {
                if (dtype != g.dtype) {
                    var errorMsg = incompatibleArgumentsError(pn, 
                             "Expected %s dtype but got %s dtype".format(dtype2str(dtype), 
                                    dtype2str(g.dtype)));
                    cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return errorMsg;
                }
            }
            // accumulate size from each array size
            size += g.size;
        }

        // allocate a new array in the symboltable
        // and copy in arrays
        select objtype {
            when "str" {
                var segName = st.nextName();
                var esegs = st.addEntry(segName, size, int);
                ref esa = esegs.a;
                var valName = st.nextName();
                var evals = st.addEntry(valName, nbytes, uint(8));
                ref eva = evals.a;
                var segStart = 0;
                var valStart = 0;
                for (rawName, i) in zip(names, 1..) {
                    var (segName, valName) = rawName.splitMsgToTuple('+', 2);
                    var thisSegs = toSymEntry(st.lookup(segName), int);
                    var newSegs = thisSegs.a + valStart;
                    var thisVals = toSymEntry(st.lookup(valName), uint(8));
                    forall (i, s) in zip(newSegs.domain, newSegs) with (var agg = newDstAggregator(int)) {
                        agg.copy(esa[i+segStart], s);
                    }
                    forall (i, v) in zip(thisVals.aD, thisVals.a) with (var agg = newDstAggregator(uint(8))) {
                        agg.copy(eva[i+valStart], v);
                    }
                    segStart += thisSegs.size;
                    valStart += thisVals.size;
                }
                var repMsg = "created " + st.attrib(segName) + "+created " + st.attrib(valName);
                cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "created concatenated pdarray %s".format(st.attrib(valName)));
                return repMsg;
            }
            when "pdarray" {
                var rname = st.nextName();
                cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                             "creating pdarray %s of type %t".format(rname,dtype));
                select (dtype) {
                    when DType.Int64 {
                        // create array to copy into
                        var e = st.addEntry(rname, size, int);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            var o = toSymEntry(st.lookup(name), int);
                            ref ea = e.a;
                            // copy array into concatenation array
                            forall (i, v) in zip(o.aD, o.a) with (var agg = newDstAggregator(int)) {
                              agg.copy(ea[start+i], v);
                            }
                            // update new start for next array copy
                            start += o.size;
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    when DType.Float64 {
                        // create array to copy into
                        var e = st.addEntry(rname, size, real);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            var o = toSymEntry(st.lookup(name), real);
                            ref ea = e.a;
                            // copy array into concatenation array
                            forall (i, v) in zip(o.aD, o.a) with (var agg = newDstAggregator(real)) {
                              agg.copy(ea[start+i], v);
                            }
                            // update new start for next array copy
                            start += o.size;
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    when DType.Bool {
                        // create array to copy into
                        var e = st.addEntry(rname, size, bool);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            var o = toSymEntry(st.lookup(name), bool);
                            ref ea = e.a;
                            // copy array into concatenation array
                            forall (i, v) in zip(o.aD, o.a) with (var agg = newDstAggregator(bool)) {
                              agg.copy(ea[start+i], v);
                            }
                            // update new start for next array copy
                            start += o.size;
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    otherwise {
                        var errorMsg = notImplementedError("concatenate",dtype);
                        cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return errorMsg;                         
                    }
                }
                repMsg = "created " + st.attrib(rname);
                cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return repMsg;
            }
            otherwise { 
                var errorMsg = notImplementedError(pn, objtype); 
                cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return errorMsg;
            }
        }
    }
}
