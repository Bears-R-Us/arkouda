module SegArraySetopsMsg {
    use ServerConfig;

    use Time only;
    use Math only;
    use Reflection only;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use SegmentedArray;
    use ServerErrorStrings;

    use SegArraySetops;
    use Indexing;
    use RadixSortLSD;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;

    private config const logLevel = ServerConfig.logLevel;
    const segLogger = new Logger(logLevel);

    proc setopsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (sub_command, seg1_name, vals1_name, s1_str, seg2_name, vals2_name, s2_str, assume_unique) = payload.splitMsgToTuple(8);
        var isUnique = if assume_unique != "" then stringtobool(assume_unique) else false;

        var gEnt: borrowed GenSymEntry = getGenericTypedArrayEntry(seg1_name, st);
        var gEnt2: borrowed GenSymEntry = getGenericTypedArrayEntry(vals1_name, st);
        var gEnt3: borrowed GenSymEntry = getGenericTypedArrayEntry(seg2_name, st);
        var gEnt4: borrowed GenSymEntry = getGenericTypedArrayEntry(vals2_name, st);

        // verify that expected integer values can be cast
        var size1: int;
        var size2: int;
        try{
            size1 = s1_str: int;
            size2 = s2_str: int;
        }
        catch {
            var errorMsg = "size1 or size2 could not be interpreted as an int size1: %s, size2: %s)".format(s1_str, s2_str);
            segLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new owned IllegalArgumentError(errorMsg);
        }

        var segments1 = toSymEntry(gEnt,int);
        var segments2 = toSymEntry(gEnt3,int);

        // set segment lengths
        var lens1: [segments1.aD] int;
        var lens2: [segments2.aD] int;
        var high = segments1.aD.high;
        forall (i, s1, l1, s2, l2) in zip(segments1.aD, segments1.a, lens1, segments2.a, lens2){
            if i == high {
                l1 = size1 - s1;
                l2 = size2 - s2;
            }
            else{
                l1 = segments1.a[i+1] - s1;
                l2 = segments2.a[i+1] - s2;
            }
        }
        var m1: int = max reduce lens1;
        var m2: int = max reduce lens2;

        // perform memory exhaustion check using the size of the largest segment present
        var itemsize = if gEnt2.dtype == DType.UInt64 then numBytes(uint) else numBytes(int);
        var sortMem1 = radixSortLSD_memEst(m1, itemsize);
        var sortMem2 = radixSortLSD_memEst(m2, itemsize);
        var union_maxMem = max(sortMem1, sortMem2);
        overMemLimit(union_maxMem);

        var s_name = st.nextName();
        var v_name = st.nextName();

        select(gEnt2.dtype, gEnt4.dtype){
        when(DType.Int64, DType.Int64){
            var values1 = toSymEntry(gEnt2,int);
            var values2 = toSymEntry(gEnt4,int);
            select(sub_command){
            when("union"){
                var (segments, values) = union_sa(segments1, values1, lens1, segments2, values2, lens2);
                st.addEntry(s_name, new shared SymEntry(segments));
                st.addEntry(v_name, new shared SymEntry(values));
            }
            when("intersect"){
                var (segments, values) = intersect(segments1, values1, lens1, segments2, values2, lens2, isUnique);
                st.addEntry(s_name, new shared SymEntry(segments));
                st.addEntry(v_name, new shared SymEntry(values));
            }
            when("setdiff"){
                var (segments, values) = setdiff(segments1, values1, lens1, segments2, values2, lens2, isUnique);
                st.addEntry(s_name, new shared SymEntry(segments));
                st.addEntry(v_name, new shared SymEntry(values));
            }
            when("setxor"){
                var (segments, values) = setxor(segments1, values1, lens1, segments2, values2, lens2, isUnique);
                st.addEntry(s_name, new shared SymEntry(segments));
                st.addEntry(v_name, new shared SymEntry(values));
            }
            otherwise {
                var errorMsg = notImplementedError("setops1d_multi", sub_command);
                segLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                   
                return new MsgTuple(errorMsg, MsgType.ERROR);              
            }
            }
        }
        when(DType.UInt64, DType.UInt64){
            var values1 = toSymEntry(gEnt2,uint);
            var values2 = toSymEntry(gEnt4,uint);
            select(sub_command){
            when("union"){
                var (segments, values) = union_sa(segments1, values1, lens1, segments2, values2, lens2);
                st.addEntry(s_name, new shared SymEntry(segments));
                st.addEntry(v_name, new shared SymEntry(values));
            }
            when("intersect"){
                var (segments, values) = intersect(segments1, values1, lens1, segments2, values2, lens2, isUnique);
                st.addEntry(s_name, new shared SymEntry(segments));
                st.addEntry(v_name, new shared SymEntry(values));
            }
            when("setdiff"){
                var (segments, values) = setdiff(segments1, values1, lens1, segments2, values2, lens2, isUnique);
                st.addEntry(s_name, new shared SymEntry(segments));
                st.addEntry(v_name, new shared SymEntry(values));
            }
            when("setxor"){
                var (segments, values) = setxor(segments1, values1, lens1, segments2, values2, lens2, isUnique);
                st.addEntry(s_name, new shared SymEntry(segments));
                st.addEntry(v_name, new shared SymEntry(values));
            }
            otherwise {
                var errorMsg = notImplementedError("segarr_setops", sub_command);
                segLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                   
                return new MsgTuple(errorMsg, MsgType.ERROR);              
            }
            }
        }
        otherwise {
            var errorMsg = notImplementedError("segarr_setops", gEnt2.dtype);
            segLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                   
            return new MsgTuple(errorMsg, MsgType.ERROR);              
        }
        }
        repMsg = "created " + st.attrib(s_name) + "+created " + st.attrib(v_name);
        segLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc stringtobool(str: string): bool throws {
        if str == "True" then return true;
        else if str == "False" then return false;
        throw getErrorWithContext(
            msg="message: assume_unique must be of type bool",
            lineNumber=getLineNumber(),
            routineName=getRoutineName(),
            moduleName=getModuleName(),
            errorClass="ErrorWithContext");
    }

    proc registerMe() {
        use CommandMap;
        registerFunction("segarr_setops", setopsMsg, getModuleName());
    }
}