    module DataFrameIndexingMsg
    {
    use ServerConfig;
    use ServerErrorStrings;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use SegmentedMsg;
    use AryUtil;
    use BigInteger;

    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;
    use IOUtils;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const dfiLogger = new Logger(logLevel, logChannel);

    // gather indexing by integer index vector
    proc dfIdxHelper(idx: borrowed SymEntry(int), columnVals: borrowed SymEntry(?t), st: borrowed SymTab, col: string, objType: string, rtnName: bool=false): string throws {
        param pn = Reflection.getRoutineName();
        // get next symbol name
        var rname = st.nextName();

        if (columnVals.size == 0) && (idx.size == 0) {
            var a = st.addEntry(rname, 0, t);
            if rtnName {
                return rname;
            }
            var repMsg = "pdarray+%s+created %s".format(col, st.attrib(rname));
            return repMsg;
        }
        var idxMin = min reduce idx.a;
        var idxMax = max reduce idx.a;
        if idxMin < 0 {
            var errorMsg = "Error: %s: OOBindex %i < 0".format(pn,idxMin);
            dfiLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new owned IllegalArgumentError(errorMsg);
        }
        if idxMax >= columnVals.size {
            var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,idxMax,columnVals.size-1);
            dfiLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            throw new owned IllegalArgumentError(errorMsg);
        }
        var a = st.addEntry(rname, idx.size, t);
        ref a2 = columnVals.a;
        ref iva = idx.a;
        ref aa = a.a;
        forall (a1,i) in zip(aa,iva) {
            a1 = a2[i];
        }

        if rtnName {
            return rname;
        }

        var repMsg =  "%s+%s+created %s".format(objType, col, st.attrib(rname));
        return repMsg;
    }

    proc df_seg_array_idx(idx: borrowed SymEntry(int), segments: borrowed SymEntry(int), values: borrowed SymEntry(?t), col: string, st: borrowed SymTab): string throws {
        var lens: [0..#idx.size] int;
        var orig_segs: [0..#idx.size] int = segments.a[idx.a];

        const high = orig_segs.domain.high;
        const mid = high/2;
        forall(i, os, l) in zip(orig_segs.domain, orig_segs, lens){
            if (i == high){
                l = values.size - os;
            }
            else if (i == mid) {
                l = 0;
            } else {
                l = orig_segs[i+1] - os;
            }
        }

        var rvals = makeDistArray(+ reduce lens, t);
        var rsegs = (+ scan lens) - lens;

        forall(i, rs, os, l) in zip(orig_segs.domain, rsegs, orig_segs, lens){
            var v = new lowLevelLocalizingSlice(values.a, os..#l);
            for j in 0..#l{
                rvals[rs+j] = v.ptr[j];
            }
        }

        var s_name = st.nextName();
        st.addEntry(s_name, createSymEntry(rsegs));
        var v_name = st.nextName();
        st.addEntry(v_name, createSymEntry(rvals));

        return "SegArray+%s+created %s+created %s".format(col, st.attrib(s_name), st.attrib(v_name));
    }

    proc dataframeBatchIndexingMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        var jsonsize = msgArgs.get("size").getIntValue();

        var eleList = msgArgs.get("columns").getList(jsonsize);

        var gIdx: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("idx_name"), st);
        var idx = toSymEntry(gIdx, int);

        var repMsgList: [0..#jsonsize] string;

        for (i, rpm, ele) in zip(repMsgList.domain, repMsgList, eleList) { 
            var ele_parts = ele.split("+");
            ref col_name = ele_parts[1];
            if ele_parts[0] == "Categorical" {
                ref codes_name = ele_parts[2];
                ref categories_name = ele_parts[3];
                dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Element at %i is Categorical\nCodes Name: %s, Categories Name: %s".format(i, codes_name, categories_name));

                var gCode: borrowed GenSymEntry = getGenericTypedArrayEntry(codes_name, st);
                var code_vals = toSymEntry(gCode, int);
                var idxCodeName = dfIdxHelper(idx, code_vals, st, col_name, "", true);
                
                var repTup = segPdarrayIndex(ObjType.STRINGS, categories_name, idxCodeName, DType.UInt8, st);
                
                if repTup.msgType == MsgType.ERROR {
                    throw new IllegalArgumentError(repTup.msg);
                }

                rpm = formatJson("Strings+%s+%s".format(col_name, repTup.msg));
            }
            else if ele_parts[0] == "Strings"{
                dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Element at %i is Strings. Name: %s".format(i, ele_parts[2]));
                var repTup = segPdarrayIndex(ObjType.STRINGS, ele_parts[2], msgArgs.getValueOf("idx_name"), DType.UInt8, st);
                
                if repTup.msgType == MsgType.ERROR {
                    throw new IllegalArgumentError(repTup.msg);
                }

                rpm = formatJson("Strings+%s+%s".format(col_name, repTup.msg));
            }
            else if ele_parts[0] == "pdarray" || ele_parts[0] == "IPv4" || 
                            ele_parts[0] == "Fields" || ele_parts[0] == "Datetime" || ele_parts[0] == "BitVector"{
                    dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Element at %i is pdarray. Name: %s".format(i, ele_parts[2]));
                    var gCol: borrowed GenSymEntry = getGenericTypedArrayEntry(ele_parts[2], st);
                    select (gCol.dtype) {
                        when (DType.Int64) {
                            var col_vals = toSymEntry(gCol, int);
                            rpm = formatJson(dfIdxHelper(idx, col_vals, st, col_name, ele_parts[0]));
                        }
                        when (DType.UInt64) {
                            var col_vals = toSymEntry(gCol, uint);
                            rpm = formatJson(dfIdxHelper(idx, col_vals, st, col_name, ele_parts[0]));
                        }
                        when (DType.Bool) {
                            var col_vals = toSymEntry(gCol, bool);
                            rpm = formatJson(dfIdxHelper(idx, col_vals, st, col_name, ele_parts[0]));
                        }
                        when (DType.Float64){
                            var col_vals = toSymEntry(gCol, real);
                            rpm = formatJson(dfIdxHelper(idx, col_vals, st, col_name, ele_parts[0]));
                        }
                        when (DType.BigInt){
                            var col_vals = toSymEntry(gCol, bigint);
                            rpm = formatJson(dfIdxHelper(idx, col_vals, st, col_name, ele_parts[0]));
                        }
                        otherwise {
                            var errorMsg = notImplementedError(pn,dtype2str(gCol.dtype));
                            dfiLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                            throw new IllegalArgumentError(errorMsg);
                        }
                    }
                }
            else if ele_parts[0] == "SegArray" {
                dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Element at %i is SegArray".format(i));
                ref segments_name = ele_parts[2];
                ref values_name = ele_parts[3];
                dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Segments Name: %s, Values Name: %s".format(segments_name, values_name));

                var gSeg: borrowed GenSymEntry = getGenericTypedArrayEntry(segments_name, st);
                var segments = toSymEntry(gSeg, int);
                var gVal: borrowed GenSymEntry = getGenericTypedArrayEntry(values_name, st);
                select(gVal.dtype){
                    when(DType.Int64){
                        var values = toSymEntry(gVal, int);
                        rpm = formatJson(df_seg_array_idx(idx, segments, values, col_name, st));
                    }
                    when(DType.UInt64){
                        var values = toSymEntry(gVal, uint);
                        rpm = formatJson(df_seg_array_idx(idx, segments, values, col_name, st));
                    }
                    when(DType.Float64){
                        var values = toSymEntry(gVal, real);
                        rpm = formatJson(df_seg_array_idx(idx, segments, values, col_name, st));
                    }
                    when(DType.Bool){
                        var values = toSymEntry(gVal, bool);
                        rpm = formatJson(df_seg_array_idx(idx, segments, values, col_name, st));
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,dtype2str(gVal.dtype));
                        dfiLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        throw new IllegalArgumentError(errorMsg);
                    }
                }
            }
            else {
                var errorMsg = notImplementedError(pn, ele_parts[0]);
                dfiLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                throw new IllegalArgumentError(errorMsg);
            }
        }
        repMsg = "[%s]".format(",".join(repMsgList));
        dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL); 
    }

    use CommandMap;
    registerFunction("dataframe_idx", dataframeBatchIndexingMsg, getModuleName());
}
