module DataFrameIndexingMsg
{
    use ServerConfig;
    use ServerErrorStrings;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
	use SegmentedMsg;

    use MultiTypeSymEntry;
    use MultiTypeSymbolTable;

    use CommAggregation;

    private config const logLevel = ServerConfig.logLevel;
    const dfiLogger = new Logger(logLevel);

	// gather indexing by integer index vector
	proc dfIdxHelper(idx: borrowed SymEntry(int), columnVals: borrowed SymEntry(?t), st: borrowed SymTab, rtnName: bool=false): string throws {
		param pn = Reflection.getRoutineName();
		// get next symbol name
        var rname = st.nextName();

		if (columnVals.size == 0) && (idx.size == 0) {
			var a = st.addEntry(rname, 0, t);
			var repMsg = "created " + st.attrib(rname);
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
			var errorMsg = "Error: %s: OOBindex %i > %i".format(pn,idxMin,columnVals.size-1);
			dfiLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
			throw new owned IllegalArgumentError(errorMsg);
		}
		var a = st.addEntry(rname, idx.size, t);
		ref a2 = columnVals.a;
		ref iva = idx.a;
		ref aa = a.a;
		forall (a1,idx) in zip(aa,iva) {
			a1 = a2[idx];
		}
		
		if rtnName {
			return rname;
		}

		var repMsg =  "pdarray+created " + st.attrib(rname);
		return repMsg;
	}

    proc dataframeBatchIndexingMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
		param pn = Reflection.getRoutineName();
		var repMsg: string; // response message
		// split request into fields
		var (jsonsize_str, iname, json_str) = payload.splitMsgToTuple(3);

		var jsonsize: int;
		try{
			jsonsize = jsonsize_str: int;
		}
		catch {
			var errorMsg = "jsonsize could not be interpreted as an int. %s)".format(jsonsize_str);
			dfiLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
			throw new owned IllegalArgumentError(errorMsg);
		}

		var eleList = jsonToPdArray(json_str, jsonsize);

		var gIdx: borrowed GenSymEntry = getGenericTypedArrayEntry(iname, st);
		var idx = toSymEntry(gIdx, int);

		var repMsgList: [0..#jsonsize] string;

		forall (i, rpm, ele) in zip(repMsgList.domain, repMsgList, eleList) { 
			var ele_parts = ele.split("+");
			select (ele_parts[0]) {
				when ("Categorical") {
					dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Element at %i is Categorical".format(i));
					ref codes_name = ele_parts[1];
					ref categories_name = ele_parts[2];
					dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Codes Name: %s, Categories Name: %s".format(codes_name, categories_name));

					var gCode: borrowed GenSymEntry = getGenericTypedArrayEntry(codes_name, st);
					var code_vals = toSymEntry(gCode, int);
					var idxCodeName = dfIdxHelper(idx, code_vals, st, true);
					
					var args: [1..2] string = [categories_name, idxCodeName];
					var repTup = segPdarrayIndex("str", args, st);
					if repTup.msgType == MsgType.ERROR {
						throw new IllegalArgumentError(repTup.msg);
					}

					rpm = "Strings+%s,".format(repTup.msg);
				}
				when ("Strings") {
					dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Element at %i is Strings".format(i));
					var args: [1..2] string = [ele_parts[1], iname];
					var repTup = segPdarrayIndex("str", args, st);
					if repTup.msgType == MsgType.ERROR {
						throw new IllegalArgumentError(repTup.msg);
					}

					rpm = "Strings+%s,".format(repTup.msg);
				}
				when ("pdarray"){
					dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Element at %i is pdarray".format(i));
					var gCol: borrowed GenSymEntry = getGenericTypedArrayEntry(ele_parts[1], st);
					select (gCol.dtype) {
						when (DType.Int64) {
							var col_vals = toSymEntry(gCol, int);
							rpm = "%s,".format(dfIdxHelper(idx, col_vals, st));
						}
						when (DType.UInt64) {
							var col_vals = toSymEntry(gCol, uint);
							rpm = "%s,".format(dfIdxHelper(idx, col_vals, st));
						}
						when (DType.Bool) {
							var col_vals = toSymEntry(gCol, bool);
							rpm = "%s,".format(dfIdxHelper(idx, col_vals, st));
						}
						when (DType.Float64){
							var col_vals = toSymEntry(gCol, real);
							rpm = "%s,".format(dfIdxHelper(idx, col_vals, st));
						}
						otherwise {
							var errorMsg = notImplementedError(pn,dtype2str(gCol.dtype));
							dfiLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
							throw new IllegalArgumentError(errorMsg);
						}
					}
				}
				// TODO - when SegArray
				otherwise {
					var errorMsg = notImplementedError(pn, ele_parts[0]);
					dfiLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
					throw new IllegalArgumentError(errorMsg);
				}
			}
		}

		// TODO - return values need to provide what type the data is so that we can properly configure the object on the client
		repMsg = "[%s]".format(",".join(repMsgList));
        dfiLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL); 
    }


    proc registerMe() {
      use CommandMap;
      registerFunction("dataframe_idx", dataframeBatchIndexingMsg, getModuleName());
    }
}