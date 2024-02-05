
module OperatorMsg
{
    use ServerConfig;

    use Math;
    use BitOps;
    use Reflection;
    use ServerErrors;
    use BinOp;
    use BigInteger;

    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings; 
    use Reflection;
    use Logging;
    use Message;

    use ArkoudaTimeCompat as Time;
    use ArkoudaBigIntCompat;
    
    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const omLogger = new Logger(logLevel, logChannel);

    /*
      Parse and respond to binopvv message.
      vv == vector op vector

      :arg reqMsg: request containing (cmd,op,aname,bname,rname)
      :type reqMsg: string 

      :arg st: SymTab to act on
      :type st: borrowed SymTab 

      :returns: (MsgTuple) 
      :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.registerND
    proc binopvvMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        
        const op = msgArgs.getValueOf("op");
        const aname = msgArgs.getValueOf("a");
        const bname = msgArgs.getValueOf("b");

        var rname = st.nextName();
        var left: borrowed GenSymEntry = getGenericTypedArrayEntry(aname, st);
        var right: borrowed GenSymEntry = getGenericTypedArrayEntry(bname, st);
        
        omLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), 
             "cmd: %? op: %? left pdarray: %? right pdarray: %?".doFormat(
                                          cmd,op,st.attrib(aname),st.attrib(bname)));

        use Set;
        // This boolOps set is a filter to determine the output type for the operation.
        // All operations that involve one of these operations result in a `bool` symbol
        // table entry.
        var boolOps: set(string);
        boolOps.add("<");
        boolOps.add("<=");
        boolOps.add(">");
        boolOps.add(">=");
        boolOps.add("==");
        boolOps.add("!=");

        var realOps: set(string);
        realOps.add("+");
        realOps.add("-");
        realOps.add("/");
        realOps.add("//");

        select (left.dtype, right.dtype) {
          when (DType.Int64, DType.Int64) {
            var l = toSymEntry(left,int, nd);
            var r = toSymEntry(right,int, nd);
            
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            } else if op == "/" {
              // True division is the only case in this int, int case
              // that results in a `real` symbol table entry.
              var e = st.addEntry(rname, l.tupShape, real);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, int);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.Int64, DType.Float64) {
            var l = toSymEntry(left,int, nd);
            var r = toSymEntry(right,real, nd);
            // Only two possible resultant types are `bool` and `real`
            // for this case
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.Float64, DType.Int64) {
            var l = toSymEntry(left,real, nd);
            var r = toSymEntry(right,int, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.UInt64, DType.Float64) {
            var l = toSymEntry(left,uint, nd);
            var r = toSymEntry(right,real, nd);
            // Only two possible resultant types are `bool` and `real`
            // for this case
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.Float64, DType.UInt64) {
            var l = toSymEntry(left,real, nd);
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.Float64, DType.Float64) {
            var l = toSymEntry(left,real, nd);
            var r = toSymEntry(right,real, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          // For cases where a boolean operand is involved, the only
          // possible resultant type is `bool`
          when (DType.Bool, DType.Bool) {
            var l = toSymEntry(left,bool, nd);
            var r = toSymEntry(right,bool, nd);
            if (op == "<<") || (op == ">>" )  {
              var e = st.addEntry(rname, l.tupShape, int);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, bool);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.Bool, DType.Int64) {
            var l = toSymEntry(left,bool, nd);
            var r = toSymEntry(right,int, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, int);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.Int64, DType.Bool) {
            var l = toSymEntry(left,int, nd);
            var r = toSymEntry(right,bool, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, int);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.Bool, DType.Float64) {
            var l = toSymEntry(left,bool, nd);
            var r = toSymEntry(right,real, nd);
           if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.Float64, DType.Bool) {
            var l = toSymEntry(left,real, nd);
            var r = toSymEntry(right,bool, nd);
           if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.Bool, DType.UInt64) {
            var l = toSymEntry(left,bool, nd);
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, uint);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.UInt64, DType.Bool) {
            var l = toSymEntry(left,uint, nd);
            var r = toSymEntry(right,bool, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, uint);
            return doBinOpvv(l, r, e, op, rname, pn, st);
          }
          when (DType.UInt64, DType.UInt64) {
            var l = toSymEntry(left,uint, nd);
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            if op == "/"{
              var e = st.addEntry(rname, l.tupShape, real);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            } else {
              var e = st.addEntry(rname, l.tupShape, uint);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
          }
          when (DType.UInt64, DType.Int64) {
            var l = toSymEntry(left,uint, nd);
            var r = toSymEntry(right,int, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r , e, op, rname, pn, st);
            }
            // +, -, /, // both result in real outputs to match NumPy
            if realOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, real);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            } else {
              // isn't +, -, /, // so we can use LHS to determine type
              var e = st.addEntry(rname, l.tupShape, uint);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
          }
          when (DType.Int64, DType.UInt64) {
            var l = toSymEntry(left,int, nd);
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
            // +, -, /, // both result in real outputs to match NumPy
            if realOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, real);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            } else {
              // isn't +, -, /, // so we can use LHS to determine type
              var e = st.addEntry(rname, l.tupShape, int);
              return doBinOpvv(l, r, e, op, rname, pn, st);
            }
          }
          when (DType.BigInt, DType.BigInt) {
            var l = toSymEntry(left,bigint, nd);
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.Int64) {
            var l = toSymEntry(left,bigint, nd);
            var r = toSymEntry(right,int, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.UInt64) {
            var l = toSymEntry(left,bigint, nd);
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.Bool) {
            var l = toSymEntry(left,bigint, nd);
            var r = toSymEntry(right,bool, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.Int64, DType.BigInt) {
            var l = toSymEntry(left,int, nd);
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.UInt64, DType.BigInt) {
            var l = toSymEntry(left,uint, nd);
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.Bool, DType.BigInt) {
            var l = toSymEntry(left,bool, nd);
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
        }
        var errorMsg = unrecognizedTypeError(pn, "("+dtype2str(left.dtype)+","+dtype2str(right.dtype)+")");
        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }
    
    /*
      Parse and respond to binopvs message.
      vs == vector op scalar

      :arg reqMsg: request containing (cmd,op,aname,dtype,value)
      :type reqMsg: string

      :arg st: SymTab to act on
      :type st: borrowed SymTab

      :returns: (MsgTuple)
      :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.registerND
    proc binopvsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string = ""; // response message

        const aname = msgArgs.getValueOf("a");
        const op = msgArgs.getValueOf("op");
        const value = msgArgs.get("value");

        const dtype = str2dtype(msgArgs.getValueOf("dtype"));
        var rname = st.nextName();
        var left: borrowed GenSymEntry = getGenericTypedArrayEntry(aname, st);

        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                           "op: %s dtype: %? pdarray: %? scalar: %?".doFormat(
                                                     op,dtype,st.attrib(aname),value.getValue()));

        use Set;
        // This boolOps set is a filter to determine the output type for the operation.
        // All operations that involve one of these operations result in a `bool` symbol
        // table entry.
        var boolOps: set(string);
        boolOps.add("<");
        boolOps.add("<=");
        boolOps.add(">");
        boolOps.add(">=");
        boolOps.add("==");
        boolOps.add("!=");

        var realOps: set(string);
        realOps.add("+");
        realOps.add("-");
        realOps.add("/");
        realOps.add("//");

        select (left.dtype, dtype) {
          when (DType.Int64, DType.Int64) {
            var l = toSymEntry(left,int, nd);
            var val = value.getIntValue();
            
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            } else if op == "/" {
              // True division is the only case in this int, int case
              // that results in a `real` symbol table entry.
              var e = st.addEntry(rname, l.tupShape, real);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, int);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.Int64, DType.Float64) {
            var l = toSymEntry(left,int, nd);
            var val = value.getRealValue();
            // Only two possible resultant types are `bool` and `real`
            // for this case
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.Float64, DType.Int64) {
            var l = toSymEntry(left,real, nd);
            var val = value.getIntValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.UInt64, DType.Float64) {
            var l = toSymEntry(left,uint, nd);
            var val = value.getRealValue();
            // Only two possible resultant types are `bool` and `real`
            // for this case
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.Float64, DType.UInt64) {
            var l = toSymEntry(left,real, nd);
            var val = value.getUIntValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.Float64, DType.Float64) {
            var l = toSymEntry(left,real, nd);
            var val = value.getRealValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          // For cases where a boolean operand is involved, the only
          // possible resultant type is `bool`
          when (DType.Bool, DType.Bool) {
            var l = toSymEntry(left,bool, nd);
            var val = value.getBoolValue();
            if (op == "<<") || (op == ">>") {
              var e = st.addEntry(rname, l.tupShape, int);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, bool);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.Bool, DType.Int64) {
            var l = toSymEntry(left,bool, nd);
            var val = value.getIntValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, int);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.Int64, DType.Bool) {
            var l = toSymEntry(left,int, nd);
            var val = value.getBoolValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, int);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.Bool, DType.Float64) {
            var l = toSymEntry(left,bool, nd);
            var val = value.getRealValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.Float64, DType.Bool) {
            var l = toSymEntry(left,real, nd);
            var val = value.getBoolValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, real);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.Bool, DType.UInt64) {
            var l = toSymEntry(left,bool, nd);
            var val = value.getUIntValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, uint);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.UInt64, DType.Bool) {
            var l = toSymEntry(left,uint, nd);
            var val = value.getBoolValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, l.tupShape, uint);
            return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
          }
          when (DType.UInt64, DType.UInt64) {
            var l = toSymEntry(left,uint, nd);
            var val = value.getUIntValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            if op == "/"{
              var e = st.addEntry(rname, l.tupShape, real);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            } else {
              var e = st.addEntry(rname, l.tupShape, uint);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
          }
          when (DType.UInt64, DType.Int64) {
            var l = toSymEntry(left,uint, nd);
            var val = value.getIntValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            // +, -, /, // both result in real outputs to match NumPy
            if realOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, real);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            } else {
              // isn't +, -, /, // so we can use LHS to determine type
              var e = st.addEntry(rname, l.tupShape, uint);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
          }
          when (DType.Int64, DType.UInt64) {
            var l = toSymEntry(left,int, nd);
            var val = value.getUIntValue();
            if boolOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, bool);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
            // +, -, /, // both result in real outputs to match NumPy
            if realOps.contains(op) {
              var e = st.addEntry(rname, l.tupShape, real);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            } else {
              // isn't +, -, /, // so we can use LHS to determine type
              var e = st.addEntry(rname, l.tupShape, int);
              return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
            }
          }
          when (DType.BigInt, DType.BigInt) {
            var l = toSymEntry(left,bigint, nd);
            var val = value.getBigIntValue();
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.Int64) {
            var l = toSymEntry(left,bigint, nd);
            var val = value.getIntValue();
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.UInt64) {
            var l = toSymEntry(left,bigint, nd);
            var val = value.getUIntValue();
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.Bool) {
            var l = toSymEntry(left,bigint, nd);
            var val = value.getBoolValue();
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.Int64, DType.BigInt) {
            var l = toSymEntry(left,int, nd);
            var val = value.getBigIntValue();
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.UInt64, DType.BigInt) {
            var l = toSymEntry(left,uint, nd);
            var val = value.getBigIntValue();
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.Bool, DType.BigInt) {
            var l = toSymEntry(left,bool, nd);
            var val = value.getBigIntValue();
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
        }
        var errorMsg = unrecognizedTypeError(pn, "("+dtype2str(left.dtype)+","+dtype2str(dtype)+")");
        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    /*
      Parse and respond to binopsv message.
      sv == scalar op vector

      :arg reqMsg: request containing (cmd,op,dtype,value,aname)
      :type reqMsg: string 

      :arg st: SymTab to act on
      :type st: borrowed SymTab 

      :returns: (MsgTuple) 
      :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.registerND
    proc binopsvMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string = ""; // response message

        const op = msgArgs.getValueOf("op");
        const aname = msgArgs.getValueOf("a");
        const value = msgArgs.get("value");

        var dtype = str2dtype(msgArgs.getValueOf("dtype"));
        var rname = st.nextName();
        var right: borrowed GenSymEntry = getGenericTypedArrayEntry(aname, st);
        
        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                 "command = %? op = %? scalar dtype = %? scalar = %? pdarray = %?".doFormat(
                                   cmd,op,dtype2str(dtype),value,st.attrib(aname)));

        use Set;
        // This boolOps set is a filter to determine the output type for the operation.
        // All operations that involve one of these operations result in a `bool` symbol
        // table entry.
        var boolOps: set(string);
        boolOps.add("<");
        boolOps.add("<=");
        boolOps.add(">");
        boolOps.add(">=");
        boolOps.add("==");
        boolOps.add("!=");
        
        var realOps: set(string);
        realOps.add("+");
        realOps.add("-");
        realOps.add("/");
        realOps.add("//");

        select (dtype, right.dtype) {
          when (DType.Int64, DType.Int64) {
            var val = value.getIntValue();
            var r = toSymEntry(right,int, nd);
            
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            } else if op == "/" {
              // True division is the only case in this int, int case
              // that results in a `real` symbol table entry.
              var e = st.addEntry(rname, r.tupShape, real);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, int);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.Int64, DType.Float64) {
            var val = value.getIntValue();
            var r = toSymEntry(right,real, nd);
            // Only two possible resultant types are `bool` and `real`
            // for this case
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, real);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.Float64, DType.Int64) {
            var val = value.getRealValue();
            var r = toSymEntry(right,int, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, real);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.UInt64, DType.Float64) {
            var val = value.getUIntValue();
            var r = toSymEntry(right,real, nd);
            // Only two possible resultant types are `bool` and `real`
            // for this case
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, real);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.Float64, DType.UInt64) {
            var val = value.getRealValue();
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, real);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.Float64, DType.Float64) {
            var val = value.getRealValue();
            var r = toSymEntry(right,real, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, real);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          // For cases where a boolean operand is involved, the only
          // possible resultant type is `bool`
          when (DType.Bool, DType.Bool) {
            var val = value.getBoolValue();
            var r = toSymEntry(right,bool, nd);
            if (op == "<<") || (op == ">>") {
              var e = st.addEntry(rname, r.tupShape, int);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, bool);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.Bool, DType.Int64) {
            var val = value.getBoolValue();
            var r = toSymEntry(right,int, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, int);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.Int64, DType.Bool) {
            var val = value.getIntValue();
            var r = toSymEntry(right,bool, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, int);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.Bool, DType.Float64) {
            var val = value.getBoolValue();
            var r = toSymEntry(right,real, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, real);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.Float64, DType.Bool) {
            var val = value.getRealValue();
            var r = toSymEntry(right,bool, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, real);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.Bool, DType.UInt64) {
            var val = value.getBoolValue();
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, uint);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.UInt64, DType.Bool) {
            var val = value.getUIntValue();
            var r = toSymEntry(right,bool, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            var e = st.addEntry(rname, r.tupShape, uint);
            return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
          }
          when (DType.UInt64, DType.UInt64) {
            var val = value.getUIntValue();
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            if op == "/"{
              var e = st.addEntry(rname, r.tupShape, real);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            } else {
              var e = st.addEntry(rname, r.tupShape, uint);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
          }
          when (DType.UInt64, DType.Int64) {
            var val = value.getUIntValue();
            var r = toSymEntry(right,int, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            // +, -, /, // both result in real outputs to match NumPy
            if realOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, real);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            } else {
              // isn't +, -, /, // so we can use LHS to determine type
              var e = st.addEntry(rname, r.tupShape, uint);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
          }
          when (DType.Int64, DType.UInt64) {
            var val = value.getIntValue();
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, bool);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
            // +, -, /, // both result in real outputs to match NumPy
            if realOps.contains(op) {
              var e = st.addEntry(rname, r.tupShape, real);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            } else {
              // isn't +, -, /, // so we can use LHS to determine type
              var e = st.addEntry(rname, r.tupShape, int);
              return doBinOpsv(val, r, e, op, dtype, rname, pn, st);
            }
          }
          when (DType.BigInt, DType.BigInt) {
            var val = value.getBigIntValue();
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.Int64) {
            var val = value.getBigIntValue();
            var r = toSymEntry(right,int, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.UInt64) {
            var val = value.getBigIntValue();
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.Bool) {
            var val = value.getBigIntValue();
            var r = toSymEntry(right,bool, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.Int64, DType.BigInt) {
            var val = value.getIntValue();
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.UInt64, DType.BigInt) {
            var val = value.getUIntValue();
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.Bool, DType.BigInt) {
            var val = value.getBoolValue();
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".doFormat(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".doFormat(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
        }
        var errorMsg = unrecognizedTypeError(pn, "("+dtype2str(dtype)+","+dtype2str(right.dtype)+")");
        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }

    /*
      Parse and respond to opeqvv message.
      vector op= vector

      :arg reqMsg: request containing (cmd,op,aname,bname)
      :type reqMsg: string

      :arg st: SymTab to act on
      :type st: borrowed SymTab

      :returns: (MsgTuple)
      :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.registerND
    proc opeqvvMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message

        const op = msgArgs.getValueOf("op");
        const aname = msgArgs.getValueOf("a");
        const bname = msgArgs.getValueOf("b");

        // retrieve left and right pdarray objects      
        var left: borrowed GenSymEntry = getGenericTypedArrayEntry(aname, st);
        var right: borrowed GenSymEntry = getGenericTypedArrayEntry(bname, st);

        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "cmd: %s op: %s left pdarray: %s right pdarray: %s".doFormat(cmd,op,
                                                         st.attrib(aname),st.attrib(bname)));

        select (left.dtype, right.dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int, nd);
                var r = toSymEntry(right,int, nd);
                select op {
                    when "+=" { l.a += r.a; }
                    when "-=" { l.a -= r.a; }
                    when "*=" { l.a *= r.a; }
                    when ">>=" { l.a >>= r.a;}
                    when "<<=" { l.a <<= r.a;}
                    when "//=" {
                        //l.a /= r.a;
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = if ri != 0 then li/ri else 0;
                    }//floordiv
                    when "%=" {
                        //l.a /= r.a;
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = if ri != 0 then li%ri else 0;
                    }
                    when "**=" { 
                        if || reduce (r.a<0){
                            var errorMsg =  "Attempt to exponentiate base of type Int64 to negative exponent";
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                        else{ l.a **= r.a; }
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Int64, DType.UInt64) {
                // The result of operations between int and uint are float by default which doesn't fit in either type
                var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.Int64, DType.Float64) {
                var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.Int64, DType.Bool) {
                var l = toSymEntry(left, int, nd);
                var r = toSymEntry(right, bool, nd);
                select op {
                    when "+=" {l.a += r.a:int;}
                    when "-=" {l.a -= r.a:int;}
                    when "*=" {l.a *= r.a:int;}
                    when ">>=" { l.a >>= r.a:int;}
                    when "<<=" { l.a <<= r.a:int;}
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Int64, DType.BigInt) {
                var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.UInt64, DType.Int64) {
                // The result of operations between int and uint are float by default which doesn't fit in either type
                var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.UInt64, DType.UInt64) {
                var l = toSymEntry(left,uint, nd);
                var r = toSymEntry(right,uint, nd);
                select op {
                    when "+=" { l.a += r.a; }
                    when "-=" {
                        l.a -= r.a;
                    }
                    when "*=" { l.a *= r.a; }
                    when "//=" {
                        //l.a /= r.a;
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = if ri != 0 then li/ri else 0;
                    }//floordiv
                    when "%=" {
                        //l.a /= r.a;
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = if ri != 0 then li%ri else 0;
                    }
                    when "**=" {
                        l.a **= r.a;
                    }
                    when ">>=" { l.a >>= r.a;}
                    when "<<=" { l.a <<= r.a;}
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.UInt64, DType.Float64) {
                var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.UInt64, DType.Bool) {
                var l = toSymEntry(left, uint, nd);
                var r = toSymEntry(right, bool, nd);
                select op {
                    when "+=" {l.a += r.a:uint;}
                    when "-=" {l.a -= r.a:uint;}
                    when "*=" {l.a *= r.a:uint;}
                    when ">>=" { l.a >>= r.a:uint;}
                    when "<<=" { l.a <<= r.a:uint;}
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.UInt64, DType.BigInt) {
                var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real, nd);
                var r = toSymEntry(right,int, nd);

                select op {
                    when "+=" {l.a += r.a;}
                    when "-=" {l.a -= r.a;}
                    when "*=" {l.a *= r.a;}
                    when "/=" {l.a /= r.a:real;} //truediv
                    when "//=" { //floordiv
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = floorDivisionHelper(li, ri);
                    }
                    when "**=" { l.a **= r.a; }
                    when "%=" {
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = modHelper(li, ri);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64, DType.UInt64) {
                var l = toSymEntry(left,real, nd);
                var r = toSymEntry(right,uint, nd);

                select op {
                    when "+=" {l.a += r.a;}
                    when "-=" {l.a -= r.a;}
                    when "*=" {l.a *= r.a;}
                    when "/=" {l.a /= r.a:real;} //truediv
                    when "//=" { //floordiv
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = floorDivisionHelper(li, ri);
                    }
                    when "**=" { l.a **= r.a; }
                    when "%=" {
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = modHelper(li, ri);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64, DType.Float64) {
                var l = toSymEntry(left,real, nd);
                var r = toSymEntry(right,real, nd);
                select op {
                    when "+=" {l.a += r.a;}
                    when "-=" {l.a -= r.a;}
                    when "*=" {l.a *= r.a;}
                    when "/=" {l.a /= r.a;}//truediv
                    when "//=" { //floordiv
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = floorDivisionHelper(li, ri);
                    }
                    when "**=" { l.a **= r.a; }
                    when "%=" {
                        ref la = l.a;
                        ref ra = r.a;
                        [(li,ri) in zip(la,ra)] li = modHelper(li, ri);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64, DType.Bool) {
                var l = toSymEntry(left, real, nd);
                var r = toSymEntry(right, bool, nd);
                select op {
                    when "+=" {l.a += r.a:real;}
                    when "-=" {l.a -= r.a:real;}
                    when "*=" {l.a *= r.a:real;}
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64, DType.BigInt) {
                var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.Bool, DType.Bool) {
                var l = toSymEntry(left, bool, nd);
                var r = toSymEntry(right, bool, nd);
                select op {
                    when "|=" {l.a |= r.a;}
                    when "&=" {l.a &= r.a;}
                    when "^=" {l.a ^= r.a;}
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.BigInt, DType.Int64) {
                var l = toSymEntry(left,bigint, nd);
                var r = toSymEntry(right,int, nd);
                ref la = l.a;
                ref ra = r.a;
                var max_bits = l.max_bits;
                var max_size = 1:bigint;
                var has_max_bits = max_bits != -1;
                if has_max_bits {
                  max_size <<= max_bits;
                  max_size -= 1;
                }
                select op {
                  when "+=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li += ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "-=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li -= ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "*=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li *= ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "//=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      if ri != 0 {
                        li /= ri;
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "%=" {
                    // we can't use li %= ri because this can result in negatives
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      if ri != 0 {
                        mod(li, li, ri);
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "**=" {
                    if || reduce (ra<0) {
                      throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
                    }
                    if has_max_bits {
                      forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                        powMod(li, li, ri, local_max_size + 1);
                      }
                    }
                    else {
                      forall (li, ri) in zip(la, ra) {
                        li **= ri:uint;
                      }
                    }
                  }
                  otherwise {
                    var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
                }
            }
            when (DType.BigInt, DType.UInt64) {
                var l = toSymEntry(left,bigint, nd);
                var r = toSymEntry(right,uint, nd);
                ref la = l.a;
                ref ra = r.a;
                var max_bits = l.max_bits;
                var max_size = 1:bigint;
                var has_max_bits = max_bits != -1;
                if has_max_bits {
                  max_size <<= max_bits;
                  max_size -= 1;
                }
                select op {
                  when "+=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li += ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "-=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li -= ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "*=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li *= ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "//=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      if ri != 0 {
                        li /= ri;
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "%=" {
                    // we can't use li %= ri because this can result in negatives
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      if ri != 0 {
                        mod(li, li, ri);
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "**=" {
                    if || reduce (ra<0) {
                      throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
                    }
                    if has_max_bits {
                      forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                        powMod(li, li, ri, local_max_size + 1);
                      }
                    }
                    else {
                      forall (li, ri) in zip(la, ra) {
                        li **= ri:uint;
                      }
                    }
                  }
                  otherwise {
                    var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
                }
            }
            when (DType.BigInt, DType.Float64) {
                var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.BigInt, DType.Bool) {
                var l = toSymEntry(left,bigint, nd);
                var r = toSymEntry(right,bool, nd);
                ref la = l.a;
                var ra = r.a:bigint;
                var max_bits = l.max_bits;
                var max_size = 1:bigint;
                var has_max_bits = max_bits != -1;
                if has_max_bits {
                  max_size <<= max_bits;
                  max_size -= 1;
                }
                select op {
                  when "+=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li += ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "-=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li -= ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "*=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li *= ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  otherwise {
                    var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
                }
            }
            when (DType.BigInt, DType.BigInt) {
                var l = toSymEntry(left,bigint, nd);
                var r = toSymEntry(right,bigint, nd);
                ref la = l.a;
                ref ra = r.a;
                var max_bits = l.max_bits;
                var max_size = 1:bigint;
                var has_max_bits = max_bits != -1;
                if has_max_bits {
                  max_size <<= max_bits;
                  max_size -= 1;
                }
                select op {
                  when "+=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li += ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "-=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li -= ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "*=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      li *= ri;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "//=" {
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      if ri != 0 {
                        li /= ri;
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "%=" {
                    // we can't use li %= ri because this can result in negatives
                    forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                      if ri != 0 {
                        mod(li, li, ri);
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "**=" {
                    if || reduce (ra<0) {
                      throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
                    }
                    if has_max_bits {
                      forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                        powMod(li, li, ri, local_max_size + 1);
                      }
                    }
                    else {
                      forall (li, ri) in zip(la, ra) {
                        li **= ri:uint;
                      }
                    }
                  }
                  otherwise {
                    var errorMsg = notImplementedError(pn,left.dtype,op,right.dtype);
                    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
                }
            }
            otherwise {
                var errorMsg = unrecognizedTypeError(pn,
                                  "("+dtype2str(left.dtype)+","+dtype2str(right.dtype)+")");
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        repMsg = "opeqvv success";
        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    /*
      Parse and respond to opeqvs message.
      vector op= scalar

      :arg reqMsg: request containing (cmd,op,aname,bname,rname)
      :type reqMsg: string

      :arg st: SymTab to act on
      :type st: borrowed SymTab

      :returns: (MsgTuple)
      :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.registerND
    proc opeqvsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message

        const op = msgArgs.getValueOf("op");
        const aname = msgArgs.getValueOf("a");
        const value = msgArgs.get("value");
        var dtype = str2dtype(msgArgs.getValueOf("dtype"));

        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                        "cmd: %s op: %s aname: %s dtype: %s scalar: %s".doFormat(
                                                 cmd,op,aname,dtype2str(dtype),value.getValue()));

        var left: borrowed GenSymEntry = getGenericTypedArrayEntry(aname, st);
 
        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "op: %? pdarray: %? scalar: %?".doFormat(op,st.attrib(aname),value.getValue()));
        select (left.dtype, dtype) {
            when (DType.Int64, DType.Int64) {
                var l = toSymEntry(left,int, nd);
                var val = value.getIntValue();
                select op {
                    when "+=" { l.a += val; }
                    when "-=" { l.a -= val; }
                    when "*=" { l.a *= val; }
                    when ">>=" { l.a >>= val; }
                    when "<<=" { l.a <<= val; }
                    when "//=" {
                        if val != 0 {l.a /= val;} else {l.a = 0;}
                    }//floordiv
                    when "%=" {
                        if val != 0 {l.a %= val;} else {l.a = 0;}
                    }
                    when "**=" {
                        if val<0 {
                            var errorMsg = "Attempt to exponentiate base of type Int64 to negative exponent";
                            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                                                              errorMsg);
                            return new MsgTuple(errorMsg, MsgType.ERROR);
                        }
                        else{ l.a **= val; }

                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);                         
                    }
                }
            }
            when (DType.Int64, DType.UInt64) {
                var l = toSymEntry(left,int, nd);
                var val = value.getUIntValue();
                select op {
                    when ">>=" { l.a >>= val; }
                    when "<<=" { l.a <<= val; }
                    otherwise {
                        // The result of operations between int and uint are float by default which doesn't fit in either type
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Int64, DType.Float64) {
                var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.Int64, DType.Bool) {
                var l = toSymEntry(left, int, nd);
                var val = value.getBoolValue();
                select op {
                    when "+=" {l.a += val:int;}
                    when "-=" {l.a -= val:int;}
                    when "*=" {l.a *= val:int;}
                    when ">>=" {l.a >>= val:int; }
                    when "<<=" {l.a <<= val:int; }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Int64, DType.BigInt) {
                var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.UInt64, DType.Int64) {
            var l = toSymEntry(left, uint, nd);
                var val = value.getIntValue();
                select op {
                    when ">>=" { l.a >>= val; }
                    when "<<=" { l.a <<= val; }
                    otherwise {
                        // The result of operations between int and uint are float by default which doesn't fit in either type
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.UInt64, DType.UInt64) {
                var l = toSymEntry(left,uint, nd);
                var val = value.getUIntValue();
                select op {
                    when "+=" { l.a += val; }
                    when "-=" {
                        l.a -= val;
                    }
                    when "*=" { l.a *= val; }
                    when "//=" {
                        if val != 0 {l.a /= val;} else {l.a = 0;}
                    }//floordiv
                    when "%=" {
                        if val != 0 {l.a %= val;} else {l.a = 0;}
                    }
                    when "**=" {
                        l.a **= val;
                    }
                    when ">>=" { l.a >>= val; }
                    when "<<=" { l.a <<= val; }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.UInt64, DType.Float64) {
                var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.UInt64, DType.Bool) {
                var l = toSymEntry(left, uint, nd);
                var val = value.getBoolValue();
                select op {
                    when "+=" {l.a += val:uint;}
                    when "-=" {l.a -= val:uint;}
                    when "*=" {l.a *= val:uint;}
                    when ">>=" { l.a >>= val:uint;}
                    when "<<=" { l.a <<= val:uint;}
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.UInt64, DType.BigInt) {
                var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.Float64, DType.Int64) {
                var l = toSymEntry(left,real, nd);
                var val = value.getIntValue();
                select op {
                    when "+=" {l.a += val;}
                    when "-=" {l.a -= val;}
                    when "*=" {l.a *= val;}
                    when "/=" {l.a /= val:real;} //truediv
                    when "//=" { //floordiv
                        ref la = l.a;
                        [li in la] li = floorDivisionHelper(li, val);
                    }
                    when "**=" { l.a **= val; }
                    when "%=" {
                        ref la = l.a;
                        [li in la] li = modHelper(li, val);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64, DType.UInt64) {
                var l = toSymEntry(left,real, nd);
                var val = value.getUIntValue();
                select op {
                    when "+=" { l.a += val; }
                    when "-=" { l.a -= val; }
                    when "*=" { l.a *= val; }
                    when "//=" {
                        ref la = l.a;
                        [li in la] li = floorDivisionHelper(li, val);
                    }//floordiv
                    when "**=" {
                        l.a **= val;
                    }
                    when "%=" {
                        ref la = l.a;
                        [li in la] li = modHelper(li, val);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64, DType.Float64) {
                var l = toSymEntry(left,real, nd);
                var val = value.getRealValue();
                select op {
                    when "+=" {l.a += val;}
                    when "-=" {l.a -= val;}
                    when "*=" {l.a *= val;}
                    when "/=" {l.a /= val;}//truediv
                    when "//=" { //floordiv
                        ref la = l.a;
                        [li in la] li = floorDivisionHelper(li, val);
                    }
                    when "**=" { l.a **= val; }
                    when "%=" {
                        ref la = l.a;
                        [li in la] li = modHelper(li, val);
                    }
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64, DType.Bool) {
                var l = toSymEntry(left, real, nd);
                var val = value.getBoolValue();
                select op {
                    when "+=" {l.a += val:real;}
                    when "-=" {l.a -= val:real;}
                    when "*=" {l.a *= val:real;}
                    otherwise {
                        var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                }
            }
            when (DType.Float64, DType.BigInt) {
                var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.BigInt, DType.Int64) {
                var l = toSymEntry(left,bigint, nd);
                var val = value.getIntValue();
                ref la = l.a;
                var max_bits = l.max_bits;
                var max_size = 1:bigint;
                var has_max_bits = max_bits != -1;
                if has_max_bits {
                  max_size <<= max_bits;
                  max_size -= 1;
                }
                select op {
                  when "+=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      li += local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "-=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      li -= local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "*=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      li *= local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "//=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      if local_val != 0 {
                        li /= local_val;
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "%=" {
                    // we can't use li %= val because this can result in negatives
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      if local_val != 0 {
                        mod(li, li, local_val);
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "**=" {
                    if val<0 {
                      throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
                    }
                    if has_max_bits {
                      forall li in la with (var local_val = val, var local_max_size = max_size) {
                        powMod(li, li, local_val, local_max_size + 1);
                      }
                    }
                    else {
                      forall li in la with (var local_val = val) {
                        li **= local_val:uint;
                      }
                    }
                  }
                  otherwise {
                    var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
                }
            }
            when (DType.BigInt, DType.UInt64) {
                var l = toSymEntry(left,bigint, nd);
                var val = value.getUIntValue();
                ref la = l.a;
                var max_bits = l.max_bits;
                var max_size = 1:bigint;
                var has_max_bits = max_bits != -1;
                if has_max_bits {
                  max_size <<= max_bits;
                  max_size -= 1;
                }
                select op {
                  when "+=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      li += local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "-=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      li -= local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "*=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      li *= local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "//=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      if local_val != 0 {
                        li /= local_val;
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "%=" {
                    // we can't use li %= val because this can result in negatives
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      if local_val != 0 {
                        mod(li, li, local_val);
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "**=" {
                    if val<0 {
                      throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
                    }
                    if has_max_bits {
                      forall li in la with (var local_val = val, var local_max_size = max_size) {
                        powMod(li, li, local_val, local_max_size + 1);
                      }
                    }
                    else {
                      forall li in la with (var local_val = val) {
                        li **= local_val:uint;
                      }
                    }
                  }
                  otherwise {
                    var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
                }
            }
            when (DType.BigInt, DType.Float64) {
                var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            when (DType.BigInt, DType.Bool) {
                var l = toSymEntry(left, bigint, nd);
                var val = value.getBoolValue();
                ref la = l.a;
                var max_bits = l.max_bits;
                var max_size = 1:bigint;
                var has_max_bits = max_bits != -1;
                if has_max_bits {
                  max_size <<= max_bits;
                  max_size -= 1;
                }
                select op {
                  // TODO change once we can cast directly from bool to bigint
                  when "+=" {
                    forall li in la with (var local_val = val:int:bigint, var local_max_size = max_size) {
                      li += local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "-=" {
                    forall li in la with (var local_val = val:int:bigint, var local_max_size = max_size) {
                      li -= local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "*=" {
                    forall li in la with (var local_val = val:int:bigint, var local_max_size = max_size) {
                      li *= local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  otherwise {
                    var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
                }
            }
            when (DType.BigInt, DType.BigInt) {
                var l = toSymEntry(left,bigint, nd);
                var val = value.getBigIntValue();
                ref la = l.a;
                var max_bits = l.max_bits;
                var max_size = 1:bigint;
                var has_max_bits = max_bits != -1;
                if has_max_bits {
                  max_size <<= max_bits;
                  max_size -= 1;
                }
                select op {
                  when "+=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      li += local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "-=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      li -= local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "*=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      li *= local_val;
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "//=" {
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      if local_val != 0 {
                        li /= local_val;
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "%=" {
                    // we can't use li %= val because this can result in negatives
                    forall li in la with (var local_val = val, var local_max_size = max_size) {
                      if local_val != 0 {
                        mod(li, li, local_val);
                      }
                      else {
                        li = 0:bigint;
                      }
                      if has_max_bits {
                        li &= local_max_size;
                      }
                    }
                  }
                  when "**=" {
                    if val<0 {
                      throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
                    }
                    if has_max_bits {
                      forall li in la with (var local_val = val, var local_max_size = max_size) {
                        powMod(li, li, local_val, local_max_size + 1);
                      }
                    }
                    else {
                      forall li in la with (var local_val = val) {
                        li **= local_val:uint;
                      }
                    }
                  }
                  otherwise {
                    var errorMsg = notImplementedError(pn,left.dtype,op,dtype);
                    omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
                }
            }
            otherwise {
              var errorMsg = unrecognizedTypeError(pn,
                                  "("+dtype2str(left.dtype)+","+dtype2str(dtype)+")");
              omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        repMsg = "opeqvs success";
        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
}
