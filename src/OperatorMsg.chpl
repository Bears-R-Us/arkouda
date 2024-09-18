
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

    use Time;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const omLogger = new Logger(logLevel, logChannel);

    /*
      Parse and respond to binopvv message.
      vv == vector op vector

      :arg reqMsg: request containing (cmd,op,aname,bname)
      :type reqMsg: string 

      :arg st: SymTab to act on
      :type st: borrowed SymTab 

      :returns: (MsgTuple) 
      :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.instantiateAndRegister
    proc binopvv(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        const l = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
              r = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
              op = msgArgs['op'].toScalar(string);

        omLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), 
             "cmd: %? op: %? left pdarray: %? right pdarray: %?".format(
                                          cmd,op,st.attrib(msgArgs['a'].val),
                                          st.attrib(msgArgs['b'].val)));

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

        if binop_dtype_a == int && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          } else if op == "/" {
            // True division is the only case in this int, int case
            // that results in a `real` symbol table entry.
            return doBinOpvv(l, r, real, op, pn, st);
          }
          return doBinOpvv(l, r, int, op, pn, st);
        } else if binop_dtype_a == int && binop_dtype_b == real {
          // Only two possible resultant types are `bool` and `real`
          // for this case
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, real, op, pn, st);
        }
        else if binop_dtype_a == uint && binop_dtype_b == real {
          // Only two possible resultant types are `bool` and `real`
          // for this case
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == real {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, real, op, pn, st);
        }
        // For cases where a boolean operand is involved, the only
        // possible resultant type is `bool`
        else if binop_dtype_a == bool && binop_dtype_b == bool {
          if (op == "<<") || (op == ">>" ) {
            return doBinOpvv(l, r, int, op, pn, st);
          }
          return doBinOpvv(l, r, bool, op, pn, st);
        }
        else if binop_dtype_a == bool && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, int, op, pn, st);
        }
        else if binop_dtype_a == int && binop_dtype_b == bool {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, int, op, pn, st);
        }
        else if binop_dtype_a == bool && binop_dtype_b == real {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == bool {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, real, op, pn, st);
        }
        else if binop_dtype_a == bool && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, uint, op, pn, st);
        }
        else if binop_dtype_a == uint && binop_dtype_b == bool {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          return doBinOpvv(l, r, uint, op, pn, st);
        }
        else if binop_dtype_a == uint && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          if op == "/"{
            return doBinOpvv(l, r, real, op, pn, st);
          } else {
            return doBinOpvv(l, r, uint, op, pn, st);
          }
        }
        else if binop_dtype_a == uint && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpvv(l, r , bool, op, pn, st);
          }
          // +, -, /, // both result in real outputs to match NumPy
          if realOps.contains(op) {
            return doBinOpvv(l, r, real, op, pn, st);
          } else {
            // isn't +, -, /, // so we can use LHS to determine type
            return doBinOpvv(l, r, uint, op, pn, st);
          }
        }
        else if binop_dtype_a == int && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpvv(l, r, bool, op, pn, st);
          }
          // +, -, /, // both result in real outputs to match NumPy
          if realOps.contains(op) {
            return doBinOpvv(l, r, real, op, pn, st);
          } else {
            // isn't +, -, /, // so we can use LHS to determine type
            return doBinOpvv(l, r, int, op, pn, st);
          }
        }
        else if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b)
        {
          if boolOps.contains(op) {
            // call bigint specific func which returns distr bool array
            return st.insert(new shared SymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
          }
          // call bigint specific func which returns dist bigint array
          const (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
          return st.insert(new shared SymEntry(tmp, max_bits));
        } else {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
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
    @arkouda.instantiateAndRegister
    proc binopvs(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        const l = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
              val = msgArgs['value'].toScalar(binop_dtype_b),
              op = msgArgs['op'].toScalar(string);

        omLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
             "cmd: %? op: %? left pdarray: %? scalar: %?".format(
                                          cmd,op,st.attrib(msgArgs['a'].val), val));

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

        if binop_dtype_a == int && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          } else if op == "/" {
            // True division is the only case in this int, int case
            // that results in a `real` symbol table entry.
            return doBinOpvs(l, val, real, op, pn, st);
          }
          return doBinOpvs(l, val, int, op, pn, st);
        } else if binop_dtype_a == int && binop_dtype_b == real {
          // Only two possible resultant types are `bool` and `real`
          // for this case
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, real, op, pn, st);
        }
        else if binop_dtype_a == uint && binop_dtype_b == real {
          // Only two possible resultant types are `bool` and `real`
          // for this case
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == real {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, real, op, pn, st);
        }
        // For cases where a boolean operand is involved, the only
        // possible resultant type is `bool`
        else if binop_dtype_a == bool && binop_dtype_b == bool {
          if (op == "<<") || (op == ">>") {
            return doBinOpvs(l, val, int, op, pn, st);
          }
          return doBinOpvs(l, val, bool, op, pn, st);
        }
        else if binop_dtype_a == bool && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, int, op, pn, st);
        }
        else if binop_dtype_a == int && binop_dtype_b == bool {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, int, op, pn, st);
        }
        else if binop_dtype_a == bool && binop_dtype_b == real {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == bool {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, real, op, pn, st);
        }
        else if binop_dtype_a == bool && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, uint, op, pn, st);
        }
        else if binop_dtype_a == uint && binop_dtype_b == bool {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          return doBinOpvs(l, val, uint, op, pn, st);
        }
        else if binop_dtype_a == uint && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          if op == "/"{
            return doBinOpvs(l, val, real, op, pn, st);
          } else {
            return doBinOpvs(l, val, uint, op, pn, st);
          }
        }
        else if binop_dtype_a == uint && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          // +, -, /, // both result in real outputs to match NumPy
          if realOps.contains(op) {
            return doBinOpvs(l, val, real, op, pn, st);
          } else {
            // isn't +, -, /, // so we can use LHS to determine type
            return doBinOpvs(l, val, uint, op, pn, st);
          }
        }
        else if binop_dtype_a == int && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpvs(l, val, bool, op, pn, st);
          }
          // +, -, /, // both result in real outputs to match NumPy
          if realOps.contains(op) {
            return doBinOpvs(l, val, real, op, pn, st);
          } else {
            // isn't +, -, /, // so we can use LHS to determine type
            return doBinOpvs(l, val, int, op, pn, st);
          }
        }
        else if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b)
        {
          if boolOps.contains(op) {
            // call bigint specific func which returns distr bool array
            return st.insert(new shared SymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
          }
          // call bigint specific func which returns dist bigint array
          const (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
          return st.insert(new shared SymEntry(tmp, max_bits));
        } else {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return MsgTuple.error(errorMsg);
        }
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
    @arkouda.instantiateAndRegister
    proc binopsv(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        const r = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
              val = msgArgs['value'].toScalar(binop_dtype_b),
              op = msgArgs['op'].toScalar(string);

        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                 "cmd: %? op = %? scalar dtype = %? scalar = %? pdarray = %?".format(
                                   cmd,op,type2str(binop_dtype_b),msgArgs['value'].val,st.attrib(msgArgs['a'].val)));

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

        if binop_dtype_a == int && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          } else if op == "/" {
            // True division is the only case in this int, int case
            // that results in a `real` symbol table entry.
            return doBinOpsv(val, r, real, op, pn, st);
          }
          return doBinOpsv(val, r, int, op, pn, st);
        }
        else if binop_dtype_a == int && binop_dtype_b == real {
          // Only two possible resultant types are `bool` and `real`
          // for this case
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, real, op, pn, st);
        }
        else if binop_dtype_a == uint && binop_dtype_b == real {
          // Only two possible resultant types are `bool` and `real`
          // for this case
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == real {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, real, op, pn, st);
        }
        // For cases where a boolean operand is involved, the only
        // possible resultant type is `bool`
        else if binop_dtype_a == bool && binop_dtype_b == bool {
          if (op == "<<") || (op == ">>") {
            return doBinOpsv(val, r, int, op, pn, st);
          }
          return doBinOpsv(val, r, bool, op, pn, st);
        }
        else if binop_dtype_a == bool && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, int, op, pn, st);
        }
        else if binop_dtype_a == int && binop_dtype_b == bool {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, int, op, pn, st);
        }
        else if binop_dtype_a == bool && binop_dtype_b == real {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, real, op, pn, st);
        }
        else if binop_dtype_a == real && binop_dtype_b == bool {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, real, op, pn, st);
        }
        else if binop_dtype_a == bool && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, uint, op, pn, st);
        }
        else if binop_dtype_a == uint && binop_dtype_b == bool {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          return doBinOpsv(val, r, uint, op, pn, st);
        }
        else if binop_dtype_a == uint && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          if op == "/"{
            return doBinOpsv(val, r, real, op, pn, st);
          } else {
            return doBinOpsv(val, r, uint, op, pn, st);
          }
        }
        else if binop_dtype_a == uint && binop_dtype_b == int {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          // +, -, /, // both result in real outputs to match NumPy
          if realOps.contains(op) {
            return doBinOpsv(val, r, real, op, pn, st);
          } else {
            // isn't +, -, /, // so we can use LHS to determine type
            return doBinOpsv(val, r, uint, op, pn, st);
          }
        }
        else if binop_dtype_a == int && binop_dtype_b == uint {
          if boolOps.contains(op) {
            return doBinOpsv(val, r, bool, op, pn, st);
          }
          // +, -, /, // both result in real outputs to match NumPy
          if realOps.contains(op) {
            return doBinOpsv(val, r, real, op, pn, st);
          } else {
            // isn't +, -, /, // so we can use LHS to determine type
            return doBinOpsv(val, r, int, op, pn, st);
          }
        }
        else if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b) {
          if boolOps.contains(op) {
            return st.insert(new shared SymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
          }
          // call bigint specific func which returns dist bigint array
          const (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
          return st.insert(new shared SymEntry(tmp, max_bits));
        } else {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return MsgTuple.error(errorMsg);
        }
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
    @arkouda.instantiateAndRegister
    proc opeqvv(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        var l = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd);
        const r = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
              op = msgArgs['op'].toScalar(string),
              nie = notImplementedError(pn,type2str(binop_dtype_a),op,type2str(binop_dtype_b));

        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "cmd: %s op: %s left pdarray: %s right pdarray: %s".format(cmd,op,
                    st.attrib(msgArgs['a'].val),st.attrib(msgArgs['b'].val)));

        if binop_dtype_a == int && binop_dtype_b == int  {
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
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == int && binop_dtype_b == bool  {
            select op {
                when "+=" {l.a += r.a:int;}
                when "-=" {l.a -= r.a:int;}
                when "*=" {l.a *= r.a:int;}
                when ">>=" { l.a >>= r.a:int;}
                when "<<=" { l.a <<= r.a:int;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == uint && binop_dtype_b == uint  {
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
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == uint && binop_dtype_b == bool  {
            select op {
                when "+=" {l.a += r.a:uint;}
                when "-=" {l.a -= r.a:uint;}
                when "*=" {l.a *= r.a:uint;}
                when ">>=" { l.a >>= r.a:uint;}
                when "<<=" { l.a <<= r.a:uint;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == int  {
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
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == uint  {
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
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == real  {
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
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == bool  {
            select op {
                when "+=" {l.a += r.a:real;}
                when "-=" {l.a -= r.a:real;}
                when "*=" {l.a *= r.a:real;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bool && binop_dtype_b == bool  {
            select op {
                when "|=" {l.a |= r.a;}
                when "&=" {l.a &= r.a;}
                when "^=" {l.a ^= r.a;}
                when "+=" {l.a |= r.a;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bigint && binop_dtype_b == int  {
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
              otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bigint && binop_dtype_b == uint  {
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
              otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bigint && binop_dtype_b == bool  {
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
              otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bigint && binop_dtype_b == bigint  {
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
              otherwise do return MsgTuple.error(nie);
            }
          } else {
            return MsgTuple.error(nie);
          }

        return MsgTuple.success();
    }

    /*
      Parse and respond to opeqvs message.
      vector op= scalar

      :arg reqMsg: request containing (cmd,op,aname,bname)
      :type reqMsg: string

      :arg st: SymTab to act on
      :type st: borrowed SymTab

      :returns: (MsgTuple)
      :throws: `UndefinedSymbolError(name)`
    */
    @arkouda.instantiateAndRegister
    proc opeqvs(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
        param pn = Reflection.getRoutineName();

        var l = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd);
        const val = msgArgs['value'].toScalar(binop_dtype_b),
              op = msgArgs['op'].toScalar(string),
              nie = notImplementedError(pn,type2str(binop_dtype_a),op,type2str(binop_dtype_b));

        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "op: %? pdarray: %? scalar: %?".format(op,st.attrib(msgArgs['a'].val),val));

        if binop_dtype_a == int && binop_dtype_b == int  {
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
                        var errorMsg = "Attempt to exponentiate base of type int64 to negative exponent";
                        omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                                                                          errorMsg);
                        return new MsgTuple(errorMsg, MsgType.ERROR);
                    }
                    else{ l.a **= val; }

                }
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == int && binop_dtype_b == uint  {
            select op {
                when ">>=" { l.a >>= val; }
                when "<<=" { l.a <<= val; }
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == int && binop_dtype_b == bool  {
            select op {
                when "+=" {l.a += val:int;}
                when "-=" {l.a -= val:int;}
                when "*=" {l.a *= val:int;}
                when ">>=" {l.a >>= val:int; }
                when "<<=" {l.a <<= val:int; }
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == uint && binop_dtype_b == int  {
            select op {
                when ">>=" { l.a >>= val; }
                when "<<=" { l.a <<= val; }
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == uint && binop_dtype_b == uint  {
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
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == uint && binop_dtype_b == bool  {
            select op {
                when "+=" {l.a += val:uint;}
                when "-=" {l.a -= val:uint;}
                when "*=" {l.a *= val:uint;}
                when ">>=" { l.a >>= val:uint;}
                when "<<=" { l.a <<= val:uint;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bool && binop_dtype_b == bool  {
            select op {
                when "+=" {l.a |= val;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == int  {
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
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == uint  {
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
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == real  {
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
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == bool  {
            select op {
                when "+=" {l.a += val:real;}
                when "-=" {l.a -= val:real;}
                when "*=" {l.a *= val:real;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bigint && binop_dtype_b == int  {
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
              otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bigint && binop_dtype_b == uint  {
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
              otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bigint && binop_dtype_b == bool  {
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
              otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bigint && binop_dtype_b == bigint  {
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
              otherwise do return MsgTuple.error(nie);
            }
        }
        else {
          return MsgTuple.error(nie);
        }
        return MsgTuple.success();
    }
}
