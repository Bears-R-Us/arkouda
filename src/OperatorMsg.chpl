
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

    proc isSmallType(type t): bool {
      return t == bool || t == uint(8) || t == uint(16) || t == int(8) || t == int(16);
    }

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

        // At this point it should handle almost every bigint case

        if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b) &&
                op != '/'
        {
          if boolOps.contains(op) {
            return st.insert(new shared SymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
          }
          // call bigint specific func which returns dist bigint array
          const (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
          return st.insert(new shared SymEntry(tmp, max_bits));
        } else if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                    (isRealType(binop_dtype_a) || isRealType(binop_dtype_b)) &&
                    boolOps.contains(op) {
          // call bigint specific func which returns distr bool array
          return st.insert(new shared SymEntry(doBigIntBinOpvvBoolReturnRealInput(l.a, r.a, op)));
        }
        else if (binop_dtype_a == bigint || binop_dtype_b == bigint) {
          return st.insert(new shared SymEntry(doBigIntBinOpvvRealReturn(l, r, op)));
        }

        if boolOps.contains(op) {
          return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
        }

        if op == "/" {
          if binop_dtype_a == real(32) && isSmallType(binop_dtype_b) {
            return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          if binop_dtype_b == real(32) && isSmallType(binop_dtype_a) {
            return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, real(64), op, pn, st);
        }

        type returnType = mySafeCast(binop_dtype_a, binop_dtype_b);

        if (!realOps.contains(op)) && (returnType == real(32) || returnType == real(64)) {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if returnType == bool {
          if op == "+" || op == "*" || (!realOps.contains(op)) {
            return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
          }
          if op == "-" {
            const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
          if op == "//" || op == "%" || op == "**" {
            return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, uint(8), op, pn, st);
          }
        }

        return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, returnType, op, pn, st);

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

        // At this point it should handle almost every bigint case

        if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b) &&
                op != '/'
        {
          if boolOps.contains(op) {
            return st.insert(new shared SymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
          }
          // call bigint specific func which returns dist bigint array
          const (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
          return st.insert(new shared SymEntry(tmp, max_bits));
        } else if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                    (isRealType(binop_dtype_a) || isRealType(binop_dtype_b)) &&
                    boolOps.contains(op) {
          // call bigint specific func which returns distr bool array
          return st.insert(new shared SymEntry(doBigIntBinOpvsBoolReturnRealInput(l.a, val, op)));
        }
        else if (binop_dtype_a == bigint || binop_dtype_b == bigint) {
          return st.insert(new shared SymEntry(doBigIntBinOpvsRealReturn(l, val, op)));
        }

        if boolOps.contains(op) {
          return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
        }

        if op == "/" {
          if binop_dtype_a == real(32) && isSmallType(binop_dtype_b) {
            return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          if binop_dtype_b == real(32) && isSmallType(binop_dtype_a) {
            return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, real(64), op, pn, st);
        }

        type returnType = mySafeCast(binop_dtype_a, binop_dtype_b);

        if (!realOps.contains(op)) && (returnType == real(32) || returnType == real(64)) {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if returnType == bool {
          if op == "+" || op == "*" || (!realOps.contains(op)) {
            return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
          }
          if op == "-" {
            const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
          if op == "//" || op == "%" || op == "**" {
            return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, uint(8), op, pn, st);
          }
        }

        return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, returnType, op, pn, st);

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

        // At this point it should handle almost every bigint case

        if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b) &&
                op != '/'
        {
          if boolOps.contains(op) {
            return st.insert(new shared SymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
          }
          // call bigint specific func which returns dist bigint array
          const (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
          return st.insert(new shared SymEntry(tmp, max_bits));
        } else if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                    (isRealType(binop_dtype_a) || isRealType(binop_dtype_b)) &&
                    boolOps.contains(op) {
          // call bigint specific func which returns distr bool array
          return st.insert(new shared SymEntry(doBigIntBinOpsvBoolReturnRealInput(val, r.a, op)));
        }
        else if (binop_dtype_a == bigint || binop_dtype_b == bigint) {
          return st.insert(new shared SymEntry(doBigIntBinOpsvRealReturn(val, r, op)));
        }

        if boolOps.contains(op) {
          return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
        }

        if op == "/" {
          if binop_dtype_a == real(32) && isSmallType(binop_dtype_b) {
            return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          if binop_dtype_b == real(32) && isSmallType(binop_dtype_a) {
            return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, real(64), op, pn, st);
        }

        type returnType = mySafeCast(binop_dtype_a, binop_dtype_b);

        if (!realOps.contains(op)) && (returnType == real(32) || returnType == real(64)) {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if returnType == bool {
          if op == "+" || op == "*" || (!realOps.contains(op)) {
            return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
          }
          if op == "-" {
            const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
          if op == "//" || op == "%" || op == "**" {
            return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, uint(8), op, pn, st);
          }
        }

        return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, returnType, op, pn, st);

    }

    // --- NumPy-ish casting helpers for in-place ufunc semantics (casting='same_kind') ---
    // These are because casting is a little different for things like += (vs. +)

    proc isArithInplaceOp(op: string): bool {
      return op == "+=" || op == "-=" || op == "*=" || op == "/=" ||
            op == "//=" || op == "%=" || op == "**=";
    }

    proc isBitInplaceOp(op: string): bool {
      return op == "&=" || op == "|=" || op == "^=" ||
            op == "<<=" || op == ">>=";
    }

    proc numpyLikeOpeqGate(type lhsT, type rhsT, op: string): int {
      const isArith = isArithInplaceOp(op);
      const isBit   = isBitInplaceOp(op);
      if !isArith && !isBit then return 3;

      param kL = splitType(lhsT);
      param kR = splitType(rhsT);

      // --- True division (/=) special rule (NumPy in-place semantics) ---
      // Only float LHS supports /= in-place in an array.
      // (Integers/bool fail because result is float; bigint-as-bigint can't hold float.)
      if op == "/=" {
        if lhsT == real(64) {
          // float LHS: allow unless RHS is bigint (you already treat that as UFuncTypeError)
          if kR == 4 then return 2;
          // RHS bool/int/uint/real are fine
          return 0;
        } else {
          // int/uint/bool/bigint LHS: in-place true divide rejects (NumPy UFuncTypeError)
          return 2;
        }
      }

      // Float LHS: arithmetic ok, bitwise/shifts TypeError; float op= bigint => UFuncTypeError
      if lhsT == real(64) {
        if kR == 4 then return 2;
        return if isBit then 1 else 0;
      }

      // Bool LHS special-case
      if lhsT == bool {
        if rhsT != bool then return 2;
        // bool op= bool:
        if op == "-=" then return 1;
        if op == "+=" || op == "*=" || op == "&=" || op == "|=" || op == "^=" then return 0;
        return 2; // //= %= **= <<= >>= are UFuncTypeError in your matrix
      }

      // Non-bigint LHS with bigint RHS => UFuncTypeError (cannot cast object/bigint result back)
      if kL != 4 && kR == 4 {
        return 2;
      }

      // Bigint LHS rules
      if kL == 4 {
        // allow **= with int/uint/bool/bigint on rhs. That's integer-like.
        // For float rhs: cannot keep bigint dtype => TypeError
        if kR == 3 then return 1;
        return 0;
      }

      // Now we are in {int64,uint64,uint8} LHS cases.
      // Mixed int64/uint64:
      if lhsT == int(64) && rhsT == uint(64) {
        return if isBit then 1 else 2;
      }
      if lhsT == uint(64) && rhsT == int(64) {
        return if isBit then 1 else 2;
      }

      // uint8 op= int64 => all UFuncTypeError
      if lhsT == uint(8) && rhsT == int(64) {
        return 2;
      }

      // integer LHS with float RHS => arithmetic UFuncTypeError, bit TypeError
      if kR == 3 {
        return if isBit then 1 else 2;
      }

      // Otherwise allow (covers i64 op= i64/u8/b; u64 op= u64/u8/b; u8 op= u8/u64/b)
      return 0;
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

      const gate = numpyLikeOpeqGate(binop_dtype_a, binop_dtype_b, op);
      param kL = splitType(binop_dtype_a);
      param kR = splitType(binop_dtype_b);
      ref la = l.a;
      select gate {
        when 0 { 

          // ---- bool/bool special-case (NumPy quirks) ----

          if kL == 0 && kR == 0 {
            select op {
              when "+=" { l.a = l.a | r.a; return MsgTuple.success(); }
              when "*=" { l.a = l.a & r.a; return MsgTuple.success(); }
              when "&=" { l.a &= r.a; return MsgTuple.success(); }
              when "|=" { l.a |= r.a; return MsgTuple.success(); }
              when "^=" { l.a ^= r.a; return MsgTuple.success(); }
              when "-=" { return new MsgTuple("TypeError", MsgType.ERROR); }
              otherwise { return new MsgTuple("TypeError", MsgType.ERROR); }
            }
          }

          // If we are instantiated with bool LHS, the only supported RHS is bool,
          // and that case returned above. Prevent the compiler from typechecking
          // the generic arithmetic/bitwise code for bool LHS instantiations.
          if kL == 0 {
            // matches numpyLikeOpeqGate for bool with non-bool RHS
            return new MsgTuple("TypeError", MsgType.ERROR);
          }

          // ---- general path (gate==0 and not bool/bool) ----
          // If instantiated with bigint LHS and real RHS, we don't support it (and we must
          // prevent the compiler from typechecking casts from real->bigint).
          if kL == 4 && kR == 3 {
            return new MsgTuple("TypeError", MsgType.ERROR);
          }
          const ra = r.a;
          var handled = true;

          // splitType: 0 bool, 1 uint, 2 int, 3 real, 4 bigint

          if isArithInplaceOp(op) {

            if op == "+=" {
              la += (ra: binop_dtype_a);

            } else if op == "-=" {
              la -= (ra: binop_dtype_a);

            } else if op == "*=" {
              la *= (ra: binop_dtype_a);

            } else if op == "**=" {
              if binop_dtype_a == int(64) {
                if || reduce ((r.a: int(64)) < 0) {
                  return new MsgTuple(
                    "Attempt to exponentiate base of type Int64 to negative exponent",
                    MsgType.ERROR
                  );
                }
              }
              // la **= (ra: binop_dtype_a);

              if kL == 4 && l.max_bits != -1 {
                const max_size = (1: bigint << l.max_bits);
                forall (t, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  powMod(t, t, ri, max_size);
                }
              } else {
                try {
                  forall (t, ri) in zip(la, ra) do t **= ri:binop_dtype_a;
                } catch {
                  return new MsgTuple (
                    "Exponentiation too large; use smaller values or set max_bits",
                    MsgType.ERROR
                  );
                }
              }

            } else if op == "//=" {

              if kL == 3 && binop_dtype_a == real(64) {
                // NumPy-like float floor-division
                const rb = (r.a: real(64));
                [(li, ri) in zip(la, rb)] li = floorDivisionHelper(li, ri);

              } else {
                // integer/bool/uint/bigint style, preserve div-by-zero->0 behavior
                ref la2 = l.a;
                const rb = (r.a: binop_dtype_a);
                [(li, ri) in zip(la2, rb)] li = if ri != 0 then li/ri else (0: binop_dtype_a);
              }

            } else if op == "%=" {

              if kL == 3 && binop_dtype_a == real(64) {
                // NumPy-like float modulo
                const rb = (r.a: real(64));
                [(li, ri) in zip(la, rb)] li = modHelper(li, ri);

              } else {
                // integer/bool/uint/bigint modulo, preserve div-by-zero->0 behavior
                ref la2 = l.a;
                const rb = (r.a: binop_dtype_a);
                [(li, ri) in zip(la2, rb)] li = if ri != 0 then li%ri else (0: binop_dtype_a);
              }

            } else if op == "/=" {
              if kL == 3 && binop_dtype_a == real(64) {
                // Only float LHS should reach here due to the gate.
                // NumPy behavior for float division by zero is inf/nan (with warnings).
                const rb = (r.a: real(64));
                [(li, ri) in zip(la, rb)] li = li / ri;
              }
            } else {
              handled = false;
            }

          } else if isBitInplaceOp(op) {

            if kL == 4 {

              if op == ">>=" || op == "<<=" {

                // If RHS isn't already bigint (possible in vv), cast to bigint *only if allowed by gate*.
                // But in practice, the gate should only let through integral-ish RHS for shifts.
                const rb = (r.a: bigint);

                // Validate shift counts: non-negative and fits in int
                const maxShift = max(int): bigint;

                // Fast reject with reductions so you don't partially mutate la
                if || reduce (rb < 0: bigint) {
                  return new MsgTuple("ValueError: negative shift count", MsgType.ERROR);
                }
                if || reduce (rb > maxShift) {
                  return new MsgTuple("ValueError: shift count too large", MsgType.ERROR);
                }

                // Apply shift (elementwise)
                if op == ">>=" {
                  forall (li, ri) in zip(la, rb) {
                    li >>= (ri:int);
                  }
                } else {
                  forall (li, ri) in zip(la, rb) {
                    li <<= (ri:int);
                  }
                }

              } else {
                select op {
                  when "&="  { la &=  (ra: binop_dtype_a); }
                  when "|="  { la |=  (ra: binop_dtype_a); }
                  when "^="  { la ^=  (ra: binop_dtype_a); }
                  otherwise { handled = false; }
                }
              }
            }

            // Bitwise + shifts must NOT compile for real LHS
            else if (kL == 0 || kL == 1 || kL == 2) {
              select op {
                when ">>=" { la >>= (ra: binop_dtype_a); }
                when "<<=" { la <<= (ra: binop_dtype_a); }
                when "&="  { la &=  (ra: binop_dtype_a); }
                when "|="  { la |=  (ra: binop_dtype_a); }
                when "^="  { la ^=  (ra: binop_dtype_a); }
                otherwise { handled = false; }
              }
            } else {
              // real LHS should not reach here; gate should have blocked it
              return new MsgTuple("TypeError", MsgType.ERROR);
            }

          } else {
            handled = false;
          }

          if !handled then return MsgTuple.error(nie);
            if kL == 4 && l.max_bits != -1 {
              const mask = (1: bigint << l.max_bits) - 1;
              la &= mask;
            }
          return MsgTuple.success();


        }
        when 1 { return new MsgTuple("TypeError", MsgType.ERROR); }
        when 2 { return new MsgTuple("TypeError", MsgType.ERROR); } // Technically numpy views these
                                                                    // as two different kinds of
                                                                    // TypeError
        otherwise { return MsgTuple.error(nie); }
      }

      return MsgTuple.success();

    }

    /*
      Parse and respond to opeqvs message.
      vector op= scalar

      scalar must be a scalar of the same type as the vector,
      unless the vector is a bigint

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
                param array_nd: int
    ): MsgTuple throws {
      param pn = Reflection.getRoutineName();

      // RHS is always the same type as LHS (typed scalar semantics)
      type binop_dtype_b = binop_dtype_a;

      var l = st[msgArgs["a"]]: borrowed SymEntry(binop_dtype_a, array_nd);
      const val = msgArgs["value"].toScalar(binop_dtype_b),
            op  = msgArgs["op"].toScalar(string),
            nie = notImplementedError(pn, type2str(binop_dtype_a), op, type2str(binop_dtype_b));

      omLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                    "op: %? pdarray: %? scalar: %?".format(op, st.attrib(msgArgs["a"].val), val));

      // Keep the same gate already built. Here lhsT == rhsT, so it will mostly:
      // - block float bitwise/shift
      // - enforce bool quirks (-=, etc.)
      // - block int/uint /= (true-div) etc.
      const gate = numpyLikeOpeqGate(binop_dtype_a, binop_dtype_b, op);
      param kL = splitType(binop_dtype_a); // 0 bool, 1 uint, 2 int, 3 real, 4 bigint

      ref la = l.a;

      select gate {
        when 0 {

          // -----------------------------
          // bool/bool special-case (NumPy quirks)
          // -----------------------------
          if kL == 0 {
            // Only bool scalar is possible due to binop_dtype_b == binop_dtype_a,
            // but keep this explicit and mirror opeqvv.
            select op {
              when "+=" { la |= val; return MsgTuple.success(); } // bool += bool -> OR
              when "*=" { la &= val; return MsgTuple.success(); } // bool *= bool -> AND
              when "&=" { la &= val; return MsgTuple.success(); }
              when "|=" { la |= val; return MsgTuple.success(); }
              when "^=" { la ^= val; return MsgTuple.success(); }
              when "-=" { return new MsgTuple("TypeError", MsgType.ERROR); }
              otherwise { return new MsgTuple("TypeError", MsgType.ERROR); }
            }
          }

          var handled = true;

          // -----------------------------
          // arithmetic inplace ops
          // -----------------------------
          if isArithInplaceOp(op) {

            if op == "+=" {
              la += val;

            } else if op == "-=" {
              la -= val;

            } else if op == "*=" {
              la *= val;

            } else if op == "/=" {
              // With rhs==lhs, gate should only allow real(64) here.
              if kL == 3 && binop_dtype_a == real(64) {
                // NumPy-like: inf/nan on div-by-zero (warnings at Python layer, if any)
                la /= val;
              } else {
                return new MsgTuple("TypeError", MsgType.ERROR);
              }

            } else if op == "**=" {

              // int64 negative exponent check
              if binop_dtype_a == int(64) {
                if val: int(64) < 0 {
                  return new MsgTuple(
                    "Attempt to exponentiate base of type Int64 to negative exponent",
                    MsgType.ERROR
                  );
                }
              }
              // bigint negative exponent check
              if kL == 4 {
                if val: bigint < 0 {
                  return new MsgTuple(
                    "Attempt to exponentiate base of type BigInt to negative exponent",
                    MsgType.ERROR
                  );
                }

                // preserve max_bits semantics
                if l.max_bits != -1 {
                  const max_size = (1: bigint << l.max_bits);
                  forall t in la with (var local_val = val: bigint,
                                      var local_max_size = max_size) {
                    powMod(t, t, local_val, local_max_size);
                  }
                } else {
                  try {
                    forall t in la with (var local_val = val: bigint) {
                      // existing code uses uint exponent when no max_bits
                      t **= local_val;
                    }
                  } catch {
                    return new MsgTuple(
                      "Exponentiation too large; use smaller values or set max_bits",
                      MsgType.ERROR
                    );
                  }
                }

              } else {
                // non-bigint normal exponentiation
                la **= val;
              }

            } else if op == "//=" {

              if kL == 3 && binop_dtype_a == real(64) {
                // NumPy-like float floor-division
                [li in la] li = floorDivisionHelper(li, val: real(64));
              } else {
                // int/uint/bigint style, preserve div-by-zero->0 behavior
                if val != 0 {
                  la /= val;
                } else {
                  la = 0: binop_dtype_a;
                }
              }

            } else if op == "%=" {

              if kL == 3 && binop_dtype_a == real(64) {
                // NumPy-like float modulo
                [li in la] li = modHelper(li, val: real(64));
              } else if kL == 4 {
                // Bigint modulo: avoid li %= val (can go negative).
                forall li in la with (var local_val = val: bigint) {
                  if local_val != 0 {
                    mod(li, li, local_val);
                  } else {
                    li = 0: bigint;
                  }
                }
              } else {
                // int/uint modulo, preserve div-by-zero->0
                if val != 0 {
                  la %= val;
                } else {
                  la = 0: binop_dtype_a;
                }
              }

            } else {
              handled = false;
            }

          // -----------------------------
          // bitwise inplace ops
          // -----------------------------
          } else if isBitInplaceOp(op) {

            // Gate should have rejected real here. Keep defensive check.
            if kL == 3 {
              return new MsgTuple("TypeError", MsgType.ERROR);
            }

            if kL == 4 && (op == ">>=" || op == "<<=") {
              // Convert bigint -> int shift count (reject negative / too large)
              if val < 0: bigint then
                return new MsgTuple("ValueError: negative shift count", MsgType.ERROR);

              // pick a bound you consider safe; at minimum, must fit in int
              const maxShift = max(int): bigint;
              if val > maxShift then
                return new MsgTuple("ValueError: shift count too large", MsgType.ERROR);

              const sh = val:int;

              if op == ">>=" then la >>= sh;
              else               la <<= sh;

            } else if kL == 4 {

              select op {
                when "&="  { la &=  val; }
                when "|="  { la |=  val; }
                when "^="  { la ^=  val; }
                otherwise  { handled = false; }
              }

            } else {

              // bool/bigint handled above; here we are int/uint
              select op {
                when ">>=" { la >>= val; }
                when "<<=" { la <<= val; }
                when "&="  { la &=  val; }
                when "|="  { la |=  val; }
                when "^="  { la ^=  val; }
                otherwise  { handled = false; }
              }

            }

          } else {
            handled = false;
          }

          if !handled then return MsgTuple.error(nie);

          // bigint post-mask (keep consistent with opeqvv)
          if kL == 4 && l.max_bits != -1 {
            const mask = (1: bigint << l.max_bits) - 1;
            la &= mask;
          }

          return MsgTuple.success();
        }

        when 1 { return new MsgTuple("TypeError", MsgType.ERROR); }
        when 2 { return new MsgTuple("TypeError", MsgType.ERROR); } // same return type, finer-grain later if desired
        otherwise { return MsgTuple.error(nie); }
      }

      return MsgTuple.success();
    }

}
