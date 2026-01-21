
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
              opStr = msgArgs['op'].toScalar(string);

        const op = operatorFromString(opStr);
        if op == Operator.Invalid {
          const errorMsg = "Unrecognized operator: %s".format(opStr);
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }


        omLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), 
             "cmd: %? op: %? left pdarray: %? right pdarray: %?".format(
                                          cmd,opStr,st.attrib(msgArgs['a'].val),
                                          st.attrib(msgArgs['b'].val)));

        // At this point it should handle almost every bigint case

        if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b) &&
                op != Operator.Div
        {
          if isBoolOp(op) {
            return st.insert(new shared SymEntry(doBigIntBinOpvvBoolReturn(l, r, op)));
          }
          // call bigint specific func which returns dist bigint array
          const (tmp, max_bits) = doBigIntBinOpvv(l, r, op);
          return st.insert(new shared SymEntry(tmp, max_bits));
        } else if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                    (isRealType(binop_dtype_a) || isRealType(binop_dtype_b)) &&
                    isBoolOp(op) {
          // call bigint specific func which returns distr bool array
          return st.insert(new shared SymEntry(doBigIntBinOpvvBoolReturnRealInput(l.a, r.a, op)));
        }
        else if (binop_dtype_a == bigint || binop_dtype_b == bigint) {
          return st.insert(new shared SymEntry(doBigIntBinOpvvRealReturn(l, r, op)));
        }

        if isBoolOp(op) {
          return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
        }

        if op == Operator.Div {
          if binop_dtype_a == real(32) && isSmallType(binop_dtype_b) {
            return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          if binop_dtype_b == real(32) && isSmallType(binop_dtype_a) {
            return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, real(64), op, pn, st);
        }

        type returnType = mySafeCast(binop_dtype_a, binop_dtype_b);

        if (!isRealOp(op)) && (returnType == real(32) || returnType == real(64)) {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if returnType == bool {
          if op == Operator.Add || op == Operator.Mul || (!isRealOp(op)) {
            return doBinOpvv(l, r, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
          }
          if op == Operator.Sub {
            const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
          if op == Operator.FloorDiv || op == Operator.Mod || op == Operator.Pow {
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
              opStr = msgArgs['op'].toScalar(string);

        const op = operatorFromString(opStr);
        if op == Operator.Invalid {
          const errorMsg = "Unrecognized operator: %s".format(opStr);
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }


        omLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
             "cmd: %? op: %? left pdarray: %? scalar: %?".format(
                                          cmd,opStr,st.attrib(msgArgs['a'].val), val));

        // This probably doesn't handle all normal bigint cases, but it handles a decent number.
        // This, at least, can be expanded when BinOp.chpl is cleaned up
        // It will be reasonably straightforward to clean up here.

        if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b)
        {
          if isBoolOp(op) {
            // call bigint specific func which returns distr bool array
            return st.insert(new shared SymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
          }
          // call bigint specific func which returns dist bigint array
          const (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
          return st.insert(new shared SymEntry(tmp, max_bits));
        } else if (binop_dtype_a == bigint || binop_dtype_b == bigint) {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if isBoolOp(op) {
          return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
        }

        if op == Operator.Div {
          if binop_dtype_a == real(32) && isSmallType(binop_dtype_b) {
            return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          if binop_dtype_b == real(32) && isSmallType(binop_dtype_a) {
            return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, real(64), op, pn, st);
        }

        type returnType = mySafeCast(binop_dtype_a, binop_dtype_b);

        if (!isRealOp(op)) && (returnType == real(32) || returnType == real(64)) {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if returnType == bool {
          if op == Operator.Add || op == Operator.Mul || (!isRealOp(op)) {
            return doBinOpvs(l, val, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
          }
          if op == Operator.Sub {
            const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
          if op == Operator.FloorDiv || op == Operator.Mod || op == Operator.Pow {
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
              opStr = msgArgs['op'].toScalar(string);

        const op = operatorFromString(opStr);
        if op == Operator.Invalid {
          const errorMsg = "Unrecognized operator: %s".format(opStr);
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }


        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                 "cmd: %? op = %? scalar dtype = %? scalar = %? pdarray = %?".format(
                                   cmd,opStr,type2str(binop_dtype_b),msgArgs['value'].val,st.attrib(msgArgs['a'].val)));

        // This probably doesn't handle all normal bigint cases, but it handles a decent number.
        // This, at least, can be expanded when BinOp.chpl is cleaned up
        // It will be reasonably straightforward to clean up here.

        if (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
                !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b)
        {
          if isBoolOp(op) {
            // call bigint specific func which returns distr bool array
            return st.insert(new shared SymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
          }
          // call bigint specific func which returns dist bigint array
          const (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
          return st.insert(new shared SymEntry(tmp, max_bits));
        } else if (binop_dtype_a == bigint || binop_dtype_b == bigint) {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if isBoolOp(op) {
          return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
        }

        if op == Operator.Div {
          if binop_dtype_a == real(32) && isSmallType(binop_dtype_b) {
            return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          if binop_dtype_b == real(32) && isSmallType(binop_dtype_a) {
            return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, real(32), op, pn, st);
          }
          return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, real(64), op, pn, st);
        }

        type returnType = mySafeCast(binop_dtype_a, binop_dtype_b);

        if (!isRealOp(op)) && (returnType == real(32) || returnType == real(64)) {
          const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
          omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        if returnType == bool {
          if op == Operator.Add || op == Operator.Mul || (!isRealOp(op)) {
            return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, bool, op, pn, st);
          }
          if op == Operator.Sub {
            const errorMsg = unrecognizedTypeError(pn, "("+type2str(binop_dtype_a)+","+type2str(binop_dtype_b)+")");
            omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }
          if op == Operator.FloorDiv || op == Operator.Mod || op == Operator.Pow {
            return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, uint(8), op, pn, st);
          }
        }

        return doBinOpsv(val, r, binop_dtype_a, binop_dtype_b, returnType, op, pn, st);
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
              opStr = msgArgs['op'].toScalar(string),
              nie = notImplementedError(pn,type2str(binop_dtype_a),opStr,type2str(binop_dtype_b));

        const op = opeqFromString(opStr);

        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                    "cmd: %s op: %s left pdarray: %s right pdarray: %s".format(cmd,opStr,
                    st.attrib(msgArgs['a'].val),st.attrib(msgArgs['b'].val)));

        if binop_dtype_a == int && binop_dtype_b == int  {
            select op {
                when OpEq.Pe { l.a += r.a; }
                when OpEq.Me { l.a -= r.a; }
                when OpEq.Te { l.a *= r.a; }
                when OpEq.Sre { l.a >>= r.a;}
                when OpEq.Sle { l.a <<= r.a;}
                when OpEq.Fde {
                    //l.a /= r.a;
                    ref la = l.a;
                    ref ra = r.a;
                    [(li,ri) in zip(la,ra)] li = if ri != 0 then li/ri else 0;
                }//floordiv
                when OpEq.Moe {
                    //l.a /= r.a;
                    ref la = l.a;
                    ref ra = r.a;
                    [(li,ri) in zip(la,ra)] li = if ri != 0 then li%ri else 0;
                }
                when OpEq.Ee {
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
                when OpEq.Pe {l.a += r.a:int;}
                when OpEq.Me {l.a -= r.a:int;}
                when OpEq.Te {l.a *= r.a:int;}
                when OpEq.Sre { l.a >>= r.a:int;}
                when OpEq.Sle { l.a <<= r.a:int;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == uint && binop_dtype_b == uint  {
            select op {
                when OpEq.Pe { l.a += r.a; }
                when OpEq.Me {
                    l.a -= r.a;
                }
                when OpEq.Te { l.a *= r.a; }
                when OpEq.Fde {
                    //l.a /= r.a;
                    ref la = l.a;
                    ref ra = r.a;
                    [(li,ri) in zip(la,ra)] li = if ri != 0 then li/ri else 0;
                }//floordiv
                when OpEq.Moe {
                    //l.a /= r.a;
                    ref la = l.a;
                    ref ra = r.a;
                    [(li,ri) in zip(la,ra)] li = if ri != 0 then li%ri else 0;
                }
                when OpEq.Ee {
                    l.a **= r.a;
                }
                when OpEq.Sre { l.a >>= r.a;}
                when OpEq.Sle { l.a <<= r.a;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == uint && binop_dtype_b == bool  {
            select op {
                when OpEq.Pe {l.a += r.a:uint;}
                when OpEq.Me {l.a -= r.a:uint;}
                when OpEq.Te {l.a *= r.a:uint;}
                when OpEq.Sre { l.a >>= r.a:uint;}
                when OpEq.Sle { l.a <<= r.a:uint;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == int  {
            select op {
                when OpEq.Pe {l.a += r.a;}
                when OpEq.Me {l.a -= r.a;}
                when OpEq.Te {l.a *= r.a;}
                when OpEq.De {l.a /= r.a:real;} //truediv
                when OpEq.Fde { //floordiv
                    ref la = l.a;
                    ref ra = r.a;
                    la = floorDivision(la, ra, real);
                }
                when OpEq.Ee { l.a **= r.a; }
                when OpEq.Moe {
                    ref la = l.a;
                    ref ra = r.a;
                    [(li,ri) in zip(la,ra)] li = modHelper(li, ri);
                }
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == uint  {
            select op {
                when OpEq.Pe {l.a += r.a;}
                when OpEq.Me {l.a -= r.a;}
                when OpEq.Te {l.a *= r.a;}
                when OpEq.De {l.a /= r.a:real;} //truediv
                when OpEq.Fde { //floordiv
                    ref la = l.a;
                    ref ra = r.a;
                    la = floorDivision(la, ra, real);
                }
                when OpEq.Ee { l.a **= r.a; }
                when OpEq.Moe {
                    ref la = l.a;
                    ref ra = r.a;
                    [(li,ri) in zip(la,ra)] li = modHelper(li, ri);
                }
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == real  {
            select op {
                when OpEq.Pe {l.a += r.a;}
                when OpEq.Me {l.a -= r.a;}
                when OpEq.Te {l.a *= r.a;}
                when OpEq.De {l.a /= r.a;}//truediv
                when OpEq.Fde { //floordiv
                    ref la = l.a;
                    ref ra = r.a;
                    la = floorDivision(la, ra, real);
                }
                when OpEq.Ee { l.a **= r.a; }
                when OpEq.Moe {
                    ref la = l.a;
                    ref ra = r.a;
                    [(li,ri) in zip(la,ra)] li = modHelper(li, ri);
                }
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == bool  {
            select op {
                when OpEq.Pe {l.a += r.a:real;}
                when OpEq.Me {l.a -= r.a:real;}
                when OpEq.Te {l.a *= r.a:real;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bool && binop_dtype_b == bool  {
            select op {
                when OpEq.Oe {l.a |= r.a;}
                when OpEq.Ae {l.a &= r.a;}
                when OpEq.Xe {l.a ^= r.a;}
                when OpEq.Pe {l.a |= r.a;}
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
              when OpEq.Pe {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li += ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Me {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li -= ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Te {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li *= ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Fde {
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
              when OpEq.Moe {
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
              when OpEq.Ee {
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
              when OpEq.Pe {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li += ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Me {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li -= ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Te {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li *= ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Fde {
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
              when OpEq.Moe {
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
              when OpEq.Ee {
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
              when OpEq.Pe {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li += ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Me {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li -= ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Te {
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
              when OpEq.Pe {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li += ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Me {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li -= ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Te {
                forall (li, ri) in zip(la, ra) with (var local_max_size = max_size) {
                  li *= ri;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Fde {
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
              when OpEq.Moe {
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
              when OpEq.Ee {
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

        // b is always the same type as a
        type binop_dtype_b = binop_dtype_a;

        var l = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd);
        const val = msgArgs['value'].toScalar(binop_dtype_b),
              opStr = msgArgs['op'].toScalar(string),
              nie = notImplementedError(pn,type2str(binop_dtype_a),opStr,type2str(binop_dtype_b));

        const op = opeqFromString(opStr);

        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "op: %? pdarray: %? scalar: %?".format(opStr,st.attrib(msgArgs['a'].val),val));

        if binop_dtype_a == int && binop_dtype_b == int  {
            select op {
                when OpEq.Pe { l.a += val; }
                when OpEq.Me { l.a -= val; }
                when OpEq.Te { l.a *= val; }
                when OpEq.Sre { l.a >>= val; }
                when OpEq.Sle { l.a <<= val; }
                when OpEq.Fde {
                    if val != 0 {l.a /= val;} else {l.a = 0;}
                }//floordiv
                when OpEq.Moe {
                    if val != 0 {l.a %= val;} else {l.a = 0;}
                }
                when OpEq.Ee {
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
        else if binop_dtype_a == uint && binop_dtype_b == uint  {
            select op {
                when OpEq.Pe { l.a += val; }
                when OpEq.Me {
                    l.a -= val;
                }
                when OpEq.Te { l.a *= val; }
                when OpEq.Fde {
                    if val != 0 {l.a /= val;} else {l.a = 0;}
                }//floordiv
                when OpEq.Moe {
                    if val != 0 {l.a %= val;} else {l.a = 0;}
                }
                when OpEq.Ee {
                    l.a **= val;
                }
                when OpEq.Sre { l.a >>= val; }
                when OpEq.Sle { l.a <<= val; }
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == bool && binop_dtype_b == bool  {
            select op {
                when OpEq.Pe {l.a |= val;}
                otherwise do return MsgTuple.error(nie);
            }
        }
        else if binop_dtype_a == real && binop_dtype_b == real  {
            select op {
                when OpEq.Pe {l.a += val;}
                when OpEq.Me {l.a -= val;}
                when OpEq.Te {l.a *= val;}
                when OpEq.De {l.a /= val;}//truediv
                when OpEq.Fde { //floordiv
                    ref la = l.a;
                    la = floorDivision(la, val, real);
                }
                when OpEq.Ee { l.a **= val; }
                when OpEq.Moe {
                    ref la = l.a;
                    [li in la] li = modHelper(li, val);
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
              when OpEq.Pe {
                forall li in la with (var local_val = val, var local_max_size = max_size) {
                  li += local_val;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Me {
                forall li in la with (var local_val = val, var local_max_size = max_size) {
                  li -= local_val;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Te {
                forall li in la with (var local_val = val, var local_max_size = max_size) {
                  li *= local_val;
                  if has_max_bits {
                    li &= local_max_size;
                  }
                }
              }
              when OpEq.Fde {
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
              when OpEq.Moe {
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
              when OpEq.Ee {
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
