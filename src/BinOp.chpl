
module BinOp
{
  use ServerConfig;

  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Logging;
  use Message;
  use BitOps;
  use BigInteger;
  use Set;
  use Math;


  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const omLogger = new Logger(logLevel, logChannel);

  proc splitType(type dtype) param : int {
    // 0 -> bool, 1 -> uint, 2 -> int, 3 -> real

    if dtype == bool then return 0;
    else if dtype == uint(8) then return 1;
    else if dtype == uint(16) then return 1;
    else if dtype == uint(32) then return 1;
    else if dtype == uint(64) then return 1;
    else if dtype == int(8) then return 2;
    else if dtype == int(16) then return 2;
    else if dtype == int(32) then return 2;
    else if dtype == int(64) then return 2;
    else if dtype == real(32) then return 3;
    else if dtype == real(64) then return 3;
    else if dtype == bigint then return 4;
    else return 0;

  }

  proc mySafeCast(type dtype1, type dtype2) type {
    param typeKind1 = splitType(dtype1);
    param bitSize1 = if dtype1 == bool then 8 else numBits(dtype1);
    param typeKind2 = splitType(dtype2);
    param bitSize2 = if dtype2 == bool then 8 else numBits(dtype2);

    if typeKind1 == 2 && typeKind2 == 1 && bitSize1 <= bitSize2 {
      select bitSize2 {
        when 64 { return real(64); }
        when 32 { return int(64); }
        when 16 { return int(32); }
        when 8 { return int(16); }
      }
    }

    if typeKind2 == 2 && typeKind1 == 1 && bitSize2 <= bitSize1 {
      select bitSize1 {
        when 64 { return real(64); }
        when 32 { return int(64); }
        when 16 { return int(32); }
        when 8 { return int(16); }
      }
    }

    if dtype1 == real(32) && (dtype2 == int(32) || dtype2 == uint(32)) {
      return real(64);
    }

    if dtype2 == real(32) && (dtype1 == int(32) || dtype1 == uint(32)) {
      return real(64);
    }

    if typeKind1 == 3 || typeKind2 == 3 {
      select max(bitSize1, bitSize2) {
        when 64 { return real(64); }
        when 32 { return real(32); }
      }
    }

    if typeKind1 == 2 || typeKind2 == 2 {
      select max(bitSize1, bitSize2) {
        when 64 { return int(64); }
        when 32 { return int(32); }
        when 16 { return int(16); }
        when 8 { return int(8); }
      }
    }

    if typeKind1 == 1 || typeKind2 == 1 {
      select max(bitSize1, bitSize2) {
        when 64 { return uint(64); }
        when 32 { return uint(32); }
        when 16 { return uint(16); }
        when 8 { return uint(8); }
      }
    }

    return bool;

  }

  /*
    Helper function to ensure that floor division cases are handled in accordance with numpy
  */

  inline proc floorDivisionHelper(numerator: real(64), denom: real(64)): real(64) {
    if (numerator == 0 && denom == 0) || (isInf(numerator) && (denom != 0 || isInf(denom))){
      return nan;
    }
    else if (numerator > 0 && denom == -inf) || (numerator < 0 && denom == inf){
      return -1:real;
    }
    else if denom == 0 || isInf(denom) {
      return numerator / denom;
    }

    const q  = numerator / denom;
    const fq = floor(q);
    if q != fq then return fq;

    // fma does (-q * denom) + numerator and stores that in r
    // From https://en.cppreference.com/w/c/numeric/math/fma.html
    // Computes (x * y) + z as if to infinite precision and rounded only once to fit the result type.
    const r = fma(-q, denom, numerator);
    if r == 0.0 then return q;

    // This next part may seem a little weird, but if r = -q * denom + numerator
    // Then numerator / denom = q + r / denom (This should remind you of the division algorithm)
    // r should be small, it's basically the floating point error when we calculated q.
    // At this point, q is an integer because we already returned if q != fq.
    // If r / denom is negative then we need to actually round down. If r / denom is positive, then q
    // is the correct floor division.
    // Rather than actually dividing the values, we only need to check if the signs differ or are the
    // same. If they differ, then r / denom is negative, otherwise it is positive.
    const qIsBelow = ((r > 0.0) != (denom > 0.0));
    return if qIsBelow then (q - 1.0) else q;
  }

  /*
    Helper function to ensure that mod cases are handled in accordance with numpy
  */
  inline proc modHelper(dividend: ?t, divisor: ?t2): real {
    extern proc fmod(x: real, y: real): real;

    var res = fmod(dividend, divisor);
    // to convert fmod (truncated) results into mod (floored) results
    // when the dividend and divsor have opposite signs,
    // we add the divsor into the result
    // except for when res == 0 (divsor even divides dividend)
    // see https://en.wikipedia.org/wiki/Modulo#math_1 for more information
    if res != 0 && (((dividend < 0) && (divisor > 0)) || ((dividend > 0) && (divisor < 0))) {
      // we do + either way because we want to shift up for positive divisors and shift down for negative
      res += divisor;
    }
    return res;
  }

  //
  // TODO: these checks sets are used only to check if a string matches
  // i.e. realOps.contains(op) or boolOps.contains(op)
  // however, a faster way to do this is to write a function that checks
  // the ascii bytes
  // for example, the boolOps can be written as:
  //
  // return (op.numBytes == 1 && (op.byte[0] == 60 ||
  //                             op.byte[0] == 62)) ||
  //        (op.numBytes == 2 && ((op.byte[0] == 60 && op.byte[1] == 61) ||
  //                              (op.byte[0] == 62 && op.byte[1] == 61) ||
  //                              (op.byte[0] == 61 && op.byte[1] == 61) ||
  //                              (op.byte[0] == 33 && op.byte[1] == 61)));
  //
  // it should also be possible to use the `toByte` method at compile-time to
  // improve the readability, although this has not been tested to see how it impacts
  // compile-time and runtime performance:
  //
  // return (op.numBytes == 1 && (op.byte[0] == "<".toByte() ||
  //                             op.byte[0] == ">".toByte())) ||
  //        (op.numBytes == 2 && ((op.byte[0] == "<".toByte() && op.byte[1] == "=".toByte()) ||
  //                              (op.byte[0] == ">".toByte() && op.byte[1] == "=".toByte()) ||
  //                              (op.byte[0] == "=".toByte() && op.byte[1] == "=".toByte()) ||
  //                              (op.byte[0] == "!".toByte() && op.byte[1] == "=".toByte())));
  //

  const realOps = new set(string, ["+", "-", "*", "//", "%", "**"]);

  // All operations that involve one of these operations result in a `bool`
  // symbol table entry.
  const boolOps = new set(string, ["<", "<=", ">", ">=", "==", "!="]);

  const smallOps = new set(string, ["<<", ">>", "**"]);

  proc doBoolBoolBitOp(
    op: string, ref e: [] bool, l: [] bool, r /*: [] bool OR bool*/
  ): bool {
    var handled = false;
    if op == "|" || op == "+" {
      e = l | r;
      handled = true;
    } else if op == "&" || op == "*" {
      e = l & r;
      handled = true;
    } else if op == "^" {
      e = l ^ r;
      handled = true;
    }
    return handled;
  }

  /*
  Generic function to execute a binary operation on pdarray entries 
  in the symbol table

  :arg l: symbol table entry of the LHS operand

  :arg r: symbol table entry of the RHS operand

  :arg e: symbol table entry to store result of operation

  :arg op: string representation of binary operation to execute
  :type op: string

  :arg pn: routine name of callsite function
  :type pn: string

  :arg st: SymTab to act on
  :type st: borrowed SymTab 

  :returns: (MsgTuple) 
  :throws: `UndefinedSymbolError(name)`
  */
  proc _doBinOpVLeftVec(l, r, type lType, type rType, type etype, op: string, pn, st): MsgTuple throws {
    var e = makeDistArray((...l.tupShape), etype);

    const nie = notImplementedError(pn,l.dtype,op,r.dtype);

    type castType = mySafeCast(lType, rType);

    // The compiler complains that maybe etype is bool if it gets down below this
    // without returning, so we have to kind of chunk this next piece off.

    // For similar reasons, everything else is kinda split off into its own thing.
    // The compiler has no common sense about things (and that's not really its fault)

    if etype == bool {

      if boolOps.contains(op) {

        select op {

          when "<" { e = (l.a: castType) < (r.a: castType); }
          when "<=" { e = (l.a: castType) <= (r.a: castType); }
          when ">" { e = (l.a: castType) > (r.a: castType); }
          when ">=" { e = (l.a: castType) >= (r.a: castType); }
          when "==" { e = (l.a: castType) == (r.a: castType); }
          when "!=" { e = (l.a: castType) != (r.a: castType); }
          otherwise do return MsgTuple.error(nie); // Shouldn't happen

        }
        return st.insert(new shared SymEntry(e));
      }

      if lType == bool && rType == bool {
        if !doBoolBoolBitOp(op, e, l.a, r.a) {
          return MsgTuple.error(nie);
        }
        return st.insert(new shared SymEntry(e));
      }

      return MsgTuple.error(nie);

    }

    else if lType == bool && rType == bool && etype == uint(8) { // Both bools is kinda weird
      select op {
        when "%" { e = (0: uint(8)); } // numpy has these as int(8), but Arkouda doesn't really support that type.
        when "//" { e = (l.a & r.a): uint(8); }
        when "**" { e = (!l.a & r.a): uint(8); }
        when "<<" { e = (l.a: uint(8)) << (r.a: uint(8)); }
        when ">>" { e = (l.a: uint(8)) >> (r.a: uint(8)); }
        otherwise do return MsgTuple.error(nie);
        // >>> and <<< could probably be implemented as int(8) or uint(8) things
      }
      return st.insert(new shared SymEntry(e));
    }

    else if etype == real(32) || etype == real(64) {

      select op {
        when "*" { e = (l.a: etype * r.a: etype): etype; }
        when "+" { e = (l.a: etype + r.a: etype): etype; }
        when "-" { e = (l.a: etype - r.a: etype): etype; }
        when "/" { e = (l.a: etype / r.a: etype): etype; }
        when "%" {
          ref ea = e;
          ref la = l.a;
          ref ra = r.a;
          [(ei,li,ri) in zip(ea,la,ra)] ei = modHelper(li: etype, ri: etype): etype;
        }
        when "//" {
          ref ea = e;
          ref la = l.a;
          ref ra = r.a;
          [(ei,li,ri) in zip(ea,la,ra)] ei = floorDivisionHelper(li: etype, ri: etype): etype;
        }
        when "**" {
          e = ((l.a: etype) ** (r.a: etype)): etype;
        }
        otherwise do return MsgTuple.error(nie);
      }
      return st.insert(new shared SymEntry(e));

    }

    else {

      select op {
        when "|" { e = (l.a | r.a): etype; }
        when "&" { e = (l.a & r.a): etype; }
        when "*" { e = (l.a * r.a): etype; }
        when "^" { e = (l.a ^ r.a): etype; }
        when "+" { e = (l.a + r.a): etype; }
        when "-" { e = (l.a - r.a): etype; }
        when "/" { e = (l.a: etype) / (r.a: etype); }
        when "%" {
          ref ea = e;
          ref la = l.a;
          ref ra = r.a;
          [(ei,li,ri) in zip(ea,la,ra)] ei = if ri != 0 then li%ri else 0;
        }
        when "//" {
          ref ea = e;
          ref la = l.a;
          ref ra = r.a;
          [(ei,li,ri) in zip(ea,la,ra)] ei = if ri != 0 then (li/ri): etype else 0: etype;
        }
        when "**" {
          if || reduce (r.a<0)
            then return MsgTuple.error("Attempt to exponentiate base of type Int or UInt to negative exponent");
          e = (l.a: etype) ** (r.a: etype);
        }
        when "<<" {
          ref ea = e;
          ref la = l.a;
          ref ra = r.a;
          [(ei,li,ri) in zip(ea,la,ra)] if (0 <= ri && ri < numBits(etype)) then ei = ((li: etype) << (ri: etype)): etype;
        }
        when ">>" {
          ref ea = e;
          ref la = l.a;
          ref ra = r.a;
          [(ei,li,ri) in zip(ea,la,ra)] if (0 <= ri && ri < numBits(etype)) then ei = ((li: etype) >> (ri: etype)): etype;
        }
        when "<<<" { e = rotl(l.a: etype, r.a: etype); }
        when ">>>" { e = rotr(l.a: etype, r.a: etype); }
        otherwise do return MsgTuple.error(nie);
      }
      return st.insert(new shared SymEntry(e));
    }

    return MsgTuple.error(nie);

  }

  // Public entrypoints (vv/vs/sv) delegate to vector-left kernels.
  proc doBinOpvv(l, r, type lType, type rType, type etype, op: string, pn, st): MsgTuple throws {
    return _doBinOpVLeftVec(l, r, lType, rType, etype, op, pn, st);
  }
  
  proc _doBinOpVLeftScalar(vec, val, type lType, type rType, type etype,
                           op: string, pn, st,
                           param swapOperands: bool = false): MsgTuple throws {
    var e = makeDistArray((...vec.tupShape), etype);

    const nie = if !swapOperands
                  then notImplementedError(pn,"%s %s %s".format(type2str(vec.a.eltType), op, type2str(val.type)))
                  else notImplementedError(pn,"%s %s %s".format(type2str(val.type), op, type2str(vec.a.eltType)));

    type castType = mySafeCast(lType, rType);

    if etype == bool {

      if boolOps.contains(op) {
        select op {
          when "<"  { e = if !swapOperands then (vec.a: castType) <  (val: castType)
                                          else (val: castType)    <  (vec.a: castType); }
          when "<=" { e = if !swapOperands then (vec.a: castType) <= (val: castType)
                                          else (val: castType)   <= (vec.a: castType); }
          when ">"  { e = if !swapOperands then (vec.a: castType) >  (val: castType)
                                          else (val: castType)   >  (vec.a: castType); }
          when ">=" { e = if !swapOperands then (vec.a: castType) >= (val: castType)
                                          else (val: castType)   >= (vec.a: castType); }
          when "==" { e = if !swapOperands then (vec.a: castType) == (val: castType)
                                          else (val: castType)   == (vec.a: castType); }
          when "!=" { e = if !swapOperands then (vec.a: castType) != (val: castType)
                                          else (val: castType)   != (vec.a: castType); }
          otherwise do return MsgTuple.error(nie);
        }
        return st.insert(new shared SymEntry(e));
      }

      if lType == bool && rType == bool {
        // Bit-ops on bools are commutative for the supported operators.
        if !doBoolBoolBitOp(op, e, vec.a, val) {
          return MsgTuple.error(nie);
        }
        return st.insert(new shared SymEntry(e));
      }

      return MsgTuple.error(nie);
    }

    else if lType == bool && rType == bool && etype == uint(8) {
      // Both bools is kinda weird.
      select op {
        when "%" { e = (0: uint(8)); }
        when "//" { e = if !swapOperands then (vec.a & val): uint(8) else (val & vec.a): uint(8); }
        when "**" { e = if !swapOperands then (!vec.a & val): uint(8) else (!val & vec.a): uint(8); }
        when "<<" { e = if !swapOperands then (vec.a: uint(8)) << (val: uint(8))
                                     else (val: uint(8)) << (vec.a: uint(8)); }
        when ">>" { e = if !swapOperands then (vec.a: uint(8)) >> (val: uint(8))
                                     else (val: uint(8)) >> (vec.a: uint(8)); }
        otherwise do return MsgTuple.error(nie);
      }
      return st.insert(new shared SymEntry(e));
    }

    else if etype == real(32) || etype == real(64) {

      select op {
        when "*" { e = (vec.a: etype * val: etype): etype; }  // commutative
        when "+" { e = (vec.a: etype + val: etype): etype; }  // commutative
        when "-" { e = if !swapOperands then (vec.a: etype - val: etype): etype
                                      else (val: etype - vec.a: etype): etype; }
        when "/" { e = if !swapOperands then (vec.a: etype / val: etype): etype
                                      else (val: etype / vec.a: etype): etype; }
        when "%" {
          ref ea = e;
          ref va = vec.a;
          if !swapOperands {
            [(ei, li) in zip(ea, va)] ei = modHelper(li: etype, val: etype): etype;
          } else {
            [(ei, ri) in zip(ea, va)] ei = modHelper(val: etype, ri: etype): etype;
          }
        }
        when "//" {
          ref ea = e;
          ref va = vec.a;
          if !swapOperands {
            [(ei, li) in zip(ea, va)] ei = floorDivisionHelper(li: etype, val: etype): etype;
          } else {
            [(ei, ri) in zip(ea, va)] ei = floorDivisionHelper(val: etype, ri: etype): etype;
          }
        }
        when "**" {
          e = if !swapOperands then ((vec.a: etype) ** (val: etype)): etype
                               else ((val: etype) ** (vec.a: etype)): etype;
        }
        otherwise do return MsgTuple.error(nie);
      }
      return st.insert(new shared SymEntry(e));
    }

    else {
      select op {
        when "|" { e = (vec.a | val): etype; }
        when "&" { e = (vec.a & val): etype; }
        when "*" { e = (vec.a * val): etype; }
        when "^" { e = (vec.a ^ val): etype; }
        when "+" { e = (vec.a + val): etype; }
        when "-" { e = if !swapOperands then (vec.a - val): etype else (val - vec.a): etype; }
        when "/" { e = if !swapOperands then (vec.a: etype) / (val: etype)
                                    else (val: etype) / (vec.a: etype); }
        when "%" {
          ref ea = e;
          ref va = vec.a;
          if !swapOperands {
            [(ei, li) in zip(ea, va)] ei = if val != 0 then li%val else 0;
          } else {
            [(ei, ri) in zip(ea, va)] ei = if ri != 0 then val%ri else 0;
          }
        }
        when "//" {
          ref ea = e;
          ref va = vec.a;
          if !swapOperands {
            [(ei, li) in zip(ea, va)] ei = if val != 0 then (li/val): etype else 0: etype;
          } else {
            [(ei, ri) in zip(ea, va)] ei = if ri != 0 then (val/ri): etype else 0: etype;
          }
        }
        when "**" {
          if !swapOperands {
            if val < 0 then return MsgTuple.error("Attempt to exponentiate base of type Int or UInt to negative exponent");
            e = (vec.a: etype) ** (val: etype);
          } else {
            if || reduce (vec.a < 0) then return MsgTuple.error("Attempt to exponentiate base of type Int or UInt to negative exponent");
            e = (val: etype) ** (vec.a: etype);
          }
        }
        when "<<" {
          ref ea = e;
          ref va = vec.a;
          if !swapOperands {
            [(ei, li) in zip(ea, va)] if (0 <= val && val < numBits(etype)) then ei = ((li: etype) << (val: etype)): etype;
          } else {
            [(ei, ri) in zip(ea, va)] if (0 <= ri && ri < numBits(etype)) then ei = ((val: etype) << (ri: etype)): etype;
          }
        }
        when ">>" {
          ref ea = e;
          ref va = vec.a;
          if !swapOperands {
            [(ei, li) in zip(ea, va)] if (0 <= val && val < numBits(etype)) then ei = ((li: etype) >> (val: etype)): etype;
          } else {
            [(ei, ri) in zip(ea, va)] if (0 <= ri && ri < numBits(etype)) then ei = ((val: etype) >> (ri: etype)): etype;
          }
        }
        when "<<<" {
          e = if !swapOperands then rotl(vec.a: etype, val: etype)
                               else rotl(val: etype, vec.a: etype);
        }
        when ">>>" {
          e = if !swapOperands then rotr(vec.a: etype, val: etype)
                               else rotr(val: etype, vec.a: etype);
        }
        otherwise do return MsgTuple.error(nie);
      }
      return st.insert(new shared SymEntry(e));
    }
  }

  proc doBinOpvs(l, val, type lType, type rType, type etype, op: string, pn, st): MsgTuple throws {
    return _doBinOpVLeftScalar(l, val, lType, rType, etype, op, pn, st, false);
  }

  proc doBinOpsv(val, r, type lType, type rType, type etype, op: string, pn, st): MsgTuple throws {
    // Guarantee a vector on the left; scalar-left is handled by swapping operands in the scalar kernel.
    return _doBinOpVLeftScalar(r, val, lType, rType, etype, op, pn, st, true);
  }

  proc doBigIntBinOpvv(l, r, op: string) throws {
    var max_bits = max(l.max_bits, r.max_bits);
    var max_size = 1:bigint;
    var has_max_bits = max_bits != -1;
    if has_max_bits {
      max_size <<= max_bits;
      max_size -= 1;
    }
    ref la = l.a;
    ref ra = r.a;
    var tmp = if l.etype == bigint then la else la:bigint;

    // had to create bigint specific BinOp procs which return
    // the distributed array because we need it at SymEntry creation time
    if l.etype == bigint && r.etype != bigint && smallOps.contains(op) {
      // ops that only work with a left hand side of bigint and right hand side non-bigint
      // Just bitshifts and exponentiation without local_max_size
      select op {
        when "<<" {
          forall (t, ri) in zip(tmp, ra) do
            t = if has_max_bits && ri >= max_bits then 0: bigint else t << ri;
        }
        when ">>" {
          forall (t, ri) in zip(tmp, ra) do
            t = if has_max_bits && ri >= max_bits then 0: bigint else t >> ri;
        }
        when "**" {
          if || reduce (ra<0) { // In the future, this should actually lead into real number territory
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
              powMod(t, t, ri, local_max_size + 1);
            }
          }
          else {
            forall (t, ri) in zip(tmp, ra) do t **= ri:uint;
          }
        }
      }
    }
    else {
      select op {
        when "&" { forall (t, ri) in zip(tmp, ra) do t &= ri: bigint; }
        when "|" { forall (t, ri) in zip(tmp, ra) do t |= ri: bigint; }
        when "^" { forall (t, ri) in zip(tmp, ra) do t ^= ri: bigint; }
        when "+" { forall (t, ri) in zip(tmp, ra) do t += ri: bigint; }
        when "-" { forall (t, ri) in zip(tmp, ra) do t -= ri: bigint; }
        when "*" { forall (t, ri) in zip(tmp, ra) do t *= ri: bigint; }
        when "//" {
          forall (t, ri) in zip(tmp, ra) {
            const denom: bigint = ri: bigint;   // <- cast bool/int/uint/etc to bigint
            if denom != 0 {
              var q: bigint;
              // floor-style integer division, like Python's //
              div(q, t, denom, roundingMode.down);
              t = q;
            } else {
              // whatever semantics you want for division by zero:
              t = 0: bigint;
            }
          }
        }
        when "%" {
          forall (t, ri) in zip(tmp, ra) {
            if ri != 0 {
              mod(t, t, ri: bigint);
            } else {
              t = 0: bigint;
            }
          }
        }
        when "**" {
          if || reduce (ra<0) {
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
              powMod(t, t, ri: bigint, local_max_size + 1);
            }
          }
          else {
            throw new Error("Attempt to exponentiate base of type BigInt to BigInt without max_bits");
          }
        }
        when "<<<" {
          if !has_max_bits { // This could be expanded if l.etype is not bigint but r.etype is
            throw new Error("Must set max_bits to rotl");
          }
          var botBits = la: bigint;
          if r.etype == bigint {
            var modded_shift = 0: bigint;
            forall (t, ri, bot_bits) in zip(tmp, ra, botBits) with (var loc_modded_shift = modded_shift) {
              mod(loc_modded_shift, ri: bigint, max_bits); // If max_bits is an int, loc_modded_shift is no bigger than int size
              t <<= loc_modded_shift: int;
              var shift_amt = max_bits - loc_modded_shift: int;
              bot_bits >>= shift_amt;
              t += bot_bits;
            }
          } else {
            forall (t, ri, bot_bits) in zip(tmp, ra, botBits) {
              var modded_shift = if r.etype == int then ri % max_bits else ri % max_bits:uint;
              t <<= modded_shift;
              var shift_amt = if r.etype == int then max_bits - modded_shift else max_bits:uint - modded_shift;
              bot_bits >>= shift_amt;
              t += bot_bits;
            }
          }
        }
        when ">>>" {
          if !has_max_bits {
            throw new Error("Must set max_bits to rotr");
          }
          var topBits = la: bigint;
          if r.etype == bigint {
            var modded_shift = 0:bigint;
            forall (t, ri, tB) in zip(tmp, ra, topBits) with (var loc_modded_shift = modded_shift) {
              mod(loc_modded_shift, ri: bigint, max_bits);
              t >>= loc_modded_shift: int;
              var shift_amt = max_bits - loc_modded_shift: int;
              tB <<= shift_amt;
              t += tB;
            }
          } else {
            forall (t, ri, tB) in zip(tmp, ra, topBits) {
              var modded_shift = if r.etype == int then ri % max_bits else ri % max_bits:uint;
              t >>= modded_shift;
              var shift_amt = if r.etype == int then max_bits - modded_shift else max_bits:uint - modded_shift;
              tB <<= shift_amt;
              t += tB;
            }
          }
        }
        otherwise do throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ r.etype:string);
      }
    }

    if has_max_bits then forall t in tmp with (const local_max_size = max_size) do t &= local_max_size;

    return (tmp, max_bits);
  }

  proc doBigIntBinOpvvBoolReturn(l, r, op: string) throws {
    select op {
      when "<" { return l.a < r.a; }
      when ">" { return l.a > r.a; }
      when "<=" { return l.a <= r.a; }
      when ">=" { return l.a >= r.a; }
      when "==" { return l.a == r.a; }
      when "!=" { return l.a != r.a; }
      otherwise {
        // we should never reach this since we only enter this proc
        // if boolOps.contains(op)
        throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ r.etype:string);
      }
    }
  }

  // Compare a bigint to a real(64).
  // Returns -1 if b < r, 0 if b == r, 1 if b > r.
  // (Assumes r is not NaN; NaN handled by wrappers below.)
  inline proc cmpBigReal(b: bigint, r: real(64)): int {
    // infinities
    if !isFinite(r) {
      if r > 0.0 {
        return -1; // b < +inf
      } else {
        return 1;  // b > -inf
      }
    }

    // Check if r is integer-valued in real64
    const ri = trunc(r);     // ri is real(64), trunc toward zero
    if r == ri {
      var zi: bigint;        // exact integer to compare against
      zi.set(ri);            // safe because r == ri (no fractional part)
      if b < zi then return -1;
      else if b > zi then return 1;
      else return 0;
    }

    // Non-integer r: use the integer part to bound r
    var tz: bigint;          // tz = trunc(r) as bigint
    tz.set(ri);

    // non-integer cases
    if r > 0.0 {
      // r in (tz, tz+1)
      if b <= tz {
        return -1;
      } else {
        return 1;
      }
    } else {
      // r in (tz-1, tz)
      if b >= tz {
        return 1;
      } else {
        return -1;
      }
    }
  }

  // Convenience predicates derived from cmpBigReal + NaN rules
  inline proc ltBigReal(b: bigint, r: real(64)): bool {
    return !isNan(r) && cmpBigReal(b, r) < 0;
  }
  inline proc gtBigReal(b: bigint, r: real(64)): bool {
    return !isNan(r) && cmpBigReal(b, r) > 0;
  }
  inline proc leBigReal(b: bigint, r: real(64)): bool {
    return !isNan(r) && cmpBigReal(b, r) <= 0;
  }
  inline proc geBigReal(b: bigint, r: real(64)): bool {
    return !isNan(r) && cmpBigReal(b, r) >= 0;
  }
  inline proc eqBigReal(b: bigint, r: real(64)): bool {
    return !isNan(r) && cmpBigReal(b, r) == 0;
  }
  inline proc neBigReal(b: bigint, r: real(64)): bool {
    return isNan(r) || cmpBigReal(b, r) != 0;
  }

  proc doBigIntBinOpvvBoolReturnRealInput(const ref la: [?d] ?t1, const ref ra: [d] ?t2, op: string) throws
    where ( (t1 == bigint && t2 == real(64)) ||
            (t1 == real(64) && t2 == bigint) )
  {

    var e = makeDistArray(d, bool);
    ref ea = e;
    select op {
      when "<"  {
        if t1 == bigint {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = ltBigReal(li, ri);
        } else { // t2 == bigint
          forall (ei, li, ri) in zip(ea, la, ra) do ei = gtBigReal(ri, li); // li<ri  <=>  ri>li
        }
      }
      when ">"  {
        if t1 == bigint {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = gtBigReal(li, ri);
        } else {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = ltBigReal(ri, li);
        }
      }
      when "<=" {
        if t1 == bigint {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = leBigReal(li, ri);
        } else {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = geBigReal(ri, li);
        }
      }
      when ">=" {
        if t1 == bigint {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = geBigReal(li, ri);
        } else {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = leBigReal(ri, li);
        }
      }
      when "==" {
        if t1 == bigint {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = eqBigReal(li, ri);
        } else {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = eqBigReal(ri, li);
        }
      }
      when "!=" {
        if t1 == bigint {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = neBigReal(li, ri);
        } else {
          forall (ei, li, ri) in zip(ea, la, ra) do ei = neBigReal(ri, li);
        }
      }
      otherwise do
        throw new Error("Unsupported operation: " + t1:string + " " + op + " " + t2:string);
    }

    return e;
  }

  proc doBigIntBinOpvvBoolReturnRealInput(const ref la: [?d] ?t1, const ref ra: [d] ?t2, op: string) throws
    where ( (t1 != bigint || t2 != real(64)) &&
            (t1 != real(64) || t2 != bigint) )
  {
    throw new Error("Unsupported operation: " + t1:string +" "+ op +" "+ t2:string);
  }

  proc doBigIntBinOpvvRealReturn(l, r, op: string) throws {
    select op {
      when "+" { return l.a: real + r.a: real; }
      when "-" { return l.a: real - r.a: real; }
      when "*" { return l.a: real * r.a: real; }
      when "/" { return l.a: real / r.a: real; }
      when "**" { return l.a: real ** r.a: real; }
      when "%" {
        var e = makeDistArray((...l.tupShape), real);
        ref ea = e;
        ref la = l.a;
        ref ra = r.a;
        [(ei,li,ri) in zip(ea,la,ra)] ei = modHelper(li: real, ri: real): real;
        return e;
      }
      when "//" {
        var e = makeDistArray((...l.tupShape), real);
        ref ea = e;
        ref la = l.a;
        ref ra = r.a;
        [(ei,li,ri) in zip(ea,la,ra)] ei = floorDivisionHelper(li: real, ri: real): real;
        return e;
      }
      otherwise {
        throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ r.etype:string);
      }
    }
  }

  
  
  proc _doBigIntBinOpVLeftScalar(vec, val, op: string, param swapOperands: bool = false) throws {
    // swapOperands == false:   vec OP val
    // swapOperands == true:    val OP vec   (but we still "own" vec's shape)

    // Match doBigIntBinOpvv's assumptions: max_bits is driven by the array entries.
    var max_bits = vec.max_bits;
    var max_size = 1:bigint;
    var has_max_bits = max_bits != -1;
    if has_max_bits {
      max_size <<= max_bits;
      max_size -= 1;
    }

    ref veca = vec.a;
    var tmp = if vec.etype == bigint then veca else veca:bigint;

    // scalar value as bigint (for bitwise / add / mul, etc.)
    const ri: bigint = val:bigint;

    // had to create bigint specific BinOp procs which return
    // the distributed array because we need it at SymEntry creation time
    if !swapOperands && vec.etype == bigint && val.type != bigint && smallOps.contains(op) {
      // ops that only work with a left hand side of bigint and right hand side non-bigint
      // Just bitshifts and exponentiation without local_max_size
      select op {
        when "<<" {
          forall t in tmp do
            t = if has_max_bits && val >= max_bits then 0: bigint else t << val;
        }
        when ">>" {
          forall t in tmp do
            t = if has_max_bits && val >= max_bits then 0: bigint else t >> val;
        }
        when "**" {
          if val < 0 { // This could actually lead into real number territory
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            forall t in tmp with (var local_max_size = max_size) {
              powMod(t, t, val, local_max_size + 1);
            }
          }
          else {
            forall t in tmp do t = t ** val;
          }
        }
      }
    } else if swapOperands && val.type == bigint && vec.etype != bigint && smallOps.contains(op) {
      // ops that only work with a left hand side of bigint and right hand side non-bigint
      // Just bitshifts and exponentiation without local_max_size
      select op {
        when "<<" {
          forall (t, v) in zip(tmp, veca) do
            t = if has_max_bits && v >= max_bits then 0: bigint else val << v;
        }
        when ">>" {
          forall (t, v) in zip(tmp, veca) do
            t = if has_max_bits && v >= max_bits then 0: bigint else val >> v;
        }
        when "**" {
          if || reduce (tmp<0) { // This could actually lead into real number territory
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            forall (t, v) in zip(tmp, veca) with (var local_max_size = max_size) {
              powMod(t, val, v, local_max_size + 1);
            }
          }
          else {
            forall (t, v) in zip(tmp, veca) do t = val ** v;
          }
        }
      }
    } else {
      select op {
        when "&" { forall t in tmp do t &= ri; }
        when "|" { forall t in tmp do t |= ri; }
        when "^" { forall t in tmp do t ^= ri; }
        when "+" { forall t in tmp do t += ri; }
        when "*" { forall t in tmp do t *= ri; }
        when "-" { 
          if !swapOperands {
            forall t in tmp do t -= ri;
          } else {
            forall t in tmp do t = ri - t; 
          }
        }
        when "//" {
          if !swapOperands {
            forall t in tmp {
              if ri != 0 {
                var q: bigint;
                // floor-style integer division, like Python's //
                div(q, t, ri, roundingMode.down);
                t = q;
              } else {
                // whatever semantics you want for division by zero:
                t = 0: bigint;
              }
            }
          } else {
            forall t in tmp {
              const denom: bigint = t;   // <- cast bool/int/uint/etc to bigint
              if denom != 0 {
                var q: bigint;
                // floor-style integer division, like Python's //
                div(q, ri, denom, roundingMode.down);
                t = q;
              } else {
                // whatever semantics you want for division by zero:
                t = 0: bigint;
              }
            }
          }
        }
        when "%" {
          if !swapOperands {
            forall t in tmp {
              if ri != 0 {
                mod(t, t, ri);
              } else {
                t = 0: bigint;
              }
            }
          } else {
            forall t in tmp {
              if t != 0 {
                mod(t, ri, t);
              } else {
                t = 0: bigint;
              }
            }
          }
        }
        when "**" {
          if !swapOperands {
            if ri < 0 {
              throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
            }
            if has_max_bits {
              forall t in tmp with (var local_max_size = max_size) {
                powMod(t, t, ri, local_max_size + 1);
              }
            }
            else {
              throw new Error("Attempt to exponentiate base of type BigInt to BigInt without max_bits");
            }
          } else {
            if || reduce (tmp<0) {
              throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
            }
            if has_max_bits {
              forall t in tmp with (var local_max_size = max_size) {
                powMod(t, ri, t, local_max_size + 1);
              }
            }
            else {
              throw new Error("Attempt to exponentiate base of type BigInt to BigInt without max_bits");
            }
          }
        }
        when "<<<" {
          if !swapOperands {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotl");
            }
            var modded_shift = 0: bigint;
            forall (t, bot_bits) in zip(tmp, veca) with (var loc_modded_shift = modded_shift) {
              mod(loc_modded_shift, ri, max_bits);
              t <<= loc_modded_shift: int;
              const shift_amt = max_bits - loc_modded_shift: int;
              t += bot_bits >> shift_amt;
            }
          } else {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotl");
            }
            var botBits = val: bigint;
            var modded_shift = 0: bigint;
            forall (t, v) in zip(tmp, veca) with (var loc_modded_shift = modded_shift) {
              mod(loc_modded_shift, v: bigint, max_bits);
              t <<= loc_modded_shift: int;
              var shift_amt = max_bits - loc_modded_shift: int;
              t += botBits >> shift_amt;
            }
          }
        }
        when ">>>" {
          if !swapOperands {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotr");
            }
            var modded_shift = 0: bigint;
            forall (t, top_bits) in zip(tmp, veca) with (var loc_modded_shift = modded_shift) {
              mod(loc_modded_shift, ri, max_bits);
              t >>= loc_modded_shift: int;
              const shift_amt = max_bits - loc_modded_shift: int;
              t += top_bits << shift_amt;
            }
          } else {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotr");
            }
            var topBits = val: bigint;
            var modded_shift = 0: bigint;
            forall (t, v) in zip(tmp, veca) with (var loc_modded_shift = modded_shift) {
              mod(loc_modded_shift, v: bigint, max_bits);
              t >>= loc_modded_shift: int;
              var shift_amt = max_bits - loc_modded_shift: int;
              t += topBits << shift_amt;
            }
          }
        }
        otherwise do throw new Error("Unsupported operation: " + (if !swapOperands then (vec.etype:string +" "+ op +" "+ val.type:string)
                                                               else (val.type:string +" "+ op +" "+ vec.etype:string)));
      }
    }

    if has_max_bits then forall t in tmp with (const local_max_size = max_size) do t &= local_max_size;

    return (tmp, max_bits);
  }

  proc _doBigIntBinOpVLeftScalarBoolReturn(vec, val, op: string, param swapOperands: bool = false) throws {
    ref va = vec.a;
    var tmp = makeDistArray((...va.shape), bool);
    select op {
      when "<"  { forall (t, xi) in zip(tmp, va) with (var local_val = val) do t = if !swapOperands then (xi < local_val) else (local_val < xi); }
      when ">"  { forall (t, xi) in zip(tmp, va) with (var local_val = val) do t = if !swapOperands then (xi > local_val) else (local_val > xi); }
      when "<=" { forall (t, xi) in zip(tmp, va) with (var local_val = val) do t = if !swapOperands then (xi <= local_val) else (local_val <= xi); }
      when ">=" { forall (t, xi) in zip(tmp, va) with (var local_val = val) do t = if !swapOperands then (xi >= local_val) else (local_val >= xi); }
      when "==" { forall (t, xi) in zip(tmp, va) with (var local_val = val) do t = (xi == local_val); }
      when "!=" { forall (t, xi) in zip(tmp, va) with (var local_val = val) do t = (xi != local_val); }
      otherwise do throw new Error("Unsupported operation: " + (if !swapOperands then (vec.etype:string +" "+ op +" "+ val.type:string)
                                                               else (val.type:string +" "+ op +" "+ vec.etype:string)));
    }
    return tmp;
  }

  proc doBigIntBinOpvs(l, val, op: string) throws {
    return _doBigIntBinOpVLeftScalar(l, val, op, false);
  }

  proc doBigIntBinOpsv(val, r, op: string) throws {
    return _doBigIntBinOpVLeftScalar(r, val, op, true);
  }

  proc doBigIntBinOpvsBoolReturn(l, val, op: string) throws {
    return _doBigIntBinOpVLeftScalarBoolReturn(l, val, op, false);
  }

  proc doBigIntBinOpsvBoolReturn(val, r, op: string) throws {
    return _doBigIntBinOpVLeftScalarBoolReturn(r, val, op, true);
  }

  proc doBigIntBinOpvsBoolReturnRealInput(const ref la: [?d] ?t1, ri: ?t2, op: string) throws
    where ( (t1 == bigint && t2 == real(64)) ||
            (t1 == real(64) && t2 == bigint) )
  {

    var e = makeDistArray(d, bool);
    ref ea = e;
    select op {
      when "<"  {
        if t1 == bigint {
          forall (ei, li) in zip(ea, la) do ei = ltBigReal(li, ri);
        } else { // t2 == bigint
          forall (ei, li) in zip(ea, la) do ei = gtBigReal(ri, li); // li<ri  <=>  ri>li
        }
      }
      when ">"  {
        if t1 == bigint {
          forall (ei, li) in zip(ea, la) do ei = gtBigReal(li, ri);
        } else {
          forall (ei, li) in zip(ea, la) do ei = ltBigReal(ri, li);
        }
      }
      when "<=" {
        if t1 == bigint {
          forall (ei, li) in zip(ea, la) do ei = leBigReal(li, ri);
        } else {
          forall (ei, li) in zip(ea, la) do ei = geBigReal(ri, li);
        }
      }
      when ">=" {
        if t1 == bigint {
          forall (ei, li) in zip(ea, la) do ei = geBigReal(li, ri);
        } else {
          forall (ei, li) in zip(ea, la) do ei = leBigReal(ri, li);
        }
      }
      when "==" {
        if t1 == bigint {
          forall (ei, li) in zip(ea, la) do ei = eqBigReal(li, ri);
        } else {
          forall (ei, li) in zip(ea, la) do ei = eqBigReal(ri, li);
        }
      }
      when "!=" {
        if t1 == bigint {
          forall (ei, li) in zip(ea, la) do ei = neBigReal(li, ri);
        } else {
          forall (ei, li) in zip(ea, la) do ei = neBigReal(ri, li);
        }
      }
      otherwise do
        throw new Error("Unsupported operation: " + t1:string + " " + op + " " + t2:string);
    }

    return e;
  }

  proc doBigIntBinOpvsBoolReturnRealInput(const ref la: [?d] ?t1, val: ?t2, op: string) throws
    where ( (t1 != bigint || t2 != real(64)) &&
            (t1 != real(64) || t2 != bigint) )
  {
    throw new Error("Unsupported operation: " + t1:string +" "+ op +" "+ t2:string);
  }

  proc doBigIntBinOpvsRealReturn(l, val, op: string) throws {
    select op {
      when "+" { return l.a: real + val: real; }
      when "-" { return l.a: real - val: real; }
      when "*" { return l.a: real * val: real; }
      when "/" { return l.a: real / val: real; }
      when "**" { return l.a: real ** val: real; }
      when "%" {
        var e = makeDistArray((...l.tupShape), real);
        ref ea = e;
        ref la = l.a;
        [(ei,li) in zip(ea,la)] ei = modHelper(li: real, val: real): real;
        return e;
      }
      when "//" {
        var e = makeDistArray((...l.tupShape), real);
        ref ea = e;
        ref la = l.a;
        [(ei,li) in zip(ea,la)] ei = floorDivisionHelper(li: real, val: real): real;
        return e;
      }
      otherwise {
        throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ val.type:string);
      }
    }
  }

  proc doBigIntBinOpsvBoolReturnRealInput(li: ?t1, const ref ra: [?d] ?t2, op: string) throws
    where ( (t1 == bigint && t2 == real(64)) ||
            (t1 == real(64) && t2 == bigint) )
  {

    var e = makeDistArray(d, bool);
    ref ea = e;
    select op {
      when "<"  {
        if t1 == bigint {
          forall (ei, ri) in zip(ea, ra) do ei = ltBigReal(li, ri);
        } else { // t2 == bigint
          forall (ei, ri) in zip(ea, ra) do ei = gtBigReal(ri, li); // li<ri  <=>  ri>li
        }
      }
      when ">"  {
        if t1 == bigint {
          forall (ei, ri) in zip(ea, ra) do ei = gtBigReal(li, ri);
        } else {
          forall (ei, ri) in zip(ea, ra) do ei = ltBigReal(ri, li);
        }
      }
      when "<=" {
        if t1 == bigint {
          forall (ei, ri) in zip(ea, ra) do ei = leBigReal(li, ri);
        } else {
          forall (ei, ri) in zip(ea, ra) do ei = geBigReal(ri, li);
        }
      }
      when ">=" {
        if t1 == bigint {
          forall (ei, ri) in zip(ea, ra) do ei = geBigReal(li, ri);
        } else {
          forall (ei, ri) in zip(ea, ra) do ei = leBigReal(ri, li);
        }
      }
      when "==" {
        if t1 == bigint {
          forall (ei, ri) in zip(ea, ra) do ei = eqBigReal(li, ri);
        } else {
          forall (ei, ri) in zip(ea, ra) do ei = eqBigReal(ri, li);
        }
      }
      when "!=" {
        if t1 == bigint {
          forall (ei, ri) in zip(ea, ra) do ei = neBigReal(li, ri);
        } else {
          forall (ei, ri) in zip(ea, ra) do ei = neBigReal(ri, li);
        }
      }
      otherwise do
        throw new Error("Unsupported operation: " + t1:string + " " + op + " " + t2:string);
    }

    return e;
  }

  proc doBigIntBinOpsvBoolReturnRealInput(li: ?t1, const ref ra: [?d] ?t2, op: string) throws
    where ( (t1 != bigint || t2 != real(64)) &&
            (t1 != real(64) || t2 != bigint) )
  {
    throw new Error("Unsupported operation: " + t1:string +" "+ op +" "+ t2:string);
  }

  proc doBigIntBinOpsvRealReturn(l, r, op: string) throws {
    select op {
      when "+" { return l: real + r.a: real; }
      when "-" { return l: real - r.a: real; }
      when "*" { return l: real * r.a: real; }
      when "/" { return l: real / r.a: real; }
      when "**" { return l: real ** r.a: real; }
      when "%" {
        var e = makeDistArray((...r.tupShape), real);
        ref ea = e;
        ref ra = r.a;
        [(ei,ri) in zip(ea,ra)] ei = modHelper(l: real, ri: real): real;
        return e;
      }
      when "//" {
        var e = makeDistArray((...r.tupShape), real);
        ref ea = e;
        ref ra = r.a;
        [(ei,ri) in zip(ea,ra)] ei = floorDivisionHelper(l: real, ri: real): real;
        return e;
      }
      otherwise {
        throw new Error("Unsupported operation: " + l.type:string +" "+ op +" "+ r.etype:string);
      }
    }
  }

}
