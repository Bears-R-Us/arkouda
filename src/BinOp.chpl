
module BinOp
{
  use ServerConfig;

  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Logging;
  use Message;
  use BitOps;
  use BigInteger;


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
  inline proc floorDivisionHelper(numerator: ?t, denom: ?t2): real {
    if (numerator == 0 && denom == 0) || (isInf(numerator) && (denom != 0 || isInf(denom))){
      return nan;
    }
    else if (numerator > 0 && denom == -inf) || (numerator < 0 && denom == inf){
      return -1:real;
    }
    else {
      return floor(numerator/denom);
    }
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


  proc doBoolBoolBitOp(
    op:string, ref e: [] bool, l: [] bool, r /*: [] bool OR bool*/
  ): bool {
    select op {
      when "|" { e = l | r; }
      when "&" { e = l & r; }
      when "*" { e = l & r; }
      when "^" { e = l ^ r; }
      when "+" { e = l | r; }
      otherwise do return false;
    }
    return true;
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
  proc doBinOpvv(l, r, type lType, type rType, type etype, op: string, pn, st): MsgTuple throws {
    var e = makeDistArray((...l.tupShape), etype);

    const nie = notImplementedError(pn,l.dtype,op,r.dtype);

    use Set;
    var boolOps: set(string);
        boolOps.add("<");
        boolOps.add("<=");
        boolOps.add(">");
        boolOps.add(">=");
        boolOps.add("==");
        boolOps.add("!=");

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

  proc doBinOpvs(l, val, type lType, type rType, type etype, op: string, pn, st): MsgTuple throws {
    var e = makeDistArray((...l.tupShape), etype);

    const nie = notImplementedError(pn,"%s %s %s".format(type2str(l.a.eltType),op,type2str(val.type)));

    use Set;
    var boolOps: set(string);
        boolOps.add("<");
        boolOps.add("<=");
        boolOps.add(">");
        boolOps.add(">=");
        boolOps.add("==");
        boolOps.add("!=");

    type castType = mySafeCast(lType, rType);

    // The compiler complains that maybe etype is bool if it gets down below this
    // without returning, so we have to kind of chunk this next piece off.

    // For similar reasons, everything else is kinda split off into its own thing.
    // The compiler has no common sense about things (and that's not really its fault)

    if etype == bool {

      if boolOps.contains(op) {

        select op {

          when "<" { e = (l.a: castType) < (val: castType); }
          when "<=" { e = (l.a: castType) <= (val: castType); }
          when ">" { e = (l.a: castType) > (val: castType); }
          when ">=" { e = (l.a: castType) >= (val: castType); }
          when "==" { e = (l.a: castType) == (val: castType); }
          when "!=" { e = (l.a: castType) != (val: castType); }
          otherwise do return MsgTuple.error(nie); // Shouldn't happen

        }
        return st.insert(new shared SymEntry(e));
      }

      if lType == bool && rType == bool {
        if !doBoolBoolBitOp(op, e, l.a, val) {
          return MsgTuple.error(nie);
        }
        return st.insert(new shared SymEntry(e));
      }

      return MsgTuple.error(nie);

    }

    else if lType == bool && rType == bool && etype == uint(8) { // Both bools is kinda weird
      select op {
        when "%" { e = (0: uint(8)); } // numpy has these as int(8), but Arkouda doesn't really support that type.
        when "//" { e = (l.a & val): uint(8); }
        when "**" { e = (!l.a & val): uint(8); }
        when "<<" { e = (l.a: uint(8)) << (val: uint(8)); }
        when ">>" { e = (l.a: uint(8)) >> (val: uint(8)); }
        otherwise do return MsgTuple.error(nie);
        // >>> and <<< could probably be implemented as int(8) or uint(8) things
      }
      return st.insert(new shared SymEntry(e));
    }

    else if etype == real(32) || etype == real(64) {

      select op {
        when "*" { e = (l.a: etype * val: etype): etype; }
        when "+" { e = (l.a: etype + val: etype): etype; }
        when "-" { e = (l.a: etype - val: etype): etype; }
        when "/" { e = (l.a: etype / val: etype): etype; }
        when "%" {
          ref ea = e;
          ref la = l.a;
          [(ei,li) in zip(ea,la)] ei = modHelper(li: etype, val: etype): etype;
        }
        when "//" {
          ref ea = e;
          ref la = l.a;
          [(ei,li) in zip(ea,la)] ei = floorDivisionHelper(li: etype, val: etype): etype;
        }
        when "**" {
          e = ((l.a: etype) ** (val: etype)): etype;
        }
        otherwise do return MsgTuple.error(nie);
      }
      return st.insert(new shared SymEntry(e));

    }

    else {

      select op {
        when "|" { e = (l.a | val): etype; }
        when "&" { e = (l.a & val): etype; }
        when "*" { e = (l.a * val): etype; }
        when "^" { e = (l.a ^ val): etype; }
        when "+" { e = (l.a + val): etype; }
        when "-" { e = (l.a - val): etype; }
        when "/" { e = (l.a: etype) / (val: etype); }
        when "%" {
          ref ea = e;
          ref la = l.a;
          [(ei,li) in zip(ea,la)] ei = if val != 0 then li%val else 0;
        }
        when "//" {
          ref ea = e;
          ref la = l.a;
          [(ei,li) in zip(ea,la)] ei = if val != 0 then (li/val): etype else 0: etype;
        }
        when "**" {
          if val < 0
            then return MsgTuple.error("Attempt to exponentiate base of type Int or UInt to negative exponent");
          e = (l.a: etype) ** (val: etype);
        }
        when "<<" {
          ref ea = e;
          ref la = l.a;
          [(ei,li) in zip(ea,la)] if (0 <= val && val < numBits(etype)) then ei = ((li: etype) << (val: etype)): etype;
        }
        when ">>" {
          ref ea = e;
          ref la = l.a;
          [(ei,li) in zip(ea,la)] if (0 <= val && val < numBits(etype)) then ei = ((li: etype) >> (val: etype)): etype;
        }
        when "<<<" { e = rotl(l.a: etype, val: etype); }
        when ">>>" { e = rotr(l.a: etype, val: etype); }
        otherwise do return MsgTuple.error(nie);
      }
      return st.insert(new shared SymEntry(e));
    }

    return MsgTuple.error(nie);
  }

  proc doBinOpsv(val, r, type lType, type rType, type etype, op: string, pn, st) throws {
    var e = makeDistArray((...r.tupShape), etype);
    const nie = notImplementedError(pn,"%s %s %s".format(type2str(val.type),op,type2str(r.a.eltType)));

    use Set;
    var boolOps: set(string);
        boolOps.add("<");
        boolOps.add("<=");
        boolOps.add(">");
        boolOps.add(">=");
        boolOps.add("==");
        boolOps.add("!=");

    type castType = mySafeCast(lType, rType);

    // The compiler complains that maybe etype is bool if it gets down below this
    // without returning, so we have to kind of chunk this next piece off.

    // For similar reasons, everything else is kinda split off into its own thing.
    // The compiler has no common sense about things (and that's not really its fault)

    if etype == bool {

      if boolOps.contains(op) {

        select op {

          when "<" { e = (val: castType) < (r.a: castType); }
          when "<=" { e = (val: castType) <= (r.a: castType); }
          when ">" { e = (val: castType) > (r.a: castType); }
          when ">=" { e = (val: castType) >= (r.a: castType); }
          when "==" { e = (val: castType) == (r.a: castType); }
          when "!=" { e = (val: castType) != (r.a: castType); }
          otherwise do return MsgTuple.error(nie); // Shouldn't happen

        }
        return st.insert(new shared SymEntry(e));
      }

      if lType == bool && rType == bool {
        if !doBoolBoolBitOp(op, e, r.a, val) {
          return MsgTuple.error(nie);
        }
        return st.insert(new shared SymEntry(e));
      }

      return MsgTuple.error(nie);

    }

    else if lType == bool && rType == bool && etype == uint(8) { // Both bools is kinda weird
      select op {
        when "%" { e = (0: uint(8)); } // numpy has these as int(8), but Arkouda doesn't really support that type.
        when "//" { e = (val & r.a): uint(8); }
        when "**" { e = (!val & r.a): uint(8); }
        when "<<" { e = (val: uint(8)) << (r.a: uint(8)); }
        when ">>" { e = (val: uint(8)) >> (r.a: uint(8)); }
        otherwise do return MsgTuple.error(nie);
        // >>> and <<< could probably be implemented as int(8) or uint(8) things
      }
      return st.insert(new shared SymEntry(e));
    }

    else if etype == real(32) || etype == real(64) {

      select op {
        when "*" { e = (val: etype * r.a: etype): etype; }
        when "+" { e = (val: etype + r.a: etype): etype; }
        when "-" { e = (val: etype - r.a: etype): etype; }
        when "/" { e = (val: etype / r.a: etype): etype; }
        when "%" {
          ref ea = e;
          ref ra = r.a;
          [(ei,ri) in zip(ea,ra)] ei = modHelper(val: etype, ri: etype): etype;
        }
        when "//" {
          ref ea = e;
          ref ra = r.a;
          [(ei,ri) in zip(ea,ra)] ei = floorDivisionHelper(val: etype, ri: etype): etype;
        }
        when "**" {
          e = ((val: etype) ** (r.a: etype)): etype;
        }
        otherwise do return MsgTuple.error(nie);
      }
      return st.insert(new shared SymEntry(e));

    }

    else {

      select op {
        when "|" { e = (val | r.a): etype; }
        when "&" { e = (val & r.a): etype; }
        when "*" { e = (val * r.a): etype; }
        when "^" { e = (val ^ r.a): etype; }
        when "+" { e = (val + r.a): etype; }
        when "-" { e = (val - r.a): etype; }
        when "/" { e = (val: etype) / (r.a: etype); }
        when "%" {
          ref ea = e;
          ref ra = r.a;
          [(ei,ri) in zip(ea,ra)] ei = if ri != 0 then val%ri else 0;
        }
        when "//" {
          ref ea = e;
          ref ra = r.a;
          [(ei,ri) in zip(ea,ra)] ei = if ri != 0 then (val/ri): etype else 0: etype;
        }
        when "**" {
          if || reduce (r.a<0)
            then return MsgTuple.error("Attempt to exponentiate base of type Int or UInt to negative exponent");
          e = (val: etype) ** (r.a: etype);
        }
        when "<<" {
          ref ea = e;
          ref ra = r.a;
          [(ei,ri) in zip(ea,ra)] if (0 <= ri && ri < numBits(etype)) then ei = ((val: etype) << (ri: etype)): etype;
        }
        when ">>" {
          ref ea = e;
          ref ra = r.a;
          [(ei,ri) in zip(ea,ra)] if (0 <= ri && ri < numBits(etype)) then ei = ((val: etype) >> (ri: etype)): etype;
        }
        when "<<<" { e = rotl(val: etype, r.a: etype); }
        when ">>>" { e = rotr(val: etype, r.a: etype); }
        otherwise do return MsgTuple.error(nie);
      }
      return st.insert(new shared SymEntry(e));
    }

    return MsgTuple.error(nie);
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
    // these cases are not mutually exclusive,
    // so we have a flag to track if tmp is ever populated
    var visted = false;

    // had to create bigint specific BinOp procs which return
    // the distributed array because we need it at SymEntry creation time
    if l.etype == bigint && r.etype == bigint {
      // first we try the ops that only work with
      // both being bigint
      select op {
        when "&" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t &= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "|" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t |= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "^" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t ^= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "/" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t /= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
      }
    }
    if l.etype == bigint && (r.etype == bigint || r.etype == int || r.etype == uint) {
      // then we try the ops that only work with a
      // left hand side of bigint
      if r.etype != bigint {
        // can't shift a bigint by a bigint
        select op {
          when "<<" {
            forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
              if has_max_bits {
                if ri >= max_bits {
                  t = 0;
                }
                else {
                  t <<= ri;
                  t &= local_max_size;
                }
              }
              else {
                t <<= ri;
              }
            }
            visted = true;
          }
          when ">>" {
            forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
              if has_max_bits {
                if ri >= max_bits {
                  t = 0;
                }
                else {
                  t >>= ri;
                  t &= local_max_size;
                }
              }
              else {
                t >>= ri;
              }
            }
            visted = true;
          }
          when "<<<" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotl");
            }
            var botBits = la;
            forall (t, ri, bot_bits) in zip(tmp, ra, botBits) with (var local_max_size = max_size) {
              var modded_shift = if r.etype == int then ri % max_bits else ri % max_bits:uint;
              t <<= modded_shift;
              var shift_amt = if r.etype == int then max_bits - modded_shift else max_bits:uint - modded_shift;
              bot_bits >>= shift_amt;
              t += bot_bits;
              t &= local_max_size;
            }
            visted = true;
          }
          when ">>>" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotr");
            }
            var topBits = la;
            forall (t, ri, tB) in zip(tmp, ra, topBits) with (var local_max_size = max_size) {
              var modded_shift = if r.etype == int then ri % max_bits else ri % max_bits:uint;
              t >>= modded_shift;
              var shift_amt = if r.etype == int then max_bits - modded_shift else max_bits:uint - modded_shift;
              tB <<= shift_amt;
              t += tB;
              t &= local_max_size;
            }
            visted = true;
          }
        }
      }
      select op {
        when "//" { // floordiv
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            if ri != 0 {
              t /= ri;
            }
            else {
              t = 0:bigint;
            }
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "%" { // modulo " <- quote is workaround for syntax highlighter bug
          // we only do in place mod when ri != 0, tmp will be 0 in other locations
          // we can't use ei = li % ri because this can result in negatives
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            if ri != 0 {
              mod(t, t, ri);
            }
            else {
              t = 0:bigint;
            }
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "**" {
          if || reduce (ra<0) {
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
              powMod(t, t, ri, local_max_size + 1);
            }
          }
          else {
            forall (t, ri) in zip(tmp, ra) {
              t **= ri:uint;
            }
          }
          visted = true;
        }
      }
    }
    if (l.etype == bigint && r.etype == bigint) ||
       (l.etype == bigint && (r.etype == int || r.etype == uint || r.etype == bool)) ||
       (r.etype == bigint && (l.etype == int || l.etype == uint || l.etype == bool)) {
      select op {
        when "+" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t += ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "-" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t -= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "*" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t *= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
      }
    }
    if !visted {
      throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ r.etype:string);
    }
    return (tmp, max_bits);
  }

  proc doBigIntBinOpvvBoolReturn(l, r, op: string) throws {
    select op {
      when "<" {
        return l.a < r.a;
      }
      when ">" {
        return l.a > r.a;
      }
      when "<=" {
        return l.a <= r.a;
      }
      when ">=" {
        return l.a >= r.a;
      }
      when "==" {
        return l.a == r.a;
      }
      when "!=" {
        return l.a != r.a;
      }
      otherwise {
        // we should never reach this since we only enter this proc
        // if boolOps.contains(op)
        throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ r.etype:string);
      }
    }
  }

  proc doBigIntBinOpvs(l, val, op: string) throws {
    var max_bits = l.max_bits;
    var max_size = 1:bigint;
    var has_max_bits = max_bits != -1;
    if has_max_bits {
      max_size <<= max_bits;
      max_size -= 1;
    }
    ref la = l.a;
    var tmp = if l.etype == bigint then la else la:bigint;
    // these cases are not mutually exclusive,
    // so we have a flag to track if tmp is ever populated
    var visted = false;

    // had to create bigint specific BinOp procs which return
    // the distributed array because we need it at SymEntry creation time
    if l.etype == bigint && val.type == bigint {
      // first we try the ops that only work with
      // both being bigint
      select op {
        when "&" {
          forall t in tmp with (var local_val = val, var local_max_size = max_size) {
            t &= local_val;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "|" {
          forall t in tmp with (var local_val = val, var local_max_size = max_size) {
            t |= local_val;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "^" {
          forall t in tmp with (var local_val = val, var local_max_size = max_size) {
            t ^= local_val;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "/" {
          forall t in tmp with (var local_val = val, var local_max_size = max_size) {
            t /= local_val;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
      }
    }
    if l.etype == bigint && (val.type == bigint || val.type == int || val.type == uint) {
      // then we try the ops that only work with a
      // left hand side of bigint
      if val.type != bigint {
        // can't shift a bigint by a bigint
        select op {
          when "<<" {
            if has_max_bits && val >= max_bits {
              forall t in tmp with (var local_zero = 0:bigint) {
                t = local_zero;
              }
            }
            else {
              forall t in tmp with (var local_val = val, var local_max_size = max_size) {
                t <<= local_val;
                if has_max_bits {
                  t &= local_max_size;
                }
              }
            }
            visted = true;
          }
          when ">>" {
            if has_max_bits && val >= max_bits {
              forall t in tmp with (var local_zero = 0:bigint) {
                t = local_zero;
              }
            }
            else {
              forall t in tmp with (var local_max_size = max_size) {
                t >>= val;
                if has_max_bits {
                  t &= local_max_size;
                }
              }
            }
            visted = true;
          }
          when "<<<" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotl");
            }
            var botBits = la;
            var modded_shift = if val.type == int then val % max_bits else val % max_bits:uint;
            var shift_amt = if val.type == int then max_bits - modded_shift else max_bits:uint - modded_shift;
            forall (t, bot_bits) in zip(tmp, botBits) with (var local_val = modded_shift, var local_shift_amt = shift_amt, var local_max_size = max_size) {
              t <<= local_val;
              bot_bits >>= local_shift_amt;
              t += bot_bits;
              t &= local_max_size;
            }
            visted = true;
          }
          when ">>>" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotr");
            }
            var topBits = la;
            var modded_shift = if val.type == int then val % max_bits else val % max_bits:uint;
            var shift_amt = if val.type == int then max_bits - modded_shift else max_bits:uint - modded_shift;
            forall (t, tB) in zip(tmp, topBits) with (var local_val = modded_shift, var local_shift_amt = shift_amt, var local_max_size = max_size) {
              t >>= local_val;
              tB <<= local_shift_amt;
              t += tB;
              t &= local_max_size;
            }
            visted = true;
          }
        }
      }
      select op {
        when "//" { // floordiv
          forall t in tmp with (var local_val = val, var local_max_size = max_size) {
            if local_val != 0 {
              t /= local_val;
            }
            else {
              t = 0:bigint;
            }
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "%" { // modulo " <- quote is workaround for syntax highlighter bug
          // we only do in place mod when val != 0, tmp will be 0 in other locations
          // we can't use ei = li % val because this can result in negatives
          forall t in tmp with (var local_val = val, var local_max_size = max_size) {
            if local_val != 0 {
              mod(t, t, local_val);
            }
            else {
              t = 0:bigint;
            }
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "**" {
          if val<0 {
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            forall t in tmp with (var local_val = val, var local_max_size = max_size) {
              powMod(t, t, local_val, local_max_size + 1);
            }
          }
          else {
            forall t in tmp with (var local_val = val) {
              t **= local_val:uint;
            }
          }
          visted = true;
        }
      }
    }
    if (l.etype == bigint && val.type == bigint) ||
       (l.etype == bigint && (val.type == int || val.type == uint || val.type == bool)) ||
       (val.type == bigint && (l.etype == int || l.etype == uint || l.etype == bool)) {
      select op {
        when "+" {
          forall t in tmp with (var local_val = val, var local_max_size = max_size) {
            t += local_val;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "-" {
          forall t in tmp with (var local_val = val, var local_max_size = max_size) {
            t -= local_val;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "*" {
          forall t in tmp with (var local_val = val, var local_max_size = max_size) {
            t *= local_val;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
      }
    }
    if !visted {
      throw new Error("Unsupported operation: " + l.etype:string +" "+ op +" "+ val.type:string);
    }
    return (tmp, max_bits);
  }

  proc doBigIntBinOpvsBoolReturn(l, val, op: string) throws {
    ref la = l.a;
    var tmp = makeDistArray((...la.shape), bool);
    select op {
      when "<" {
        forall (t, li) in zip(tmp, la) with (var local_val = val) {
          t = (li < local_val);
        }
      }
      when ">" {
        forall (t, li) in zip(tmp, la) with (var local_val = val) {
          t = (li > local_val);
        }
      }
      when "<=" {
        forall (t, li) in zip(tmp, la) with (var local_val = val) {
          t = (li <= local_val);
        }
      }
      when ">=" {
        forall (t, li) in zip(tmp, la) with (var local_val = val) {
          t = (li >= local_val);
        }
      }
      when "==" {
        forall (t, li) in zip(tmp, la) with (var local_val = val) {
          t = (li == local_val);
        }
      }
      when "!=" {
        forall (t, li) in zip(tmp, la) with (var local_val = val) {
          t = (li != local_val);
        }
      }
      otherwise {
        // we should never reach this since we only enter this proc
        // if boolOps.contains(op)
        throw new Error("Unsupported operation: " +" "+ l.etype:string + op +" "+ val.type:string);
      }
    }
    return tmp;
  }

  proc doBigIntBinOpsv(val, r, op: string) throws {
    var max_bits = r.max_bits;
    var max_size = 1:bigint;
    var has_max_bits = max_bits != -1;
    if has_max_bits {
      max_size <<= max_bits;
      max_size -= 1;
    }
    ref ra = r.a;
    var tmp = makeDistArray((...ra.shape), bigint);
    tmp = val:bigint;
    // these cases are not mutually exclusive,
    // so we have a flag to track if tmp is ever populated
    var visted = false;

    // had to create bigint specific BinOp procs which return
    // the distributed array because we need it at SymEntry creation time
    if val.type == bigint && r.etype == bigint {
      // first we try the ops that only work with
      // both being bigint
      select op {
        when "&" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t &= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "|" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t |= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "^" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t ^= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "/" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t /= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
      }
    }
    if val.type == bigint && (r.etype == bigint || r.etype == int || r.etype == uint) {
      // then we try the ops that only work with a
      // left hand side of bigint
      if r.etype != bigint {
        // can't shift a bigint by a bigint
        select op {
          when "<<" {
            forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
              if has_max_bits {
                if ri >= max_bits {
                  t = 0;
                }
                else {
                  t <<= ri;
                  t &= local_max_size;
                }
              }
              else {
                t <<= ri;
              }
            }
            visted = true;
          }
          when ">>" {
            forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
              if has_max_bits {
                if ri >= max_bits {
                  t = 0;
                }
                else {
                  t >>= ri;
                  t &= local_max_size;
                }
              }
              else {
                t >>= ri;
              }
            }
            visted = true;
          }
          when "<<<" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotl");
            }
            var botBits = makeDistArray((...ra.shape), bigint);
            botBits = val;
            forall (t, ri, bot_bits) in zip(tmp, ra, botBits) with (var local_max_size = max_size) {
              var modded_shift = if r.etype == int then ri % max_bits else ri % max_bits:uint;
              t <<= modded_shift;
              var shift_amt = if r.etype == int then max_bits - modded_shift else max_bits:uint - modded_shift;
              bot_bits >>= shift_amt;
              t += bot_bits;
              t &= local_max_size;
            }
            visted = true;
          }
          when ">>>" {
            if !has_max_bits {
              throw new Error("Must set max_bits to rotr");
            }
            var topBits = makeDistArray((...ra.shape), bigint);
            topBits = val;
            forall (t, ri, tB) in zip(tmp, ra, topBits) with (var local_max_size = max_size) {
              var modded_shift = if r.etype == int then ri % max_bits else ri % max_bits:uint;
              t >>= modded_shift;
              var shift_amt = if r.etype == int then max_bits - modded_shift else max_bits:uint - modded_shift;
              tB <<= shift_amt;
              t += tB;
              t &= local_max_size;
            }
            visted = true;
          }
        }
      }
      select op {
        when "//" { // floordiv
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            if ri != 0 {
              t /= ri;
            }
            else {
              t = 0:bigint;
            }
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "%" { // modulo " <- quote is workaround for syntax highlighter bug
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            if ri != 0 {
              mod(t, t, ri);
            }
            else {
              t = 0:bigint;
            }
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "**" {
          if || reduce (ra<0) {
            throw new Error("Attempt to exponentiate base of type BigInt to negative exponent");
          }
          if has_max_bits {
            forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
              powMod(t, t, ri, local_max_size + 1);
            }
          }
          else {
            forall (t, ri) in zip(tmp, ra) {
              t **= ri:uint;
            }
          }
          visted = true;
        }
      }
    }
    if (val.type == bigint && r.etype == bigint) ||
       (val.type == bigint && (r.etype == int || r.etype == uint || r.etype == bool)) ||
       (r.etype == bigint && (val.type == int || val.type == uint || val.type == bool)) {
      select op {
        when "+" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t += ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "-" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t -= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
        when "*" {
          forall (t, ri) in zip(tmp, ra) with (var local_max_size = max_size) {
            t *= ri;
            if has_max_bits {
              t &= local_max_size;
            }
          }
          visted = true;
        }
      }
    }
    if !visted {
      throw new Error("Unsupported operation: " + val.type:string +" "+ op +" "+ r.etype:string);
    }
    return (tmp, max_bits);
  }

  proc doBigIntBinOpsvBoolReturn(val, r, op: string) throws {
    ref ra = r.a;
    var tmp = makeDistArray((...ra.shape), bool);
    select op {
      when "<" {
        forall (t, ri) in zip(tmp, ra) with (var local_val = val) {
          t = (local_val < ri);
        }
      }
      when ">" {
        forall (t, ri) in zip(tmp, ra) with (var local_val = val) {
          t = (local_val > ri);
        }
      }
      when "<=" {
        forall (t, ri) in zip(tmp, ra) with (var local_val = val) {
          t = (local_val <= ri);
        }
      }
      when ">=" {
        forall (t, ri) in zip(tmp, ra) with (var local_val = val) {
          t = (local_val >= ri);
        }
      }
      when "==" {
        forall (t, ri) in zip(tmp, ra) with (var local_val = val) {
          t = (local_val == ri);
        }
      }
      when "!=" {
        forall (t, ri) in zip(tmp, ra) with (var local_val = val) {
          t = (local_val != ri);
        }
      }
      otherwise {
        // we should never reach this since we only enter this proc
        // if boolOps.contains(op)
        throw new Error("Unsupported operation: " + val.type:string +" "+ op +" "+ r.etype:string);
      }
    }
    return tmp;
  }
}
