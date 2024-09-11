
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
      Supports the following binary operations between two arrays:
      +, -, *, %, **, //
    */
    @arkouda.instantiateAndRegister
    proc arithmeticOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a != bool || binop_dtype_b != bool) &&
            binop_dtype_a != bigint && binop_dtype_b != bigint
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      if a.tupShape != b.tupShape then
        return MsgTuple.error("array shapes must match for element-wise binary operations");

      type resultType = promotedType(binop_dtype_a, binop_dtype_b);
      var result = makeDistArray((...a.tupShape), resultType);

      param isRealBoolOp = (binop_dtype_a == bool && isRealType(binop_dtype_b)) ||
                         (binop_dtype_b == bool && isRealType(binop_dtype_a));

      select op {
        when '+' do result = a.a:resultType + b.a:resultType;
        when '-' do result = a.a:resultType - b.a:resultType;
        when '*' do result = a.a:resultType * b.a:resultType;
        when '%' { // '
          if isRealBoolOp then return MsgTuple.error("'//' not supported between real and bool arrays");
          ref aa = a.a;
          ref bb = b.a;
          if isRealType(resultType)
            then [(ri,ai,bi) in zip(result,aa,bb)] ri = modHelper(ai:resultType, bi:resultType);
            else [(ri,ai,bi) in zip(result,aa,bb)] ri = if bi != 0:binop_dtype_b then (ai % bi):resultType else 0:resultType;
        }
        when '**' {
          if isRealBoolOp then return MsgTuple.error("'//' not supported between real and bool arrays");

          if isIntegralType(binop_dtype_a) && (|| reduce (a.a < 0)) {
            return MsgTuple.error("Attempt to exponentiate integer base to negative exponent");
          }
          result = a.a:resultType ** b.a:resultType;
        }
        when '//' {
          if isRealBoolOp then return MsgTuple.error("'//' not supported between real and bool arrays");

          ref aa = a.a;
          ref bb = b.a;
          if isRealType(resultType)
            then [(ri,ai,bi) in zip(result,aa,bb)] ri = floorDivisionHelper(ai:resultType, bi:resultType);
            else [(ri,ai,bi) in zip(result,aa,bb)] ri = if bi != 0:binop_dtype_b then (ai / bi):resultType else 0:resultType;
        }
        otherwise return MsgTuple.error("unknown arithmetic binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bool-bool arithmetic
    proc arithmeticOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a == bool && binop_dtype_b == bool
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(bool, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(bool, array_nd),
            op = msgArgs['op'].toScalar(string);

      if a.tupShape != b.tupShape then
        return MsgTuple.error("array shapes must match for element-wise binary operations");

      var result = makeDistArray((...a.tupShape), bool);

      // TODO: implement %, **, and // following NumPy's behavior
      select op {
        when '+' do result = a.a | b.a;
        when '-' do return MsgTuple.error("cannot subtract boolean arrays");
        when '*' do result = a.a & b.a;
        when '%' do return MsgTuple.error("modulo between two boolean arrays is not supported"); // '
        when '**' do return MsgTuple.error("exponentiation between two boolean arrays is not supported");
        when '//' do return MsgTuple.error("floor-division between two boolean arrays is not supported");
        otherwise return MsgTuple.error("unknown arithmetic binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bigint arithmetic
    proc arithmeticOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint && !isRealType(binop_dtype_b)) ||
            (binop_dtype_b == bigint && !isRealType(binop_dtype_a))
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string),
            (has_max_bits, max_size, max_bits) = getMaxBits(a, b);

      if a.tupShape != b.tupShape then
        return MsgTuple.error("array shapes must match for element-wise binary operations");

      param bigintBoolOp = binop_dtype_a == bigint && binop_dtype_b == bool ||
                           binop_dtype_b == bigint && binop_dtype_a == bool;

      var result = makeDistArray((...a.tupShape), bigint);
      result = a.a:bigint;
      ref ba = b.a;

      select op {
        when '+' {
          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            rx += bx;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '-' {
          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            rx -= bx;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '*' {
          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            rx *= bx;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '%' { // '
          if bigintBoolOp then return MsgTuple.error("'%' between bigint and bool arrays is not supported");
          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            if bx != 0 then mod(rx, rx, bx);
            else rx = 0;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '**' {
          if bigintBoolOp then return MsgTuple.error("'**' between bigint and bool arrays is not supported");

          if || reduce (b.a < 0) then
            return MsgTuple.error("Attempt to exponentiate bigint base to negative exponent");

          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) do
            powMod(rx, rx, bx, local_max_size + 1);
        }
        when '//' {
          if bigintBoolOp then return MsgTuple.error("'//' between bigint and bool arrays is not supported");

          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            if bx != 0 then rx /= bx;
            else rx = 0;
            if has_max_bits then rx &= local_max_size;
          }
        }
      }

      return st.insert(new shared SymEntry(result, max_bits));
    }

    proc arithmeticOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint && isRealType(binop_dtype_b)) ||
            (binop_dtype_b == bigint && isRealType(binop_dtype_a))
    {
      return MsgTuple.error("binary arithmetic operations between real and bigint arrays are not supported");
    }

    /*
      Supports the following binary operations between two arrays:
      ==, !=, <, <=, >, >=
    */
    @arkouda.instantiateAndRegister
    proc comparisonOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where !(binop_dtype_a == bigint && isRealType(binop_dtype_b)) &&
            !(binop_dtype_b == bigint && isRealType(binop_dtype_a))
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      if a.tupShape != b.tupShape then
        return MsgTuple.error("array shapes must match for element-wise binary operations");

      var result = makeDistArray((...a.tupShape), bool);

      if (binop_dtype_a == real && binop_dtype_b == bool) ||
         (binop_dtype_a == bool && binop_dtype_b == real)
      {
        select op {
          when '==' do result = a.a:real == b.a:real;
          when '!=' do result = a.a:real != b.a:real;
          when '<'  do result = a.a:real <  b.a:real;
          when '<=' do result = a.a:real <= b.a:real;
          when '>'  do result = a.a:real >  b.a:real;
          when '>=' do result = a.a:real >= b.a:real;
          otherwise return MsgTuple.error("unknown comparison binary operation: " + op);
        }
      } else {
        select op {
          when '==' do result = a.a == b.a;
          when '!=' do result = a.a != b.a;
          when '<'  do result = a.a <  b.a;
          when '<=' do result = a.a <= b.a;
          when '>'  do result = a.a >  b.a;
          when '>=' do result = a.a >= b.a;
          otherwise return MsgTuple.error("unknown comparison binary operation: " + op);
        }
      }

      return st.insert(new shared SymEntry(result));
    }

    proc comparisonOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint && isRealType(binop_dtype_b)) ||
            (binop_dtype_b == bigint && isRealType(binop_dtype_a))
    {
      return MsgTuple.error("comparison operations between real and bigint arrays are not supported");
    }

    /*
      Supports the following binary operations between two arrays:
      |, &, ^, <<, >>
    */
    @arkouda.instantiateAndRegister
    proc bitwiseOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a != bigint && binop_dtype_b != bigint &&
            !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b) &&
            !(binop_dtype_a == bool && binop_dtype_b == bool)
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      if a.tupShape != b.tupShape then
        return MsgTuple.error("array shapes must match for element-wise binary operations");

      type resultType = if (isIntType(binop_dtype_a) && isUintType(binop_dtype_b)) ||
                           (isUintType(binop_dtype_a) && isIntType(binop_dtype_b))
          then binop_dtype_a // use LHS type for integer types with non-matching signed-ness
          else promotedType(binop_dtype_a, binop_dtype_b); // otherwise, use regular type promotion
      var result = makeDistArray((...a.tupShape), resultType);

      select op {
        when '|'  do result = (a.a | b.a):resultType;
        when '&'  do result = (a.a & b.a):resultType;
        when '^'  do result = (a.a ^ b.a):resultType;
        when '<<' {
          ref aa = a.a;
          ref bb = b.a;
          [(ri,ai,bi) in zip(result,aa,bb)] if 0 <= bi && bi < 64 then ri = ai:resultType << bi:resultType;
        }
        when '>>' {
          ref aa = a.a;
          ref bb = b.a;
          [(ri,ai,bi) in zip(result,aa,bb)] if 0 <= bi && bi < 64 then ri = ai:resultType >> bi:resultType;
        }
        when '<<<' {
          if !isIntegral(binop_dtype_a) || !isIntegral(binop_dtype_b)
            then return MsgTuple.error("cannot perform bitwise rotation with boolean arrays");
          result = rotl(a.a, b.a):resultType;
        }
        when '>>>' {
          if !isIntegral(binop_dtype_a) || !isIntegral(binop_dtype_b)
            then return MsgTuple.error("cannot perform bitwise rotation with boolean arrays");
          result = rotr(a.a, b.a):resultType;
        }
        otherwise return MsgTuple.error("unknown bitwise binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bitwise ops with two boolean arrays
    proc bitwiseOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a == bool && binop_dtype_b == bool
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      if op == '<<' || op == '>>' {
        var result = makeDistArray((...a.tupShape), int);

        if op == '<<' {
          ref aa = a.a;
          ref bb = b.a;
          [(ri,ai,bi) in zip(result,aa,bb)] if 0 <= bi && bi < 64 then ri = ai:int << bi:int;
        } else {
          ref aa = a.a;
          ref bb = b.a;
          [(ri,ai,bi) in zip(result,aa,bb)] if 0 <= bi && bi < 64 then ri = ai:int >> bi:int;
        }

        return st.insert(new shared SymEntry(result));
      } else {
        var result = makeDistArray((...a.tupShape), bool);

        select op {
          when '|'  do result = a.a | b.a;
          when '&'  do result = a.a & b.a;
          when '^'  do result = a.a ^ b.a;
          when '<<<' do return MsgTuple.error("bitwise rotation on boolean arrays is not supported");
          when '>>>' do return MsgTuple.error("bitwise rotation on boolean arrays is not supported");
          otherwise return MsgTuple.error("unknown bitwise binary operation: " + op);
        }

        return st.insert(new shared SymEntry(result));
      }
    }

    // special handling for bitwise ops with at least one bigint array
    proc bitwiseOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
            !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b)
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string),
            (has_max_bits, max_size, max_bits) = getMaxBits(a, b);

      if a.tupShape != b.tupShape then
        return MsgTuple.error("array shapes must match for element-wise binary operations");

      // 'a' must be a bigint, and 'b' must not be real for the operations below
      // (for some operations, both a and b must be bigint)
      if binop_dtype_a != bigint || isRealType(binop_dtype_b) then
        return MsgTuple.error("bitwise operations between a LHS bigint and RHS non-bigint arrays are not supported");

      var result = makeDistArray((...a.tupShape), bigint);
      result = a.a:bigint;
      ref ba = b.a;
      param bothBigint = binop_dtype_a == bigint && binop_dtype_b == bigint;

      select op {
        when '|' {
          if !bothBigint then return MsgTuple.error("bitwise OR between bigint and non-bigint arrays is not supported");
          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            rx |= bx;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '&' {
          if !bothBigint then return MsgTuple.error("bitwise AND between bigint and non-bigint arrays is not supported");
          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            rx &= bx;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '^' {
          if !bothBigint then return MsgTuple.error("bitwise XOR between bigint and non-bigint arrays is not supported");
          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            rx ^= bx;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '<<' {
          if binop_dtype_b == bigint then return MsgTuple.error("left-shift of a bigint array by a non-bigint array is not supported");
          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            if has_max_bits {
              if bx >= max_bits {
                rx = 0;
              }
              else {
                rx <<= bx;
                rx &= local_max_size;
              }
            }
            else {
              rx <<= bx;
            }
          }
        }
        when '>>' {
          if binop_dtype_b == bigint then return MsgTuple.error("right-shift of a bigint array by a non-bigint array is not supported");
          forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
            if has_max_bits {
              if bx >= max_bits {
                rx = 0;
              }
              else {
                rx >>= bx;
                rx &= local_max_size;
              }
            }
            else {
              rx >>= bx;
            }
          }
        }
        when '<<<' {
          if !has_max_bits then return MsgTuple.error("bitwise rotation on bigint arrays requires a max_bits value");
          if binop_dtype_b == bigint then return MsgTuple.error("right-shift of a bigint array by a non-bigint array is not supported");
          forall (rx, bx) in zip(result, ba) with (var local_max_size = max_size) {
            var bot_bits = rx;
            const modded_shift = if binop_dtype_b == int then bx%max_bits else bx%max_bits:uint;
            rx <<= modded_shift;
            const shift_amt = if binop_dtype_b == int then max_bits - modded_shift else max_bits:uint - modded_shift;
            bot_bits >>= shift_amt;
            rx += bot_bits;
            rx &= local_max_size;
          }
        }
        when '>>>' {
          if !has_max_bits then return MsgTuple.error("bitwise rotation on bigint arrays requires a max_bits value");
          if binop_dtype_b == bigint then return MsgTuple.error("right-shift of a bigint array by a non-bigint array is not supported");
          forall (rx, bx) in zip(result, ba) with (var local_max_size = max_size) {
            var top_bits = rx;
            const modded_shift = if binop_dtype_b == int then bx%max_bits else bx%max_bits:uint;
            rx >>= modded_shift;
            const shift_amt = if binop_dtype_b == int then max_bits - modded_shift else max_bits:uint - modded_shift;
            top_bits <<= shift_amt;
            rx += top_bits;
            rx &= local_max_size;
          }
        }
        otherwise return MsgTuple.error("unknown bitwise binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result, max_bits));
    }

    // special error message for bitwise ops with real-valued arrays
    proc bitwiseOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where isRealType(binop_dtype_a) || isRealType(binop_dtype_b)
        do return MsgTuple.error("bitwise operations with real-valued arrays are not supported");

    /*
      Supports real division between two arrays
    */
    @arkouda.instantiateAndRegister
    proc divOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd);

      if a.tupShape != b.tupShape then
        return MsgTuple.error("array shapes must match for element-wise binary operations");

      const result = a.a:real / b.a:real;

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bigint-bigint division
    proc divOpVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a == bigint && binop_dtype_b == bigint
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            (has_max_bits, max_size, max_bits) = getMaxBits(a, b);

      if a.tupShape != b.tupShape then
        return MsgTuple.error("array shapes must match for element-wise binary operations");

      var result = a.a;
      ref ba = b.a;

      forall (rx, bx) in zip(result, ba) with (const local_max_size = max_size) {
        rx /= bx;
        if has_max_bits then rx &= local_max_size;
      }

      return st.insert(new shared SymEntry(result, max_bits));
    }

    /*
      Supports the following binary operations between an array and scalar:
      +, -, *, %, **, //
    */
    @arkouda.instantiateAndRegister
    proc arithmeticOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a != bool || binop_dtype_b != bool) &&
            binop_dtype_a != bigint && binop_dtype_b != bigint
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b),
            op = msgArgs['op'].toScalar(string);

      type resultType = promotedType(binop_dtype_a, binop_dtype_b);
      var result = makeDistArray((...a.tupShape), resultType);

      select op {
        when '+' do result = a.a:resultType + val:resultType;
        when '-' do result = a.a:resultType - val:resultType;
        when '*' do result = a.a:resultType * val:resultType;
        when '%' { // '
          ref aa = a.a;
          if isRealType(resultType)
            then [(ai,ri) in zip(aa,result)] ri = modHelper(ai:resultType, val:resultType);
            else [(ai,ri) in zip(aa,result)] ri = if val != 0:binop_dtype_b then ai:resultType % val:resultType else 0:resultType;
        }
        when '**' {
          if isIntegralType(binop_dtype_a) && (|| reduce (a.a < 0)) {
            return MsgTuple.error("Attempt to exponentiate integer base to negative exponent");
          }
          result = a.a:resultType ** val:resultType;
        }
        when '//' {
          ref aa = a.a;
          if isRealType(resultType)
            then [(ai,ri) in zip(aa,result)] ri = floorDivisionHelper(ai:resultType, val:resultType);
            else [(ai,ri) in zip(aa,result)] ri = if val != 0:binop_dtype_b then (ai / val):resultType else 0:resultType;
        }
        otherwise return MsgTuple.error("unknown arithmetic binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bool-bool arithmetic
    proc arithmeticOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a == bool && binop_dtype_b == bool
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(bool, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b),
            op = msgArgs['op'].toScalar(string);

      var result = makeDistArray((...a.tupShape), bool);

      // TODO: implement %, **, and // following NumPy's behavior
      select op {
        when '+' do result = a.a | val;
        when '-' do return MsgTuple.error("cannot subtract boolean from boolean array");
        when '*' do result = a.a & val;
        when '%' do return MsgTuple.error("modulo of boolean array by a boolean is not supported"); // '
        when '**' do return MsgTuple.error("exponentiation of a boolean array by a boolean is not supported");
        when '//' do return MsgTuple.error("floor-division of a boolean array by a boolean is not supported");
        otherwise return MsgTuple.error("unknown arithmetic binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bigint arithmetic
    proc arithmeticOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint && !isRealType(binop_dtype_b)) ||
            (binop_dtype_b == bigint && !isRealType(binop_dtype_a))
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b),
            op = msgArgs['op'].toScalar(string),
            (has_max_bits, max_size, max_bits) = getMaxBits(a);

      param bigintBoolOp = binop_dtype_a == bigint && binop_dtype_b == bool ||
                           binop_dtype_b == bigint && binop_dtype_a == bool;

      var result = makeDistArray((...a.tupShape), bigint);
      result = a.a:bigint;

      select op {
        when '+' {
          forall rx in result with (const local_val = val, const local_max_size = max_size) {
            rx += local_val;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '-' {
          forall rx in result with (const local_val = val, const local_max_size = max_size) {
            rx -= local_val;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '*' {
          forall rx in result with (const local_val = val, const local_max_size = max_size) {
            rx *= local_val;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '%' { // '
          if bigintBoolOp then return MsgTuple.error("'%' between bigint and bool values is not supported");
          forall rx in result with (const local_val = val, const local_max_size = max_size) {
            if local_val != 0 then mod(rx, rx, local_val);
            else rx = 0;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '**' {
          if bigintBoolOp then return MsgTuple.error("'**' between bigint and bool values is not supported");

          forall rx in result with (const local_val = val, const local_max_size = max_size) do
            powMod(rx, rx, local_val, local_max_size + 1);
        }
        when '//' {
          if bigintBoolOp then return MsgTuple.error("'//' between bigint and bool values is not supported");

          forall rx in result with (const local_val = val, const local_max_size = max_size) {
            if local_val != 0 then rx /= local_val;
            else rx = 0;
            if has_max_bits then rx &= local_max_size;
          }
        }
      }

      return st.insert(new shared SymEntry(result, max_bits));
    }

    proc arithmeticOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint && isRealType(binop_dtype_b)) ||
            (binop_dtype_b == bigint && isRealType(binop_dtype_a))
    {
      return MsgTuple.error("binary arithmetic operations between real and bigint arrays/values are not supported");
    }

    /*
      Supports the following binary operations between an array and scalar:
      ==, !=, <, <=, >, >=
    */
    @arkouda.instantiateAndRegister
    proc comparisonOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where !(binop_dtype_a == bigint && isRealType(binop_dtype_b)) &&
            !(binop_dtype_b == bigint && isRealType(binop_dtype_a))
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b),
            op = msgArgs['op'].toScalar(string);

      var result = makeDistArray((...a.tupShape), bool);

      if (binop_dtype_a == real && binop_dtype_b == bool) ||
         (binop_dtype_a == bool && binop_dtype_b == real)
      {
        select op {
          when '==' do result = a.a:real == val:real;
          when '!=' do result = a.a:real != val:real;
          when '<'  do result = a.a:real <  val:real;
          when '<=' do result = a.a:real <= val:real;
          when '>'  do result = a.a:real >  val:real;
          when '>=' do result = a.a:real >= val:real;
          otherwise return MsgTuple.error("unknown comparison binary operation: " + op);
        }
      } else {
        select op {
          when '==' do result = a.a == val;
          when '!=' do result = a.a != val;
          when '<'  do result = a.a <  val;
          when '<=' do result = a.a <= val;
          when '>'  do result = a.a >  val;
          when '>=' do result = a.a >= val;
          otherwise return MsgTuple.error("unknown comparison binary operation: " + op);
        }
      }

      return st.insert(new shared SymEntry(result));
    }

    proc comparisonOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint && isRealType(binop_dtype_b)) ||
            (binop_dtype_b == bigint && isRealType(binop_dtype_a))
    {
      return MsgTuple.error("comparison operations between real and bigint arrays/values are not supported");
    }

    /*
      Supports the following binary operations between an array and scalar
      |, &, ^, <<, >>
    */
    @arkouda.instantiateAndRegister
    proc bitwiseOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a != bigint && binop_dtype_b != bigint &&
            !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b) &&
            !(binop_dtype_a == bool && binop_dtype_b == bool)
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b),
            op = msgArgs['op'].toScalar(string);

      type resultType = if (isIntType(binop_dtype_a) && isUintType(binop_dtype_b)) ||
                           (isUintType(binop_dtype_a) && isIntType(binop_dtype_b))
        then binop_dtype_a // use LHS type for integer types with non-matching signed-ness
        else promotedType(binop_dtype_a, binop_dtype_b); // otherwise, use regular type promotion
      var result = makeDistArray((...a.tupShape), resultType);

      select op {
        when '|'  do result = (a.a | val):resultType;
        when '&'  do result = (a.a & val):resultType;
        when '^'  do result = (a.a ^ val):resultType;
        when '<<' {
          ref aa = a.a;
          [(ri,ai) in zip(result,aa)] if 0 <= val && val < 64 then ri = ai:resultType << val:resultType;
        }
        when '>>' {
          ref aa = a.a;
          [(ri,ai) in zip(result,aa)] if 0 <= val && val < 64 then ri = ai:resultType >> val:resultType;
        }
        when '<<<' {
          if !isIntegral(binop_dtype_a) || !isIntegral(binop_dtype_b)
            then return MsgTuple.error("cannot perform bitwise rotation with boolean arrays");
          result = rotl(a.a, val):resultType;
        }
        when '>>>' {
          if !isIntegral(binop_dtype_a) || !isIntegral(binop_dtype_b)
            then return MsgTuple.error("cannot perform bitwise rotation with boolean arrays");
          result = rotr(a.a, val):resultType;
        }
        otherwise return MsgTuple.error("unknown bitwise binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bitwise ops with two boolean arrays
    proc bitwiseOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a == bool && binop_dtype_b == bool
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b),
            op = msgArgs['op'].toScalar(string);

      if op == '<<' || op == '>>' {
        var result = makeDistArray((...a.tupShape), int);

        if op == '<<' {
          ref aa = a.a;
          [(ri,ai) in zip(result,aa)] if 0 <= val && val < 64 then ri = ai:int << val:int;
        } else {
          ref aa = a.a;
          [(ri,ai) in zip(result,aa)] if 0 <= val && val < 64 then ri = ai:int >> val:int;
        }

        return st.insert(new shared SymEntry(result));
      } else {
        var result = makeDistArray((...a.tupShape), bool);

        select op {
          when '|'  do result = a.a | val;
          when '&'  do result = a.a & val;
          when '^'  do result = a.a ^ val;
          when '<<<' do return MsgTuple.error("bitwise rotation on boolean arrays is not supported");
          when '>>>' do return MsgTuple.error("bitwise rotation on boolean arrays is not supported");
          otherwise return MsgTuple.error("unknown bitwise binary operation: " + op);
        }

        return st.insert(new shared SymEntry(result));
      }
    }

    // special handling for bitwise ops with at least one bigint array/value
    proc bitwiseOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
            !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b)
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b),
            op = msgArgs['op'].toScalar(string),
            (has_max_bits, max_size, max_bits) = getMaxBits(a);

      // 'a' must be a bigint, and 'b' must not be real for the operations below
      // (for some operations, both a and b must be bigint)
      if binop_dtype_a != bigint || isRealType(binop_dtype_b) then
        return MsgTuple.error("bitwise operations between a LHS bigint and RHS non-bigint arrays are not supported");

      var result = makeDistArray((...a.tupShape), bigint);
      result = a.a:bigint;
      param bothBigint = binop_dtype_a == bigint && binop_dtype_b == bigint;

      select op {
        when '|' {
          if !bothBigint then return MsgTuple.error("bitwise OR between bigint and non-bigint arrays/values is not supported");
          forall rx in result with (var local_val = val, const local_max_size = max_size) {
            rx |= local_val;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '&' {
          if !bothBigint then return MsgTuple.error("bitwise AND between bigint and non-bigint arrays/values is not supported");
          forall rx in result with (var local_val = val, const local_max_size = max_size) {
            rx &= local_val;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '^' {
          if !bothBigint then return MsgTuple.error("bitwise XOR between bigint and non-bigint arrays/values is not supported");
          forall rx in result with (var local_val = val, const local_max_size = max_size) {
            rx ^= local_val;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '<<' {
          if binop_dtype_b == bigint then return MsgTuple.error("left-shift of a bigint array by a non-bigint value is not supported");
          forall rx in result with (var local_val = val, const local_max_size = max_size) {
            rx <<= local_val;
            if has_max_bits {
              if local_val >= max_bits then rx = 0;
              else {
                rx <<= local_val;
                rx &= local_max_size;
              }
            }
          }
        }
        when '>>' {
          if binop_dtype_b == bigint then return MsgTuple.error("right-shift of a bigint array by a non-bigint value is not supported");
          forall rx in result with (var local_val = val, const local_max_size = max_size) {
            rx >>= local_val;
            if has_max_bits {
              if local_val >= max_bits then rx = 0;
              else {
                rx >>= local_val;
                rx &= local_max_size;
              }
            }
          }
        }
        when '<<<' {
          if !has_max_bits then return MsgTuple.error("bitwise rotation on bigint arrays requires a max_bits value");
          if binop_dtype_b == bigint then return MsgTuple.error("right-shift of a bigint array by a non-bigint array is not supported");
          forall rx in result with (var local_val = val, var local_max_size = max_size) {
            var bot_bits = rx;
            const modded_shift = if binop_dtype_b == int then local_val%max_bits else local_val%max_bits:uint;
            rx <<= modded_shift;
            const shift_amt = if binop_dtype_b == int then max_bits - modded_shift else max_bits:uint - modded_shift;
            bot_bits >>= shift_amt;
            rx += bot_bits;
            rx &= local_max_size;
          }
        }
        when '>>>' {
          if !has_max_bits then return MsgTuple.error("bitwise rotation on bigint arrays requires a max_bits value");
          if binop_dtype_b == bigint then return MsgTuple.error("right-shift of a bigint array by a non-bigint array is not supported");
          forall rx in result with (var local_val = val, var local_max_size = max_size) {
            var top_bits = rx;
            const modded_shift = if binop_dtype_b == int then local_val%max_bits else local_val%max_bits:uint;
            rx >>= modded_shift;
            const shift_amt = if binop_dtype_b == int then max_bits - modded_shift else max_bits:uint - modded_shift;
            top_bits <<= shift_amt;
            rx += top_bits;
            rx &= local_max_size;
          }
        }
        otherwise return MsgTuple.error("unknown bitwise binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result, max_bits));
    }

    // special error message for bitwise ops with real-valued arrays
    proc bitwiseOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where isRealType(binop_dtype_a) || isRealType(binop_dtype_b)
        do return MsgTuple.error("bitwise operations with real-valued arrays are not supported");

    /*
      Supports real division between an array and a scalar
    */
    @arkouda.instantiateAndRegister
    proc divOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b);

      const result = a.a:real / val:real;

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bigint-bigint division
    proc divOpVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a == bigint && binop_dtype_b == bigint
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(bigint),
            (has_max_bits, max_size, max_bits) = getMaxBits(a);

      var result = a.a;

      forall rx in result with (const local_val = val, const local_max_size = max_size) {
        rx /= local_val;
        if has_max_bits then rx &= local_max_size;
      }

      return st.insert(new shared SymEntry(result, max_bits));
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
    // @arkouda.registerND
    // proc binopvsMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
    //     param pn = Reflection.getRoutineName();
    //     var repMsg: string = ""; // response message

    //     const aname = msgArgs.getValueOf("a");
    //     const op = msgArgs.getValueOf("op");
    //     const value = msgArgs.get("value");

    //     const dtype = str2dtype(msgArgs.getValueOf("dtype"));
    //     var rname = st.nextName();
    //     var left: borrowed GenSymEntry = getGenericTypedArrayEntry(aname, st);

    //     omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
    //                        "op: %s dtype: %? pdarray: %? scalar: %?".format(
    //                                                  op,dtype,st.attrib(aname),value.getValue()));

    //     use Set;
    //     // This boolOps set is a filter to determine the output type for the operation.
    //     // All operations that involve one of these operations result in a `bool` symbol
    //     // table entry.
    //     var boolOps: set(string);
    //     boolOps.add("<");
    //     boolOps.add("<=");
    //     boolOps.add(">");
    //     boolOps.add(">=");
    //     boolOps.add("==");
    //     boolOps.add("!=");

    //     var realOps: set(string);
    //     realOps.add("+");
    //     realOps.add("-");
    //     realOps.add("/");
    //     realOps.add("//");

    //     select (left.dtype, dtype) {
    //       when (DType.Int64, DType.Int64) {
    //         var l = toSymEntry(left,int, nd);
    //         var val = value.getIntValue();
            
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         } else if op == "/" {
    //           // True division is the only case in this int, int case
    //           // that results in a `real` symbol table entry.
    //           var e = st.addEntry(rname, l.tupShape, real);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, int);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.Int64, DType.Float64) {
    //         var l = toSymEntry(left,int, nd);
    //         var val = value.getRealValue();
    //         // Only two possible resultant types are `bool` and `real`
    //         // for this case
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, real);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.Float64, DType.Int64) {
    //         var l = toSymEntry(left,real, nd);
    //         var val = value.getIntValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, real);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.UInt64, DType.Float64) {
    //         var l = toSymEntry(left,uint, nd);
    //         var val = value.getRealValue();
    //         // Only two possible resultant types are `bool` and `real`
    //         // for this case
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, real);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.Float64, DType.UInt64) {
    //         var l = toSymEntry(left,real, nd);
    //         var val = value.getUIntValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, real);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.Float64, DType.Float64) {
    //         var l = toSymEntry(left,real, nd);
    //         var val = value.getRealValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, real);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       // For cases where a boolean operand is involved, the only
    //       // possible resultant type is `bool`
    //       when (DType.Bool, DType.Bool) {
    //         var l = toSymEntry(left,bool, nd);
    //         var val = value.getBoolValue();
    //         if (op == "<<") || (op == ">>") {
    //           var e = st.addEntry(rname, l.tupShape, int);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, bool);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.Bool, DType.Int64) {
    //         var l = toSymEntry(left,bool, nd);
    //         var val = value.getIntValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, int);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.Int64, DType.Bool) {
    //         var l = toSymEntry(left,int, nd);
    //         var val = value.getBoolValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, int);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.Bool, DType.Float64) {
    //         var l = toSymEntry(left,bool, nd);
    //         var val = value.getRealValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, real);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.Float64, DType.Bool) {
    //         var l = toSymEntry(left,real, nd);
    //         var val = value.getBoolValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, real);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.Bool, DType.UInt64) {
    //         var l = toSymEntry(left,bool, nd);
    //         var val = value.getUIntValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, uint);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.UInt64, DType.Bool) {
    //         var l = toSymEntry(left,uint, nd);
    //         var val = value.getBoolValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         var e = st.addEntry(rname, l.tupShape, uint);
    //         return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //       }
    //       when (DType.UInt64, DType.UInt64) {
    //         var l = toSymEntry(left,uint, nd);
    //         var val = value.getUIntValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         if op == "/"{
    //           var e = st.addEntry(rname, l.tupShape, real);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         } else {
    //           var e = st.addEntry(rname, l.tupShape, uint);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //       }
    //       when (DType.UInt64, DType.Int64) {
    //         var l = toSymEntry(left,uint, nd);
    //         var val = value.getIntValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         // +, -, /, // both result in real outputs to match NumPy
    //         if realOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, real);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         } else {
    //           // isn't +, -, /, // so we can use LHS to determine type
    //           var e = st.addEntry(rname, l.tupShape, uint);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //       }
    //       when (DType.Int64, DType.UInt64) {
    //         var l = toSymEntry(left,int, nd);
    //         var val = value.getUIntValue();
    //         if boolOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, bool);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //         // +, -, /, // both result in real outputs to match NumPy
    //         if realOps.contains(op) {
    //           var e = st.addEntry(rname, l.tupShape, real);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         } else {
    //           // isn't +, -, /, // so we can use LHS to determine type
    //           var e = st.addEntry(rname, l.tupShape, int);
    //           return doBinOpvs(l, val, e, op, dtype, rname, pn, st);
    //         }
    //       }
    //       when (DType.BigInt, DType.BigInt) {
    //         var l = toSymEntry(left,bigint, nd);
    //         var val = value.getBigIntValue();
    //         if boolOps.contains(op) {
    //           // call bigint specific func which returns distr bool array
    //           var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
    //           var repMsg = "created %s".format(st.attrib(rname));
    //           return new MsgTuple(repMsg, MsgType.NORMAL);
    //         }
    //         // call bigint specific func which returns dist bigint array
    //         var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
    //         var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
    //         var repMsg = "created %s".format(st.attrib(rname));
    //         return new MsgTuple(repMsg, MsgType.NORMAL);
    //       }
    //       when (DType.BigInt, DType.Int64) {
    //         var l = toSymEntry(left,bigint, nd);
    //         var val = value.getIntValue();
    //         if boolOps.contains(op) {
    //           // call bigint specific func which returns distr bool array
    //           var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
    //           var repMsg = "created %s".format(st.attrib(rname));
    //           return new MsgTuple(repMsg, MsgType.NORMAL);
    //         }
    //         // call bigint specific func which returns dist bigint array
    //         var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
    //         var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
    //         var repMsg = "created %s".format(st.attrib(rname));
    //         return new MsgTuple(repMsg, MsgType.NORMAL);
    //       }
    //       when (DType.BigInt, DType.UInt64) {
    //         var l = toSymEntry(left,bigint, nd);
    //         var val = value.getUIntValue();
    //         if boolOps.contains(op) {
    //           // call bigint specific func which returns distr bool array
    //           var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
    //           var repMsg = "created %s".format(st.attrib(rname));
    //           return new MsgTuple(repMsg, MsgType.NORMAL);
    //         }
    //         // call bigint specific func which returns dist bigint array
    //         var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
    //         var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
    //         var repMsg = "created %s".format(st.attrib(rname));
    //         return new MsgTuple(repMsg, MsgType.NORMAL);
    //       }
    //       when (DType.BigInt, DType.Bool) {
    //         var l = toSymEntry(left,bigint, nd);
    //         var val = value.getBoolValue();
    //         if boolOps.contains(op) {
    //           // call bigint specific func which returns distr bool array
    //           var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
    //           var repMsg = "created %s".format(st.attrib(rname));
    //           return new MsgTuple(repMsg, MsgType.NORMAL);
    //         }
    //         // call bigint specific func which returns dist bigint array
    //         var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
    //         var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
    //         var repMsg = "created %s".format(st.attrib(rname));
    //         return new MsgTuple(repMsg, MsgType.NORMAL);
    //       }
    //       when (DType.Int64, DType.BigInt) {
    //         var l = toSymEntry(left,int, nd);
    //         var val = value.getBigIntValue();
    //         if boolOps.contains(op) {
    //           // call bigint specific func which returns distr bool array
    //           var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
    //           var repMsg = "created %s".format(st.attrib(rname));
    //           return new MsgTuple(repMsg, MsgType.NORMAL);
    //         }
    //         // call bigint specific func which returns dist bigint array
    //         var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
    //         var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
    //         var repMsg = "created %s".format(st.attrib(rname));
    //         return new MsgTuple(repMsg, MsgType.NORMAL);
    //       }
    //       when (DType.UInt64, DType.BigInt) {
    //         var l = toSymEntry(left,uint, nd);
    //         var val = value.getBigIntValue();
    //         if boolOps.contains(op) {
    //           // call bigint specific func which returns distr bool array
    //           var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
    //           var repMsg = "created %s".format(st.attrib(rname));
    //           return new MsgTuple(repMsg, MsgType.NORMAL);
    //         }
    //         // call bigint specific func which returns dist bigint array
    //         var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
    //         var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
    //         var repMsg = "created %s".format(st.attrib(rname));
    //         return new MsgTuple(repMsg, MsgType.NORMAL);
    //       }
    //       when (DType.Bool, DType.BigInt) {
    //         var l = toSymEntry(left,bool, nd);
    //         var val = value.getBigIntValue();
    //         if boolOps.contains(op) {
    //           // call bigint specific func which returns distr bool array
    //           var e = st.addEntry(rname, createSymEntry(doBigIntBinOpvsBoolReturn(l, val, op)));
    //           var repMsg = "created %s".format(st.attrib(rname));
    //           return new MsgTuple(repMsg, MsgType.NORMAL);
    //         }
    //         // call bigint specific func which returns dist bigint array
    //         var (tmp, max_bits) = doBigIntBinOpvs(l, val, op);
    //         var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
    //         var repMsg = "created %s".format(st.attrib(rname));
    //         return new MsgTuple(repMsg, MsgType.NORMAL);
    //       }
    //     }
    //     var errorMsg = unrecognizedTypeError(pn, "("+dtype2str(left.dtype)+","+dtype2str(dtype)+")");
    //     omLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
    //     return new MsgTuple(errorMsg, MsgType.ERROR);
    // }

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
                 "command = %? op = %? scalar dtype = %? scalar = %? pdarray = %?".format(
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
              var repMsg = "created %s".format(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".format(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.Int64) {
            var val = value.getBigIntValue();
            var r = toSymEntry(right,int, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".format(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".format(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.UInt64) {
            var val = value.getBigIntValue();
            var r = toSymEntry(right,uint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".format(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".format(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.BigInt, DType.Bool) {
            var val = value.getBigIntValue();
            var r = toSymEntry(right,bool, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".format(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".format(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.Int64, DType.BigInt) {
            var val = value.getIntValue();
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".format(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".format(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.UInt64, DType.BigInt) {
            var val = value.getUIntValue();
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".format(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".format(st.attrib(rname));
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when (DType.Bool, DType.BigInt) {
            var val = value.getBoolValue();
            var r = toSymEntry(right,bigint, nd);
            if boolOps.contains(op) {
              // call bigint specific func which returns distr bool array
              var e = st.addEntry(rname, createSymEntry(doBigIntBinOpsvBoolReturn(val, r, op)));
              var repMsg = "created %s".format(st.attrib(rname));
              return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            // call bigint specific func which returns dist bigint array
            var (tmp, max_bits) = doBigIntBinOpsv(val, r, op);
            var e = st.addEntry(rname, createSymEntry(tmp, max_bits));
            var repMsg = "created %s".format(st.attrib(rname));
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
                    "cmd: %s op: %s left pdarray: %s right pdarray: %s".format(cmd,op,
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
                    when "+=" {l.a |= r.a;}
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
                        "cmd: %s op: %s aname: %s dtype: %s scalar: %s".format(
                                                 cmd,op,aname,dtype2str(dtype),value.getValue()));

        var left: borrowed GenSymEntry = getGenericTypedArrayEntry(aname, st);
 
        omLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "op: %? pdarray: %? scalar: %?".format(op,st.attrib(aname),value.getValue()));
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
            when (DType.Bool, DType.Bool) {
                var l = toSymEntry(left, bool, nd);
                var val = value.getBoolValue();
                select op {
                    when "+=" {l.a |= val;}
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


    /*
      Helper to determine the max_bits between two SymEntry's

      returns a three tuple with:
      * bool: whether at least on SymEntry has a max_bits setting
      * bigint: the maximum size
      * max_bits: the maximum of the two array's max_bits values
    */
    proc getMaxBits(a: borrowed SymEntry(?), b: borrowed SymEntry(?)): (bool, bigint, int) {
      return _getMaxBits(max(a.max_bits, b.max_bits));
    }

    proc getMaxBits(a: borrowed SymEntry(?)): (bool, bigint, int) {
      return _getMaxBits(a.max_bits);
    }

    proc _getMaxBits(max_bits: int): (bool, bigint, int) {
      const has_max_bits = max_bits != -1;

      var max_size = 1:bigint;
      if has_max_bits {
        max_size <<= max_bits;
        max_size -= 1;
      }

      return (has_max_bits, max_size, max_bits);
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
}
