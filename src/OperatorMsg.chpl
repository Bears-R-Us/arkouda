
module OperatorMsg
{
    use ServerConfig;

    use Math;
    use BitOps;
    use Reflection;
    use ServerErrors;
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
      |, &, ^, <<, >>, <<<, >>>
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
      |, &, ^, <<, >>, <<<, >>>
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
      Supports the following binary operations between an array and scalar:
      +, -, *, %, **, //
    */
    @arkouda.instantiateAndRegister
    proc arithmeticOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a != bool || binop_dtype_b != bool) &&
            binop_dtype_a != bigint && binop_dtype_b != bigint
    {
      const val = msgArgs['value'].toScalar(binop_dtype_a),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      type resultType = promotedType(binop_dtype_a, binop_dtype_b);
      var result = makeDistArray((...b.tupShape), resultType);

      select op {
        when '+' do result = val:resultType + b.a:resultType;
        when '-' do result = val:resultType - b.a:resultType;
        when '*' do result = val:resultType * b.a:resultType;
        when '%' { // '
          ref bb = b.a;
          if isRealType(resultType)
            then [(bi,ri) in zip(bb,result)] ri = modHelper(val:resultType, bi:resultType);
            else [(bi,ri) in zip(bb,result)] ri = if val != 0:binop_dtype_a then val:resultType % bi:resultType else 0:resultType;
        }
        when '**' {
          if isIntegralType(binop_dtype_b) && (|| reduce (b.a < 0)) {
            return MsgTuple.error("Attempt to exponentiate integer base to negative exponent");
          }
          result = val:resultType ** b.a:resultType;
        }
        when '//' {
          ref bb = b.a;
          if isRealType(resultType)
            then [(bi,ri) in zip(bb,result)] ri = floorDivisionHelper(val:resultType, bi:resultType);
            else [(bi,ri) in zip(bb,result)] ri = if val != 0:binop_dtype_a then (val / bi):resultType else 0:resultType;
        }
        otherwise return MsgTuple.error("unknown arithmetic binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bool-bool arithmetic
    proc arithmeticOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a == bool && binop_dtype_b == bool
    {
      const val = msgArgs['value'].toScalar(binop_dtype_a),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      var result = makeDistArray((...b.tupShape), bool);

      // TODO: implement %, **, and // following NumPy's behavior
      select op {
        when '+' do result = val | b.a;
        when '-' do return MsgTuple.error("cannot subtract boolean array from boolean");
        when '*' do result = val & b.a;
        when '%' do return MsgTuple.error("modulo of boolean by a boolean array is not supported"); // '
        when '**' do return MsgTuple.error("exponentiation of a boolean by a boolean array is not supported");
        when '//' do return MsgTuple.error("floor-division of a boolean by a boolean array is not supported");
        otherwise return MsgTuple.error("unknown arithmetic binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bigint arithmetic
    proc arithmeticOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint && !isRealType(binop_dtype_b)) ||
            (binop_dtype_b == bigint && !isRealType(binop_dtype_a))
    {
      const val = msgArgs['value'].toScalar(binop_dtype_a),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string),
            (has_max_bits, max_size, max_bits) = getMaxBits(b);

      var result = makeDistArray((...b.tupShape), bigint);
      result = val:bigint;

      ref bb = b.a;

      select op {
        when '+' {
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            rx += bi;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '-' {
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            rx -= bi;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '*' {
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            rx *= bi;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '%' { // '
          if binop_dtype_a != bigint then return MsgTuple.error("cannot mod a non-bigint value by a bigint array");
          if binop_dtype_b == bool then return MsgTuple.error("cannot mod a bigint value by a bool array");
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            if bi != 0 then mod(rx, rx, bi); else rx = 0:bigint;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '**' {
          if binop_dtype_a != bigint then return MsgTuple.error("cannot exponentiate a non-bigint value by a bigint array");
          if binop_dtype_b == bool then return MsgTuple.error("cannot exponentiate a bigint value by a bool array");
          if || reduce (bb < 0) then return MsgTuple.error("Attempt to exponentiate bigint base to negative exponent");

          if has_max_bits
            then forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) do
              powMod(rx, rx, bi, local_max_size + 1);
            else forall (rx, bi) in zip(result, bb) do
              rx **= bi:uint;
        }
        when '//' {
          if binop_dtype_a != bigint then return MsgTuple.error("cannot floor-div a non-bigint value by a bigint array");
          if binop_dtype_b == bool then return MsgTuple.error("cannot floor-div a bigint value by a bool array");

          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            if bi != 0 then rx /= bi; else rx = 0:bigint;
            if has_max_bits then rx &= local_max_size;
          }
        }
      }

      return st.insert(new shared SymEntry(result, max_bits));
    }

    proc arithmeticOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
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
    proc comparisonOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where !(binop_dtype_a == bigint && isRealType(binop_dtype_b)) &&
            !(binop_dtype_b == bigint && isRealType(binop_dtype_a))
    {
      const val = msgArgs['value'].toScalar(binop_dtype_a),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      var result = makeDistArray((...b.tupShape), bool);

      if (binop_dtype_a == real && binop_dtype_b == bool) ||
         (binop_dtype_a == bool && binop_dtype_b == real)
      {
        select op {
          when '==' do result = val:real == b.a:real;
          when '!=' do result = val:real != b.a:real;
          when '<'  do result = val:real <  b.a:real;
          when '<=' do result = val:real <= b.a:real;
          when '>'  do result = val:real >  b.a:real;
          when '>=' do result = val:real >= b.a:real;
          otherwise return MsgTuple.error("unknown comparison binary operation: " + op);
        }
      } else {
        select op {
          when '==' do result = val == b.a;
          when '!=' do result = val != b.a;
          when '<'  do result = val <  b.a;
          when '<=' do result = val <= b.a;
          when '>'  do result = val >  b.a;
          when '>=' do result = val >= b.a;
          otherwise return MsgTuple.error("unknown comparison binary operation: " + op);
        }
      }

      return st.insert(new shared SymEntry(result));
    }

    proc comparisonOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
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
      |, &, ^, <<, >>, <<<, >>>
    */
    @arkouda.instantiateAndRegister
    proc bitwiseOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a != bigint && binop_dtype_b != bigint &&
            !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b) &&
            !(binop_dtype_a == bool && binop_dtype_b == bool)
    {
      const val = msgArgs['value'].toScalar(binop_dtype_a),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      type resultType = if (isIntType(binop_dtype_a) && isUintType(binop_dtype_b)) ||
                           (isUintType(binop_dtype_a) && isIntType(binop_dtype_b))
        then binop_dtype_a // use LHS type for integer types with non-matching signed-ness
        else promotedType(binop_dtype_a, binop_dtype_b); // otherwise, use regular type promotion
      var result = makeDistArray((...b.tupShape), resultType);

      select op {
        when '|'  do result = (val | b.a):resultType;
        when '&'  do result = (val & b.a):resultType;
        when '^'  do result = (val ^ b.a):resultType;
        when '<<' {
          ref bb = b.a;
          [(ri,bi) in zip(result,bb)] if 0 <= bi && bi < 64 then ri = val:resultType << bi:resultType;
        }
        when '>>' {
          ref bb = b.a;
          [(ri,bi) in zip(result,bb)] if 0 <= bi && bi < 64 then ri = val:resultType >> bi:resultType;
        }
        when '<<<' {
          if !isIntegral(binop_dtype_a) || !isIntegral(binop_dtype_b)
            then return MsgTuple.error("cannot perform bitwise rotation with boolean arrays");
          result = rotl(val, b.a):resultType;
        }
        when '>>>' {
          if !isIntegral(binop_dtype_a) || !isIntegral(binop_dtype_b)
            then return MsgTuple.error("cannot perform bitwise rotation with boolean arrays");
          result = rotr(val, b.a):resultType;
        }
        otherwise return MsgTuple.error("unknown bitwise binary operation: " + op);
      }

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bitwise ops with two boolean arrays
    proc bitwiseOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a == bool && binop_dtype_b == bool
    {
      const val = msgArgs['value'].toScalar(binop_dtype_a),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      if op == '<<' || op == '>>' {
        var result = makeDistArray((...b.tupShape), int);

        if op == '<<' {
          ref bb = b.a;
          [(ri,bi) in zip(result,bb)] if 0 <= bi && bi < 64 then ri = val:int << bi:int;
        } else {
          ref bb = b.a;
          [(ri,bi) in zip(result,bb)] if 0 <= bi && bi < 64 then ri = val:int >> bi:int;
        }

        return st.insert(new shared SymEntry(result));
      } else {
        var result = makeDistArray((...b.tupShape), bool);

        select op {
          when '|'  do result = val | b.a;
          when '&'  do result = val & b.a;
          when '^'  do result = val ^ b.a;
          when '<<<' do return MsgTuple.error("bitwise rotation on boolean arrays is not supported");
          when '>>>' do return MsgTuple.error("bitwise rotation on boolean arrays is not supported");
          otherwise return MsgTuple.error("unknown bitwise binary operation: " + op);
        }

        return st.insert(new shared SymEntry(result));
      }
    }

    // special handling for bitwise ops with at least one bigint array/value
    proc bitwiseOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
            !isRealType(binop_dtype_a) && !isRealType(binop_dtype_b)
    {
      const val = msgArgs['value'].toScalar(binop_dtype_a),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string),
            (has_max_bits, max_size, max_bits) = getMaxBits(b);

      // 'a' must be a bigint, and 'b' must not be real for the operations below
      // (for some operations, both a and b must be bigint)
      if binop_dtype_a != bigint || isRealType(binop_dtype_b) then
        return MsgTuple.error("bitwise operations between a LHS bigint and RHS non-bigint arrays are not supported");

      var result = makeDistArray((...b.tupShape), bigint);
      result = val:bigint;
      param bothBigint = binop_dtype_a == bigint && binop_dtype_b == bigint;
      ref bb = b.a;

      select op {
        when '|' {
          if !bothBigint then return MsgTuple.error("bitwise OR between bigint and non-bigint arrays/values is not supported");
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            rx |= bi;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '&' {
          if !bothBigint then return MsgTuple.error("bitwise AND between bigint and non-bigint arrays/values is not supported");
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            rx &= bi;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '^' {
          if !bothBigint then return MsgTuple.error("bitwise XOR between bigint and non-bigint arrays/values is not supported");
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            rx ^= bi;
            if has_max_bits then rx &= local_max_size;
          }
        }
        when '<<' {
          if binop_dtype_b == bigint then return MsgTuple.error("left-shift of a bigint array by a non-bigint value is not supported");
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            if has_max_bits {
              if bi >= max_bits then rx = 0:bigint;
              else {
                rx <<= bi;
                rx &= local_max_size;
              }
            } else {
              rx <<= bi;
            }
          }
        }
        when '>>' {
          if binop_dtype_b == bigint then return MsgTuple.error("right-shift of a bigint array by a non-bigint value is not supported");
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            if has_max_bits {
              if bi >= max_bits then rx = 0:bigint;
              else {
                rx >>= bi;
                rx &= local_max_size;
              }
            } else {
              rx >>= bi;
            }
          }
        }
        when '<<<' {
          if !has_max_bits then return MsgTuple.error("bitwise rotation on bigint arrays requires a max_bits value");
          if binop_dtype_a == bigint then return MsgTuple.error("right-shift of a bigint array by a non-bigint array is not supported");
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            var bot_bits = rx;
            const modded_shift = if binop_dtype_b == int then bi%max_bits else bi%max_bits:uint;
            rx <<= modded_shift;
            const shift_amt = if binop_dtype_b == int then max_bits - modded_shift else max_bits:uint - modded_shift;
            bot_bits >>= shift_amt;
            rx += bot_bits;
            rx &= local_max_size;
          }
        }
        when '>>>' {
          if !has_max_bits then return MsgTuple.error("bitwise rotation on bigint arrays requires a max_bits value");
          if binop_dtype_a == bigint then return MsgTuple.error("right-shift of a bigint array by a non-bigint array is not supported");
          forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
            var top_bits = rx;
            const modded_shift = if binop_dtype_b == int then bi%max_bits else bi%max_bits:uint;
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
    proc bitwiseOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where isRealType(binop_dtype_a) || isRealType(binop_dtype_b)
        do return MsgTuple.error("bitwise operations with real-valued arrays are not supported");

    /*
      Supports real division between a scalar and an array
    */
    @arkouda.instantiateAndRegister
    proc divOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
      const val = msgArgs['value'].toScalar(binop_dtype_a),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd);

      const result = val:real / b.a:real;

      return st.insert(new shared SymEntry(result));
    }

    // special handling for bigint-bigint division
    proc divOpSV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where binop_dtype_a == bigint && binop_dtype_b == bigint
    {
      const val = msgArgs['value'].toScalar(binop_dtype_a),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            (has_max_bits, max_size, max_bits) = getMaxBits(b);

      var result = makeDistArray((...b.tupShape), bigint);
      result = val:bigint;
      ref bb = b.a;

      forall (rx, bi) in zip(result, bb) with (const local_max_size = max_size) {
        rx /= bi;
        if has_max_bits then rx &= local_max_size;
      }

      return st.insert(new shared SymEntry(result, max_bits));
    }

    @arkouda.instantiateAndRegister
    proc opeqVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      // result of operation must be the same type as the left operand
      where binop_dtype_a == promotedType(binop_dtype_a, binop_dtype_b) &&
            binop_dtype_a != bigint && binop_dtype_b != bigint
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd),
            op = msgArgs['op'].toScalar(string);

      const unsupErrorMsg = "unsupported op=: %s %s %s".format(binop_dtype_a:string, op, binop_dtype_b:string);

      if isIntegralType(binop_dtype_a) && isIntegralType(binop_dtype_b) {
        select op {
          when "+=" do a.a += b.a;
          when "-=" do a.a -= b.a;
          when "*=" do a.a *= b.a;
          when ">>=" do a.a >>= b.a;
          when "<<=" do a.a <<= b.a;
          when "//=" {
            ref aa = a.a;
            ref bb = b.a;
            [(ai,bi) in zip(aa,bb)] ai = if bi != 0 then ai/bi else 0;
          }
          when "%=" {
            ref aa = a.a;
            ref bb = b.a;
            [(ai,bi) in zip(aa,bb)] ai = if bi != 0 then ai%bi else 0;
          }
          when "**=" {
            if || reduce (b.a<0)
              then return MsgTuple.error("Attempt to exponentiate base of type Int64 to negative exponent");
              else a.a **= b.a;
          }
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else if isIntegralType(binop_dtype_a) && binop_dtype_b == bool {
        select op {
          when "+=" do a.a += b.a:binop_dtype_a;
          when "-=" do a.a -= b.a:binop_dtype_a;
          when "*=" do a.a *= b.a:binop_dtype_a;
          when ">>=" do a.a >>= b.a:binop_dtype_a;
          when "<<=" do a.a <<= b.a:binop_dtype_a;
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else if binop_dtype_a == bool && binop_dtype_b == bool {
        select op {
          when "|=" do a.a |= b.a;
          when "&=" do a.a &= b.a;
          when "^=" do a.a ^= b.a;
          when "+=" do a.a |= b.a;
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else if isRealType(binop_dtype_a) && (isRealType(binop_dtype_b) || isIntegralType(binop_dtype_b)) {
        select op {
          when "+=" do a.a += b.a;
          when "-=" do a.a -= b.a;
          when "*=" do a.a *= b.a;
          when "/=" do a.a /= b.a:binop_dtype_a;
          when "//=" {
            ref aa = a.a;
            ref bb = b.a;
            [(ai,bi) in zip(aa,bb)] ai = floorDivisionHelper(ai, bi);
          }
          when "**=" do a.a **= b.a;
          when "%=" {
            ref aa = a.a;
            ref bb = b.a;
            [(ai,bi) in zip(aa,bb)] ai = modHelper(ai, bi);
          }
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else if isRealType(binop_dtype_a) && binop_dtype_b == bool {
        select op {
          when "+=" do a.a += b.a:binop_dtype_a;
          when "-=" do a.a -= b.a:binop_dtype_a;
          when "*=" do a.a *= b.a:binop_dtype_a;
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else {
        return MsgTuple.error(unsupErrorMsg);
      }

      return MsgTuple.success();
    }

    // special handling for bigint ops
    proc opeqVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
            !isRealType(binop_dtype_b)
    {
      const op = msgArgs['op'].toScalar(string),
            unsupErrorMsg = "unsupported op=: %s %s %s".format(binop_dtype_a:string, op, binop_dtype_b:string);

      if binop_dtype_a != bigint then return MsgTuple.error(unsupErrorMsg);

      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            b = st[msgArgs['b']]: borrowed SymEntry(binop_dtype_b, array_nd);

      const (has_max_bits, max_size, max_bits) =
        if binop_dtype_b == bigint then getMaxBits(a, b) else getMaxBits(a);

      ref aa = a.a;
      ref bb = b.a;

      select op {
        when "+=" {
          forall (ai, bi) in zip(aa, bb) with (var local_max_size = max_size) {
            ai += bi;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "-=" {
          forall (ai, bi) in zip(aa, bb) with (var local_max_size = max_size) {
            ai -= bi;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "*=" {
          forall (ai, bi) in zip(aa, bb) with (var local_max_size = max_size) {
            ai *= bi;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "//=" {
          if binop_dtype_b == bool then return MsgTuple.error(unsupErrorMsg);
          forall (ai, bi) in zip(aa, bb) with (var local_max_size = max_size) {
            if bi != 0 then ai /= bi; else ai = 0:bigint;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "%=" {
          if binop_dtype_b == bool then return MsgTuple.error(unsupErrorMsg);
          // we can't use ai %= bi because this can result in negatives
          forall (ai, bi) in zip(aa, bb) with (var local_max_size = max_size) {
            if bi != 0 then mod(ai, ai, bi); else ai = 0:bigint;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "**=" {
          if binop_dtype_b == bool then return MsgTuple.error(unsupErrorMsg);
          if || reduce (b.a<0) then return MsgTuple.error("Attempt to exponentiate base of type BigInt to negative exponent");
          if has_max_bits {
            forall (ai, bi) in zip(aa, bb) with (var local_max_size = max_size) {
              powMod(ai, ai, bi, local_max_size + 1);
            }
          } else {
            forall (ai, bi) in zip(aa, bb) {
              ai **= bi:uint;
            }
          }
        }
        otherwise do return MsgTuple.error(unsupErrorMsg);
      }

      return MsgTuple.success();
    }

    // error message when result could not be stored in 'binop_dtype_a'
    proc opeqVV(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
      const op = msgArgs['op'].toScalar(string);
      return MsgTuple.error(notImplementedError(
        cmd,
        whichDtype(binop_dtype_a),
        op,
        whichDtype(binop_dtype_b)
      ));
    }


    @arkouda.instantiateAndRegister
    proc opeqVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      // result of operation must be the same type as the left operand
      where binop_dtype_a == promotedType(binop_dtype_a, binop_dtype_b) &&
            binop_dtype_a != bigint && binop_dtype_b != bigint
    {
      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b),
            op = msgArgs['op'].toScalar(string);

      const unsupErrorMsg = "unsupported op=: %s %s %s".format(binop_dtype_a:string, op, binop_dtype_b:string);

      if isIntegralType(binop_dtype_a) && isIntegralType(binop_dtype_b) {
        select op {
          when "+=" do a.a += val;
          when "-=" do a.a -= val;
          when "*=" do a.a *= val;
          when ">>=" do a.a >>= val;
          when "<<=" do a.a <<= val;
          when "//=" {
            if val != 0 then a.a /= val; else a.a = 0;
          }
          when "%=" {
            if val != 0 then a.a %= val; else a.a = 0;
          }
          when "**=" {
            if val<0
              then return MsgTuple.error("Attempt to exponentiate base of type Int64 to negative exponent");
              else a.a **= val;
          }
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else if isIntegralType(binop_dtype_a) && binop_dtype_b == bool {
        select op {
          when "+=" do a.a += val:binop_dtype_a;
          when "-=" do a.a -= val:binop_dtype_a;
          when "*=" do a.a *= val:binop_dtype_a;
          when ">>=" do a.a >>= val:binop_dtype_a;
          when "<<=" do a.a <<= val:binop_dtype_a;
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else if binop_dtype_a == bool && binop_dtype_b == bool {
        select op {
          when "|=" do a.a |= val;
          when "&=" do a.a &= val;
          when "^=" do a.a ^= val;
          when "+=" do a.a |= val;
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else if isRealType(binop_dtype_a) && (isRealType(binop_dtype_b) || isIntegralType(binop_dtype_b)) {
        select op {
          when "+=" do a.a += val;
          when "-=" do a.a -= val;
          when "*=" do a.a *= val;
          when "/=" do a.a /= val:binop_dtype_a;
          when "//=" {
            ref aa = a.a;
            [ai in aa] ai = floorDivisionHelper(ai, val);
          }
          when "**=" do a.a **= val;
          when "%=" {
            ref aa = a.a;
            [ai in aa] ai = modHelper(ai, val);
          }
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else if isRealType(binop_dtype_a) && binop_dtype_b == bool {
        select op {
          when "+=" do a.a += val:binop_dtype_a;
          when "-=" do a.a -= val:binop_dtype_a;
          when "*=" do a.a *= val:binop_dtype_a;
          otherwise do return MsgTuple.error(unsupErrorMsg);
        }
      } else {
        return MsgTuple.error(unsupErrorMsg);
      }

      return MsgTuple.success();
    }

    // special handling for bigint ops
    proc opeqVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws
      where (binop_dtype_a == bigint || binop_dtype_b == bigint) &&
            !isRealType(binop_dtype_b)
    {
      const op = msgArgs['op'].toScalar(string),
            unsupErrorMsg = "unsupported op=: %s %s %s".format(binop_dtype_a:string, op, binop_dtype_b:string);

      if binop_dtype_a != bigint then return MsgTuple.error(unsupErrorMsg);

      const a = st[msgArgs['a']]: borrowed SymEntry(binop_dtype_a, array_nd),
            val = msgArgs['value'].toScalar(binop_dtype_b),
            (has_max_bits, max_size, max_bits) = getMaxBits(a);
      ref aa = a.a;

      select op {
        when "+=" {
          forall ai in aa with (var local_val = val, var local_max_size = max_size) {
            ai += local_val;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "-=" {
          forall ai in aa with (var local_val = val, var local_max_size = max_size) {
            ai -= local_val;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "*=" {
          forall ai in aa with (var local_val = val, var local_max_size = max_size) {
            ai *= local_val;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "//=" {
          if binop_dtype_b == bool then return MsgTuple.error(unsupErrorMsg);
          forall ai in aa with (var local_val = val, var local_max_size = max_size) {
            if local_val != 0 then ai /= local_val; else ai = 0:bigint;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "%=" {
          if binop_dtype_b == bool then return MsgTuple.error(unsupErrorMsg);
          // we can't use ai %= bi because this can result in negatives
          forall ai in aa with (var local_val = val, var local_max_size = max_size) {
            if local_val != 0 then mod(ai, ai, local_val); else ai = 0:bigint;
            if has_max_bits then ai &= local_max_size;
          }
        }
        when "**=" {
          if binop_dtype_b == bool then return MsgTuple.error(unsupErrorMsg);
          if val < 0 then return MsgTuple.error("Attempt to exponentiate base of type BigInt to negative exponent");
          if has_max_bits {
            forall ai in aa with (var local_val = val, var local_max_size = max_size) {
              powMod(ai, ai, local_val, local_max_size + 1);
            }
          } else {
            forall ai in aa with (var local_val = val) {
              ai **= local_val:uint;
            }
          }
        }
        otherwise do return MsgTuple.error(unsupErrorMsg);
      }

      return MsgTuple.success();
    }

    // error message when result could not be stored in 'binop_dtype_a'
    proc opeqVS(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab,
      type binop_dtype_a,
      type binop_dtype_b,
      param array_nd: int
    ): MsgTuple throws {
      const op = msgArgs['op'].toScalar(string);
      return MsgTuple.error(notImplementedError(
        cmd,
        whichDtype(binop_dtype_a),
        op,
        whichDtype(binop_dtype_b)
      ));
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
