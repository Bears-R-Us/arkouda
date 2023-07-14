module ArkoudaBigIntCompat {
  public use BigInteger;

  inline operator bigint.:(src: bool, type toType: bigint): bigint throws {
    return new bigint(src:int);
  }

  proc mod(ref result: bigint, const ref a: bigint, const ref b: bigint) {
    result.mod(a, b);
  }

  proc mod(ref result: bigint, const ref a: bigint, b: integral) : int {
    return result.mod(a, b);
  }

  proc powMod(ref result: bigint,
              const ref base: bigint,
              const ref exp: bigint,
              const ref mod: bigint) {
    result.powMod(base, exp, mod);
  }

  proc powMod(ref result: bigint,
              const ref base: bigint,
              exp: int,
              const ref mod: bigint)  {
    result.powMod(base, exp, mod);
  }

  proc powMod(ref result: bigint,
              const ref base: bigint,
              exp: uint,
              const ref mod: bigint)  {
    result.powMod(base, exp, mod);
  }

  // methods needed to to right shift for int and uint
  use CTypes;
  use GMP;
  extern type mp_bitcnt_t = c_ulong;

  proc rightShift(const ref a: bigint, b: int): bigint {
    var c = new bigint();
    shiftRight(c, a, b);
    return c;
  }

  proc rightShift(const ref a: bigint, b: uint): bigint {
    return a >> b;
  }

  proc rightShiftEq(ref a: bigint, b:  int) {
    shiftRight(a, a, b);
  }

  proc rightShiftEq(ref a: bigint, b:  uint) {
    a >>= b;
  }

  private inline proc shiftRight(ref result: bigint, const ref a: bigint, b: int) {
    if b >= 0 {
      shiftRight(result, a, b:uint);
    } else {
      mul_2exp(result, a, (0 - b):uint);
    }
  }

  private inline proc shiftRight(ref result: bigint, const ref a: bigint, b: uint) {
    divQ2Exp(result, a, b);
  }

  proc mul_2exp(ref result: bigint, const ref a: bigint, b: integral) {
    const b_ = b.safeCast(mp_bitcnt_t);

    if _local {
      mpz_mul_2exp(result.mpz, a.mpz, b_);
    } else if result.localeId == chpl_nodeID {
      const a_ = a;
      mpz_mul_2exp(result.mpz, a_.mpz, b_);
    } else {
      const resultLoc = chpl_buildLocaleID(result.localeId, c_sublocid_any);
      on __primitive("chpl_on_locale_num", resultLoc) {
        const a_ = a;
        mpz_mul_2exp(result.mpz, a_.mpz, b_);
      }
    }
  }

  proc divQ2Exp(ref result: bigint,
                const ref numer: bigint,
                exp: integral) {
    const exp_ = exp.safeCast(mp_bitcnt_t);

    if _local {
      mpz_fdiv_q_2exp(result.mpz, numer.mpz, exp_);
    } else if result.localeId == chpl_nodeID {
      const numer_ = numer;
      mpz_fdiv_q_2exp(result.mpz, numer_.mpz, exp_);
    } else {
      const resultLoc = chpl_buildLocaleID(result.localeId, c_sublocid_any);
      on __primitive("chpl_on_locale_num", resultLoc) {
        const numer_ = numer;
        mpz_fdiv_q_2exp(result.mpz, numer_.mpz, exp_);
      }
    }
  }
}
