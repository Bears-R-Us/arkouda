module ArkoudaBigIntCompat {
  use BigInteger;

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
}