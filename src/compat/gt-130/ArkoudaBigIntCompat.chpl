module ArkoudaBigIntCompat {
  proc rightShift(const ref a: bigint, b: int): bigint {
    return a >> b;
  }

  proc rightShift(const ref a: bigint, b: uint): bigint {
    return a >> b;
  }

  proc rightShiftEq(ref a: bigint, b:  int) {
    a >>= b;
  }

  proc rightShiftEq(ref a: bigint, b:  uint) {
    a >>= b;
  }
}
