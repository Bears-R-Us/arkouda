module ArkoudaBitOpsCompat {
  use BitOps;

  proc popCount(x: integral) {
    return popcount(x);
  }
}
