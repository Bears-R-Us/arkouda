module ArkoudaRangeCompat {
  type stridableRange = range(stridable=true);
  proc stridable(a) param {
    return a.stridable;
  }
}