module ArkoudaRangeCompat {
  type stridableRange = range(strides=strideKind.any);
  proc stridable(a) param {
    return !(a.strides==strideKind.one);
  }
}