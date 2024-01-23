module ArkoudaMathCompat {
  inline proc nan param : real(64) do return chpl_NAN;
  inline proc inf param : real(64) do return chpl_INFINITY;

  inline proc isNan(x: real(64)): bool do return isnan(x);
  inline proc isInf(x: real(64)): bool do return isinf(x);
  inline proc isFinite(x: real(64)): bool do return isfinite(x);

  inline proc mathRound(x) do return AutoMath.round(x);
  inline proc mathAbs(x) do return abs(x);
}
