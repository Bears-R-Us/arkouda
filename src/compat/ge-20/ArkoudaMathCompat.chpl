module ArkoudaMathCompat {
    private import Math;
    inline proc mathRound(x: real(64)): real(64) do return Math.round(x);
    inline proc mathAbs(x) do return Math.abs(x);
}
