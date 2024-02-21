module ArkoudaRandomCompat {
  use Random;
  proc sample(arr: [?d] ?t, n: int, withReplacement: bool): [] t throws {
    var r = new randomStream(int);
    return r.choice(arr, size=n, replace=withReplacement);
  }
}
