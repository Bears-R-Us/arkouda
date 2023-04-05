module ArkoudaArrayCompat {
  proc _array.find(dset, ref idx: int) {
    var (col_exists, i) = this.find(dset);
    idx = i;
    return col_exists;
  }
}
