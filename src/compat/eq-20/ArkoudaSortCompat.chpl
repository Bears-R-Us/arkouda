module ArkoudaSortCompat {
  public use Sort except defaultComparator, DefaultComparator;

  interface keyComparator {}
  interface keyPartComparator {}
  interface relativeComparator {}

  module keyPartStatus {
    proc pre param do return -1;
    proc returned param do return 0;
    proc post param do return 1;
  }

  proc defaultComparator type {
    import Sort;
    return Sort.DefaultComparator;
  }

  proc compatSort(ref x: [] ?T, comparator) {
    // Tag each item with its original index to enforce stable ordering
    var tagged: [x.domain] (T, int);
    forall i in x.domain do
      tagged[i] = (x[i], i);

    record StableComparator {
      var cmp;

      proc compare(a: (T, int), b: (T, int)) {
        const ka = cmp.key(a[0]);
        const kb = cmp.key(b[0]);
        if ka < kb then return -1;
        else if ka > kb then return 1;
        else return compare(a[1], b[1]); // tie-break on original index
      }
    }

    sort(tagged, comparator = new StableComparator(cmp = comparator));

    // Restore to x
    forall i in x.domain do
      x[i] = tagged[i][0];
  }

}
