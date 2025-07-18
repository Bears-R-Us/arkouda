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

    // Comparator with tie-breaking on index
    record StableComparator {
      var cmp;

      proc key(t: (T, int)) {
        return cmp.key(t[0]);
      }

      proc ties(t: (T, int)) {
        return t[1];
      }
    }

    sort(tagged, comparator = new StableComparator(cmp = comparator));

    // Restore to x
    forall i in x.domain do
      x[i] = tagged[i][0];
  }

}
