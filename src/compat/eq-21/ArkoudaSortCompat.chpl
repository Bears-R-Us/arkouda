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

  proc compatSort(ref x, comparator) {
    // Tag each item with its original index to enforce stable sort
    var indexed: [x.domain] (int, x.eltType);
    forall i in x.domain do
      indexed[i] = (i, x[i]);

    // Define a comparator that respects the original index for stability
    record StableComparator {
      var cmp;
      proc key(t: (int, ?T)) do return cmp.key(t[1]);
      proc ties(t: (int, ?T)) do return t[0];
    }

    sort(indexed, comparator=new StableComparator(cmp=comparator));

    forall i in x.domain do
      x[i] = indexed[i][1];
  }
}
