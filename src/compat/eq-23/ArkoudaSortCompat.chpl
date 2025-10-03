module ArkoudaSortCompat {
  public use Sort;

  proc compatSort(ref x, comparator) {
    sort(x, comparator=comparator, stable=true); // real stable sort
  }
}
