module ArkoudaSortCompat {
  public use Sort except defaultComparator, DefaultComparator;

  proc defaultComparator type {
    import Sort;
    return Sort.DefaultComparator;
  }

  proc compatSort(ref x, comparator) {
    sort(x, comparator=comparator, stable=true); // real stable sort
  }
}
