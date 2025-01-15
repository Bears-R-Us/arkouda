module ArkoudaSortCompat {
  public use Sort except defaultComparator, DefaultComparator;

  proc defaultComparator type {
    import Sort;
    return Sort.DefaultComparator;
  }
}
