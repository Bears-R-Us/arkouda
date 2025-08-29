module ArkoudaSortCompat {
  public use Sort except defaultComparator;

  proc defaultComparator type {
    import Sort;
    return Sort.DefaultComparator;
  }
}
