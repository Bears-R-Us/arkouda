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
}
