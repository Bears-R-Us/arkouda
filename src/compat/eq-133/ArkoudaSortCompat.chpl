module ArkoudaSortCompat {
  use Sort;
  proc twoArrayRadixSort(ref Data:[], comparator:?rec=defaultComparator) {
      Sort.TwoArrayRadixSort.twoArrayRadixSort(Data, comparator);
  }
}
