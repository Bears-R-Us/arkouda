module ArkoudaSortCompat {

  use Sort;

  // 1.34 split distributed and non-distributed two-array sorters
  // into separate routines
  // Previous to this, Sort.TwoArrayRadixSort.twoArrayRadixSort
  // handled both distributed and non-distributed sorting
  // by checking isDefaultRectangular() as this does below.
  proc twoArrayRadixSort(ref Data:[], comparator:?rec=defaultComparator) {
    if Data._instance.isDefaultRectangular() {
      Sort.TwoArrayRadixSort.twoArrayRadixSort(Data, comparator);
    } else {
      Sort.TwoArrayDistributedRadixSort.twoArrayDistributedRadixSort(Data, comparator);
    }
  }
}
