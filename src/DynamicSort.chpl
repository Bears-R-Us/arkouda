module DynamicSort {
    private use ArkoudaSortCompat;

    proc dynamicTwoArrayRadixSort(ref Data:[], comparator:?rec=new defaultComparator()) {
      if Data._instance.isDefaultRectangular() {
        ArkoudaSortCompat.TwoArrayRadixSort.twoArrayRadixSort(Data, comparator);
      } else {
        ArkoudaSortCompat.TwoArrayDistributedRadixSort.twoArrayDistributedRadixSort(Data, comparator);
      }
    }

}