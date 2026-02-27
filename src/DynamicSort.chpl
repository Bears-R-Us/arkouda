module DynamicSort {
    private use Sort;

    proc dynamicTwoArrayRadixSort(ref Data:[], comparator:?rec=new defaultComparator()) {
      if Data._instance.isDefaultRectangular() {
        Sort.TwoArrayRadixSort.twoArrayRadixSort(Data, comparator);
      } else {
        Sort.sort(Data, comparator);
      }
    }

}
