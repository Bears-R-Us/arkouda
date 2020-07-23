/*
 * kextreme data structure that is used
 * to track the minimum or maximum `size`
 * values that are in an array. Acts similar
 * to a heap data structure, but can be
 * sorted and merged to extract the
 * extreme values from multiple kextreme
 * objects.
 */

module KExtreme {
  use SymArrayDmap;
  use Sort;
  record kextreme {
    type eltType;
    var size: int;
    var dom = {0..#size};
    var numEmpty: int = size-1;
    var isSorted: bool = false;
    var isMinReduction=true;
    var _data: [dom] eltType = if isMinReduction then max(eltType) else min(eltType);

    
    proc pushArr(arr: [?D]) {
      for i in D {
        push(arr[i]);
      }
    }

    // Push a value into the kextreme
    // instance, only pushes a value if
    // it is an encountered extreme
    proc push(val: eltType) {
      if isMinReduction {
        if(numEmpty > 1 && _data[0] == max(eltType)) {
          _data[numEmpty] = val;
          numEmpty-=1;
        } else if val < _data[0] {
          _data[0] = val;
          heapifyDown();
        }
      } else {
        if(numEmpty > 1 && _data[0] == min(eltType)) {
          _data[numEmpty] = val;
          numEmpty-=1;
        } else if val > _data[0] {
          _data[0] = val;
          heapifyDown();
        }
      }
    }

    // Restore heap property from the
    // top element down
    proc heapifyDown() {
      if isMinReduction {
        var i = 0;
        while(i < size) {
          var initial = i;
          var l = 2*i+1; // left child
          var r = 2*i+2; // right child
          if (l < size && _data[l] > _data[i]) then
            i = l;
          // if right child is larger than largest so far 
          if (r < size && _data[r] > _data[i]) then
            i = r;
          // if the largest value isn't the initial 
          if (initial != i) {
            _data[i] <=> _data[initial];
          } else break;
        }
      } else {
        var i = 0;
        while(i < size) {
          var initial = i;
          var l = 2*i+1; // left child
          var r = 2*i+2; // right child
          if (l < size && _data[l] < _data[i]) then
            i = l;
          // if right child is larger than largest so far 
          if (r < size && _data[r] < _data[i]) then
            i = r;
          // if the largest value isn't the initial 
          if (initial != i) {
            _data[i] <=> _data[initial];
          } else break;
        }
      }
    }

    // Sort the kextreme values if needed,
    // moving from a heap to a sorted array
    proc doSort() {
      sort(_data);
      isSorted = true;
    }

    iter these() {
      for e in _data {
        yield e;
      }
    }
  }

  // Sort both heaps and then merge them
  // returns an array that contains the
  // smallest values from each array sorted.
  // Returned array is size of the original heaps.
  proc merge(ref v1: kextreme(int), ref v2: kextreme(int)): [v1._data.domain] int {
    if v1.isMinReduction {
      if !v1.isSorted then v1.doSort();
      if !v2.isSorted then v2.doSort();

      var first = v1._data;
      var second = v2._data;

            writeln("v1: ", v1._data);
      writeln("v2: ", v2._data);

      
      var ret: [first.domain] v1.eltType;
      var a,b: int = 0;
      for i in ret.domain {
        if(first[a] < second[b]) {
          ret[i] = first[a];
          a += 1;
        } else {
          ret[i] = second[b];
          b += 1;
        }
      }
      return ret;
    } else {
      if !v1.isSorted then v1.doSort();
      if !v2.isSorted then v2.doSort();

      var first = v1._data;
      var second = v2._data;
      var a,b,i = first.domain.high;
      var ret: [first.domain] v1.eltType;
      while(i >= 0) {
        if(first[a] > second[b]) {
          ret[i] = first[a];
          i-=1;
          a-=1;
        } else {
          ret[i] = second[b];
          b-=1;
          i-=1;
        }
      }
      return ret;
    }
  } 
}
