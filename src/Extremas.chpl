// Extremas module with KExtreme record that is
// used in
//
//

module Extremas {
  use SymArrayDmap;
  use Sort;

/*
:record:`KExtreme` data structure that is used
to track the minimum or maximum ``size``
values that are in an array. Acts similar
to a heap data structure, but can be
sorted and merged to extract the
extreme values from multiple kextreme
objects.

This record is used in the user-defined ``kreduce``
reuction by having one :record: `KExtreme` instance
per task, which is then built up by pushing the
values into the heap-like :record: `KExtreme` by
the ``accumulate()`` function. Once all of the values
from each task have been accumulated, the ``combine()``
function is called on each :record: `KExtreme` instance,
which is handled by sorting the record, giving up
its heap-like quality, and using it as a sorted array
to increase the efficiency of the merge step.
*/
  record KExtreme {
    type eltType;
    var size: int;
    var dom = {0..#size};
    var numEmpty: int = size-2;
    var isSorted: bool = false;
    var isMinReduction=true;
    var _data: [dom] (eltType, int) = if isMinReduction then (max(eltType), -1) else (min(eltType), -1);
    
    proc pushArr(arr: [?D]) {
      for i in D {
        push(arr[i]);
      }
    }

    // Push a value into the KExtreme
    // instance, only pushes a value if
    // it is an encountered extreme
    proc push(val: (eltType, int)) {

      // Accumulate/push can be called after combine/merge. This routine
      // expects the data to be a heap and not sorted, so it would fail if we
      // tried to insert now. The way Arkouda uses this, val will be the
      // identity values, so we can ignore it.
      if isSorted {
        const identity = if isMinReduction then (max(eltType), -1) else (min(eltType), -1);
        assert(val == identity);
        return;
      }

      const shouldAdd = if isMinReduction then val(0)<_data[0](0) else val(0)>_data[0](0);
      const shouldAddEmpty = (numEmpty>0) && if isMinReduction then
        _data[0](0)==max(eltType) else _data[0](0)==min(eltType);

      if shouldAddEmpty {
        _data[numEmpty+1] = val;
        numEmpty-=1;
      } else if shouldAdd {
        _data[0] = val;
        heapifyDown();
      }
    }

    // Restore heap property from the
    // top element down
    proc heapifyDown() {
      var i = 0;
      while(i < size) {
        const initial = i;
        const l = 2*i+1; // left child
        const r = 2*i+2; // right child
        const cmpLeft = l<size && if isMinReduction then
          _data[l](0) > _data[i](0) else _data[l](0) < _data[i](0);
        if (cmpLeft) then i = l;
        // if right child is more extreme than largest so far
        const cmpRight = r<size && if isMinReduction then
          _data[r](0) > _data[i](0) else _data[r](0) < _data[i](0);
        if (cmpRight) then i = r;
        // if the extreme value isn't the initial 
        if (initial != i) {
          _data[i] <=> _data[initial];
        } else break;
      }
    }

    // Sort the KExtreme values if needed,
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
  proc merge(ref v1: KExtreme(?t), ref v2: KExtreme(t)): [v1._data.domain] (t ,int) {
    const isMin = v1.isMinReduction;
    if !v1.isSorted then v1.doSort();
    if !v2.isSorted then v2.doSort();

    const first = v1._data;
    const second = v2._data;
    var ret: [first.domain] (v1.eltType, int);
    var a,b,i: int = if v1.isMinReduction then
      0 else first.domain.high;
    const increment = if v1.isMinReduction then 1 else -1;
    
    while(if isMin then i <= ret.domain.high else i >= 0) {
      if(if isMin then first[a] < second[b] else first[a] > second[b]) {
        ret[i] = first[a];
        a += increment;
      } else {
        ret[i] = second[b];
        b += increment;
      }
      i+= increment;
    }
    return ret;
  }
}
