module ObjectPool {
  use List;
  use LisExprData;
  
  record pool {
    var freeIntList: list(unmanaged ValueClass(int));

    proc init() {
      freeIntList = new list(unmanaged ValueClass(int));
    }

    proc deinit() {
      forall val in freeIntList do
        delete val;
    }

    proc getInt(val: int) {
      if freeIntList.size == 0 {
        return new unmanaged ValueClass(val);
      } else {
        return freeIntList.pop();
      }
    }

    proc freeInt(val: unmanaged ValueClass(int)) {
      freeIntList.append(val);
    }
  }
}
