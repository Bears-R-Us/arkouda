module ObjectPool {
  use List;
  use LisExprData;
  
  record pool {
    var freeIntList: [0..6] unmanaged ValueClass(int)?;
    var freeRealList: [0..6] unmanaged ValueClass(real)?;
    var intCounter = 0;
    var realCounter = 0;

    proc init() {
      this.complete();
      for i in 0..6 do
        freeRealList[i] = new unmanaged ValueClass(0.0)?;
    }

    proc deinit() {
      forall val in freeIntList do
        delete val;
      forall val in freeRealList do
        delete val;
    }

    proc freeAll() {
      intCounter = 0;
      realCounter = 0;
    }

    proc getInt(val: int) throws {
      if intCounter < 6 {
        intCounter+=1;
        return freeIntList[intCounter-1]!;
      } else {
        throw new Error("Went over");
      }
    }

    // TODO: figure out how to efficiently set the value
    // when popping
    proc getReal(val: real) throws {
      if realCounter < 6 {
        realCounter+=1;
        return freeRealList[realCounter-1]!;
      } else {
        throw new Error("Went over");
      }
    }
  }
}
