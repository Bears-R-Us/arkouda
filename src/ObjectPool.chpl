module ObjectPool {
  use List;
  use LisExprData;
  
  record pool {
    var freeRealList: [0..10] unmanaged ValueClass(real) = new unmanaged ValueClass(0.0);
    var realCounter = 0;

    proc deinit() {
      writeln("DEINIT");
      forall val in freeRealList {
        writeln(val);
        delete val;
      }
    }

    proc freeAll() {
      realCounter = 0;
    }

    // TODO: figure out how to efficiently set the value
    // when popping
    proc getReal(val: real) throws {
      if realCounter < 10 {
        ref curr = freeRealList[realCounter];
        curr.v = val;
        realCounter+=1;
        return curr;
      } else {
        throw new Error("Went over");
      }
    }
  }
}
