import FileSystem;

use TestBase;
use ParquetMsg;

// for now, this is the same as Parquet.chpl, maybe we should not make it
// private there?
private config const ROWGROUPS = 512*1024*1024 / numBytes(int); // 512 mb of int64
config const n = 100;

proc testWriteRead() {
  var st = new owned SymTab();

  var arrName = st.nextName();
  var a = st.addEntry(arrName, n, int);

  a.a = 2;

  var (ok, filenames, sizes) = writeDistArrayToParquet(a.a, "test.parquet",
                                                       "col", "int64",
                                                       ROWGROUPS, compression=0,
                                                       mode=0);

  var readEntry = createSymEntry(n, int);
  readFilesByName(readEntry.a, filenames, sizes, "col", "int64");
  var valName = st.nextName();
  st.addEntry(valName, readEntry);

  assert(&& reduce (readEntry.a == a.a));

  defer {
    for filename in filenames {
      if FileSystem.exists(filename) {
        FileSystem.remove(filename);
      }
    }
  }
}

proc testWriteReadMultiCol() {
  var st = new owned SymTab();

  var (arrName, arrEntry) = createArray(int, 4, st);
  arrEntry.a = 2;

  var (segArrSegsName, segArrSegsEntry) = createArray(int, 4, st);
  var (segArrValsName, segArrValsEntry) = createArray(int, n, st);
  segArrSegsEntry.a = [0, 10, 20, 30];
  segArrValsEntry.a = 3;

  var segArrJson = "segments: %s, values: %s".format(segArrSegsName,
                                                     segArrValsName);
  writeMultiColParquet(filename="test.parquet",
                       col_names=["arrCol", "segArrCol"],
                       ncols=2,
                       sym_names=[arrName, segArrJson],
                       col_objTypes=["pdarray", "segarray"],
                       targetLocales=Locales,
                       compression=0,
                       st=st.borrow());


  // TODO read it back

}

proc createArray(type t, size, st) {
  var arrName = st.nextName();
  var arrEntry = st.addEntry(arrName, size, int);

  return (arrName, arrEntry);

}

proc main() {
  testWriteRead();
  testWriteReadMultiCol();
}
