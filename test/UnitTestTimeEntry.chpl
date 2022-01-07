use UnitTest;
use TestBase;
use TimeEntryModule;
use GenSymIO;
use Message;
import FileSystem;

config const size = 1_000;
config const file_path = "resources/tmp_UnitTestHDF5";

proc createDatetime(name: string, st: borrowed SymTab) {
  // Create and store a TimeEntry
  var dt = st.addTimeEntry(name, size, DType.Datetime64);
  // Step by 1 second starting from a few days before 2020
  const start = 50*365*24*60*60*10**9;
  const step = 10**9;
  //const a = makeDistArray(size, int);
  forall (ai, d) in zip(dt.a, start..#size by step) {
    ai = d;
  }
  return dt;
}

proc testDatetime(t: borrowed Test) {
  var st = new owned SymTab();
  var dt = createDatetime("mydt", st);
  // Get it back as TimeEntry and do a time-specific computation
  var dt2 = getTimeEntry("mydt", st).floor(TimeUnit.Minutes);
  st.addEntry("byminute", dt2);
  /* writeln("orig:    ", dt.a[0..#5]); */
  /* writeln("rounded: ", dt2.a[0..#5]); */
  var diffSeconds = (dt.a - dt2.a) / (10**9);
  var isCorrect = && reduce ((diffSeconds >= 0) & (diffSeconds < 60));
  t.assertTrue(isCorrect);

  // Retrieve as SymEntry(int)
  var asint = toSymEntry(toGenSymEntry(st.lookup("mydt")), int);
  var isEqual = && reduce (asint.a == dt.a);
  t.assertTrue(isEqual);
  // writeln("(SymEntry == TimeEntry)? >>> ", isEqual, " <<<");
}

/**
 * Utility method to clean up files
 */
proc removeFile(fn:string) {
    try {
        if (FileSystem.exists(fn)) {
            unlink(fn);
        }
    } catch {}
}

proc testIO(t: borrowed Test) throws {
  var st = new owned SymTab();
  var dt = createDatetime("mydt", st);
  var cmd = "tohdf";
  var path_prefix = file_path+"_datetime";
  defer {
    removeFile(path_prefix + "_LOCALE0000");
  }
  // payload -> arrayName, dsetName, modeStr, jsonfile, dataType
  var payload = "mydt" + " times 0 [" + Q+path_prefix+Q + "] datetime64";
  var msg = tohdfMsg(cmd, payload, st);
  t.assertTrue(msg.msg.strip() == "wrote array to file");
  cmd = "readAllHdfMsg";
  payload = "false 1 1 false false ["+Q+"times"+Q+"] | ["+Q+path_prefix+"_LOCALE0000"+Q+"]";
  msg = readAllHdfMsg(cmd, payload, st);
  t.assertTrue(msg.msg.find("created id_1 datetime64 ") > 0);
  var dt2 = getTimeEntry("id_1", st);
  var isEqual = && reduce (dt2.a == dt.a);
  t.assertTrue(isEqual);
}

proc main() {
  var t = new Test();
  testDatetime(t);
  testIO(t);
}