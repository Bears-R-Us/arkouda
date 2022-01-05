use TestBase;
use TimeClasses;

config const size = 1_000;

proc testDatetime() {
  var st = new owned SymTab();
  // Step by 1 second starting from a few days before 2020
  const start = 50*365*24*60*60*10**9;
  const step = 10**9;
  const a = makeDistArray(size, int);
  forall (ai, d) in zip(a, start..#size by step) {
    ai = d;
  }
  // Create and store a TimeEntry
  var dt = new shared TimeEntry(a, DType.Datetime64);
  st.addEntry("mydt", dt);
  // Get it back as TimeEntry and do a time-specific computation
  var dt2 = getTimeEntry("mydt", st).floor(TimeUnit.Minutes);
  st.addEntry("byminute", dt2);
  writeln("orig:    ", dt.a[0..#5]);
  writeln("rounded: ", dt2.a[0..#5]);

  // Retrieve as SymEntry(int)
  var asint = toSymEntry(toGenSymEntry(st.lookup("mydt")), int);
  var isEqual = && reduce (asint.a == dt.a);
  writeln("(SymEntry == TimeEntry)? >>> ", isEqual, " <<<");
}

proc main() {
  testDatetime();
}