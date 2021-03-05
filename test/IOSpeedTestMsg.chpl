use TestBase;
use GenSymIO;

config const size = 10**4;
config const write = true;
config const read = true;

proc main() {
  var st = new owned SymTab();
  var A = st.addEntry("A", size*numLocales, int);
  A.a = A.aD;
  const GiB = (8*A.aD.size):real / (2**30):real;

  var d: Diags;

  if write {
    d.start();
    var cmd = "tohdf";
    var payload = "A array 0 [\"file\"] int64";
    tohdfMsg(cmd, payload, st);
    d.stop(printTime=false);
    if printTimes then writeln("write: %.2dr GiB/s (%.2drs)".format(GiB/d.elapsed(), d.elapsed()));
  }

  if read {
    d.start();
    var cmd = "readAllHdf";
    var payload = "True 1 1 [\"array\"] | [\"file_LOCALE*\"]";
    var repMsg = readAllHdfMsg(cmd, payload, st).msg;
    var B = toSymEntry(st.lookup(parseName(repMsg)), int);
    d.stop(printTime=false);
    if printTimes then writeln("read: %.2dr GiB/s (%.2drs)".format(GiB/d.elapsed(), d.elapsed()));
    forall (a, b) in zip (A.a, B.a) do assert(a == b);
  }
}
