use TestBase;
use FileIO;
use GenSymIO;
use HDF5Msg;
use Set;

config const size = 10**4;

proc main() {
  var D = makeDistDom(size*numLocales);
  var A, B: [D] int;
  A = D;
  const GiB = (8*D.size):real / (2**30):real;

  var d: Diags;

  d.start();
  write1DDistArray("file", TRUNCATE, "dst", A, DType.Bool);
  d.stop(printTime=false);
  if printTimes then writeln("write: %.2dr GiB/s (%.2drs)".format(GiB/d.elapsed(), d.elapsed()));

  var filenames = generateFilenames("file", "", B.targetLocales().size);
  var (subdoms, _, skips)  = get_subdoms(filenames, "dst");
  d.start();
  var skipSet = new set(string);
  read_files_into_distributed_array(B, subdoms, filenames, "dst", skipSet);
  d.stop(printTime=false);
  if printTimes then writeln("read: %.2dr GiB/s (%.2drs)".format(GiB/d.elapsed(), d.elapsed()));
  forall (a, b) in zip (A, B) do assert(a == b);
}
