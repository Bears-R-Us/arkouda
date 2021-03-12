use TestBase;

config const size = 10**4;
config const minLen = 1;
config const maxLen = 16;

proc main() {
  var st = new owned SymTab();
  var (segs, vals) = newRandStringsUniformLength(size*numLocales, minLen, maxLen);
  var strings = getSegString(segs, vals, st);

  var d: Diags;
  d.start();
  strings.argsort();
  d.stop(printTime=false);

  const MB = byteToMB(vals.size);
  if printTimes then
    writef("Sorted %i strings (%.1dr MB) in %.2dr seconds (%.2dr MB/s)\n", size*numLocales, MB, d.elapsed(), MB/d.elapsed());
}
