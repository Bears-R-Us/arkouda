use Parquet;

proc main() {
  var ArrowVersion = getVersionInfo();
  writeln("Found Arrow version: ", ArrowVersion);
  return 0;
}
