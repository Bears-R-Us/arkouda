use Version;

proc main() {
  if chplVersion < createVersion(1,25) {
    use Regexp;
    var myRegex = compile("a+");
  }
  else {
    use Regex;
    var myRegex = compile("a+");
  }
  writeln("Chapel correctly made with RE2 support\n");
  return 0;
}
