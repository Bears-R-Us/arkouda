proc main() {
  use Regexp;
  var myRegex = compile("a+");
  writeln("Chapel correctly made with RE2 support\n");
  return 0;
}
