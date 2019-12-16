use ZMQ;

proc main() {
  var (Zmajor, Zminor, Zmicro) = ZMQ.version;
  writeln("Found ZMQ version: %t.%t.%t".format(Zmajor, Zminor, Zmicro));
  return 0;
}