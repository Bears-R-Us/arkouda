use ZMQ;

proc main() {
  var (Zmajor, Zminor, Zmicro) = ZMQ.version;
  writef("Found ZMQ version: %t.%t.%t\n", Zmajor, Zminor, Zmicro);
  return 0;
}
