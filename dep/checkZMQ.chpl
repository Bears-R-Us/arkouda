use ZMQ;
use ArkoudaIOCompat;

proc main() {
  var (Zmajor, Zminor, Zmicro) = ZMQ.version;
  writefCompat("Found ZMQ version: %?.%?.%?\n", Zmajor, Zminor, Zmicro);
  return 0;
}
