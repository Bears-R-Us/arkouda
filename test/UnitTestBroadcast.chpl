use TestBase;
use BroadcastMsg;
use Broadcast;

proc testBroadcast() {
  var st = new owned SymTab();
  var segs = st.addEntry("segs", 3, int);
  segs.a = [0, 3, 6];
  var vals = st.addEntry("vals", 3, int);
  vals.a = [-1, 2, -3];
  var reqMsg = "perm segs vals False 9";
  var repMsg = broadcastMsg(cmd="broadcast", payload=reqMsg, st).msg;
  var resName = parseName(repMsg);
  var res = toSymEntry(st.lookup(resName), int);
  var testvec = makeDistArray(9, int);
  testvec = [-1, -1, -1, 2, 2, 2, -3, -3, -3];
  const correct = && reduce (testvec == res.a);
  writeln("Correct without perm? >>> ", correct, " <<<");
  var perm = st.addEntry("perm", 9, int);
  perm.a = [0, 3, 6, 1, 4, 7, 2, 5, 8];
  reqMsg = "perm segs vals True 9";
  repMsg = broadcastMsg(cmd="broadcast", payload=reqMsg, st).msg;
  var res2Name = parseName(repMsg);
  var res2 = toSymEntry(st.lookup(res2Name), int);
  var testvec2 = makeDistArray(9, int);
  testvec2 = [-1, 2, -3, -1, 2, -3, -1, 2, -3];
  const correct2 = && reduce (testvec2 == res2.a);
  writeln("correct with perm? >>> ", correct2, " <<<");
}

proc main() {
  testBroadcast();
}
