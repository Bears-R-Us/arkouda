use TestBase;

use ConcatenateMsg;

config const N = 10_000;
config const MINLEN = 1;
config const MAXLEN = 20;

proc addStrings(n:int, minLen:int, maxLen:int, st: borrowed SymTab) {
  var (s1, v1) = newRandStringsUniformLength(n, minLen, maxLen);
  var s1Name = st.nextName();
  st.addEntry(s1Name, s1);
  var v1Name = st.nextName();
  st.addEntry(v1Name, v1);
  return (s1, v1, s1Name, v1Name);
}

proc testConcat(n:int, minLen:int, maxLen:int) {
  var st = new owned SymTab();
  var (s1, v1) = newRandStringsUniformLength(n, minLen, maxLen);
  var (s2, v2) = newRandStringsUniformLength(n, minLen, maxLen);
  var s1Name = st.nextName();
  var s1e = st.addEntry(s1Name, s1.size, int);
  s1e.a = s1;
  var v1Name = st.nextName();
  var v1e = st.addEntry(v1Name, v1.size, uint(8));
  v1e.a = v1;
  var str1 = new owned SegString(s1Name, v1Name, st);
  writeSegString("Str 1, %i elem, %i bytes".format(str1.size, str1.nBytes), str1);
  var s2Name = st.nextName();
  var s2e = st.addEntry(s2Name, s2.size, int);
  s2e.a = s2;
  var v2Name = st.nextName();
  var v2e = st.addEntry(v2Name, v2.size, uint(8));
  v2e.a = v2;
  var str2 = new owned SegString(s2Name, v2Name, st);

  writeSegString("\nStr 2, %i elem, %i bytes".format(str2.size, str2.nBytes), str2);

  var reqMsg = "2 str %s+%s %s+%s".format(s1Name, v1Name, s2Name, v2Name);
  writeReq(reqMsg);
  var repMsg = concatenateMsg(cmd="concatenate", payload=reqMsg.encode(), st);
  writeRep(repMsg);
  var (resSegAttribStr, resValAttribStr) = repMsg.splitMsgToTuple('+', 2);
  var resSegAttrib = parseName(resSegAttribStr);
  var resValName = parseName(resValAttribStr);
  var resStr = new owned SegString(resSegAttrib, resValName, st);
  writeSegString("\nResult, %i elem, %i bytes".format(resStr.size, resStr.nBytes), resStr);

  var correct = true; // TODO
  /* forall (i, j) in zip(str1.offsets.a, resStr.offsets.a) { */
  /*   if i != j { */
  /*     correct = false; */
  /*   } */
  /* } */
  /* var correct = && reduce (str1.offsets.a == resStr.offsets.a[{0..#str1.size}]); */
  /* correct &&= && reduce (str1.values.a == resStr.values.a[{0..#str1.nBytes}]); */
  /* correct &&= && reduce (str2.offsets.a == resStr.offsets.a[{str1.size..#str2.size}]); */
  /* correct &&= && reduce (str2.values.a == resStr.values.a[{str1.nBytes..#str2.nBytes}]); */
  writeln("Correct answer? >>> ", correct, " <<<");
}

proc main() {
  testConcat(N, MINLEN, MAXLEN);
}
