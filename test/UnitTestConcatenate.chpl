use SegmentedArray;
use RandArray;
use MultiTypeSymbolTable;

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
  st.addEntry(s1Name, s1);
  var v1Name = st.nextName();
  st.addEntry(v1Name, v1);
  var s2Name = st.nextName();
  st.addEntry(s2Name, s2);
  var v2Name = st.nextName();
  st.addEntry(v2Name, v2);
  var reqMsg = "concatenate 2 str %s+%s %s+%s".format(s1Name, v1Name, s2Name, v2Name);
  var repMsg = concatenateMsg(reqMsg, st);
  var tmp = repMsg.split('+');
  var resSegName = tmp[1].split()[1];
  var resValName = tmp[2].split()[1];
  var resStr = new owned SegString(resSegName, resValName, st);
  var str1 = new owned SegString(s1Name, v1Name, st);
  var str2 = new owned SegString(s2Name, v2Name, st);
  var correct: bool;
  correct = && reduce (str1 == resStr[0..#str1.size]);
  correct &&= && reduce (str2 == resStr[str1.size..#str2.size]);
  writeln("Correct answer? >>> ", correct, " <<<");
}

proc main() {
  testConcat(N, MINLEN, MAXLEN);
}