use TestBase;

use SegmentedMsg;

config const N: int = 10_000;
config const MINLEN: int = 6;
config const MAXLEN: int = 30;
config const SUBSTRING: string = "hi";
config const DEBUG = false;

proc make_strings(substr, n, minLen, maxLen, characters, st) {
  const nb = substr.numBytes;
  const sbytes: [0..#nb] uint(8) = for b in substr.chpl_bytes() do b;
  var (segs, vals) = newRandStringsUniformLength(n, minLen, maxLen, characters);
  var strings = new owned SegString(segs, vals, st);
  var lengths = strings.getLengths() - 1;
  var r: [segs.domain] int;
  fillInt(r, 0, 100);
  var splits = (r % 3);
  /* var present = (r < 5) & (lengths >= nb); */
  /* present[present.domain.high] = true; */
  /* present[present.domain.low] = true; */
  forall (rn, o, l, s) in zip(r, segs, lengths, splits) {
    if l >= nb {
      if (s == 1) || ((s == 2) && (l < 2*(nb+1))) {
        if l == nb {
          vals[{o..#nb}] = sbytes;
        } else {
          const i = rn % (l - nb);
          vals[{(o+i)..#nb}] = sbytes;
        }
      } else if s == 2 {
        const i = rn % ((l/2) - nb);
        vals[{(o+i)..#nb}] = sbytes;
        const j = i + (l/2);
        vals[{(o+j)..#nb}] = sbytes;
      }
    }
  }
  var strings2 = new shared SegString(segs, vals, st);
  return (splits, strings2);
}

proc testPeel(substr:string, n:int, minLen:int, maxLen:int, characters:charSet = charSet.Uppercase) throws {
  var st = new owned SymTab();
  var t = new Timer();
  writeln("Generating random strings..."); stdout.flush();
  t.start();
  var (answer, strings) = make_strings(substr, n, minLen, maxLen, characters, st);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  const lengths = strings.getLengths();
  var allSuccess = true;
  for param times in 1..2 {
    for param id in 0..1 {
      param includeDelimiter:bool = id > 0;
      for param kp in 0..1 {
        param keepPartial:bool = kp > 0;
        for param l in 0..1 {
          param left:bool = l > 0;
          writeln("strings.peel(%s, %i, includeDelimiter=%t, keepPartial=%t, left=%t)".format(substr, times, includeDelimiter, keepPartial, left));
          t.start();
          var (leftOffsets, leftVals, rightOffsets, rightVals) = strings.peel(substr, times, includeDelimiter, keepPartial, left);
          t.stop();
          writeln("%t seconds".format(t.elapsed())); stdout.flush();
          var lstr = new owned SegString(leftOffsets, leftVals, st);
          var rstr = new owned SegString(rightOffsets, rightVals, st);
          if DEBUG {
            var llen = lstr.getLengths();
            var rlen = rstr.getLengths();
            var badLen = + reduce ((llen <= 0) | (rlen <= 0));
            writeln("Lengths <= 0: %t".format(badLen));
            if badLen > 0 {
              var n = 0;
              for (i, ll, rl, ol) in zip(lstr.offsets.aD, llen, rlen, lengths) {
                if (ll <= 0) {
                  n += 1;
                  writeln("%i: %s (%i) -> <bad> (%i) | %s (%i)".format(i, strings[i], ol, ll, rstr[i], rl));
                } else if (rl <= 0) {
                  n += 1;
                  writeln("%i: %s (%i) -> %s (%i) | <bad> (%i)".format(i, strings[i], ol, lstr[i], ll, rl));
                }
                if n >= 5 {
                  stdout.flush();
                  break;
                }
              }
            }
          } 
          for i in 0..#min(lstr.size, 5) {
            writeln("%i: %s  |  %s".format(i, lstr[i], rstr[i]));
          }
          if lstr.size >= 10 {
            for i in (lstr.size-5)..#4 {
              writeln("%i: %s  |  %s".format(i, lstr[i], rstr[i]));
            }
          }
          const delim = if includeDelimiter then "" else substr;
          var temp: owned SegString?;
          if left {
            var (roundOff, roundVals) = lstr.stick(rstr, delim, true);
            temp = new owned SegString(roundOff, roundVals, st);
          } else {
            var (roundOff, roundVals) = rstr.stick(lstr, delim, false);
            temp = new owned SegString(roundOff, roundVals, st);
          }
          var roundTrip: borrowed SegString = temp!;
          var eq = (strings == roundTrip) | (answer < times);
          var success = && reduce eq;
          writeln("\nRound trip success? >>> %t <<<\n".format(success));
          allSuccess &&= success;
          if !success {
            var n = 0;
            const rtlen = roundTrip.getLengths();
            for (i, e) in zip(strings.offsets.aD, eq) {
              if !e {
                n += 1;
                writeln("%i: %s (%i) -> %s (%i)".format(i, strings[i], lengths[i], roundTrip[i], rtlen[i]));
              }
              if n >= 5 {
                break;
              }
            }
          }
        }
      }
    }
  }
  writeln("All round trip tests passed? >>> %t <<<".format(allSuccess));
}

proc testMessageLayer(substr, n, minLen, maxLen) throws {
  var st = new owned SymTab();
  var t = new Timer();
  writeln("Generating random strings..."); stdout.flush();
  t.start();
  var (answer, strings) = make_strings(substr, n, minLen, maxLen, charSet.Uppercase, st);
  t.stop();
  writeln("%t seconds".format(t.elapsed())); stdout.flush(); t.clear();
  var reqMsg = "segmentedEfunc peel str %s %s str 1 True True True %jt".format(strings.offsetName, strings.valueName, [substr]);
  writeln(reqMsg);
  var repMsg = segmentedEfuncMsg(reqMsg, st);
  writeln(repMsg);
  var (loAttribs,lvAttribs,roAttribs,rvAttribs) = repMsg.splitMsgToTuple('+', 4);
  var loname = parseName(loAttribs);
  var lvname = parseName(lvAttribs);
  var roname = parseName(roAttribs);
  var rvname = parseName(rvAttribs);
  reqMsg = "segBinopvv stick str %s %s str %s %s False %jt".format(loname, lvname, roname, rvname, [""]);
  writeln(reqMsg);
  repMsg = segBinopvvMsg(reqMsg, st);
  writeln(repMsg);
  var (rtoAttribs,rtvAttribs) = repMsg.splitMsgToTuple('+', 2);
  var rtoname = parseName(rtoAttribs);
  var rtvname = parseName(rtvAttribs);
  var roundTrip = new owned SegString(rtoname, rtvname, st);
  var success = && reduce (strings == roundTrip);
  writeln("Round trip successful? >>> %t <<<".format(success));
}

proc main() {
  try! testPeel(SUBSTRING, N, MINLEN, MAXLEN);
  try! testMessageLayer(SUBSTRING, N, MINLEN, MAXLEN);
}
  
