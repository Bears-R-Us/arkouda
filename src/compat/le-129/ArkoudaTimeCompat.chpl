module ArkoudaTimeCompat {
  use Time;

  record TimeTmpRec {
    proc totalSeconds() {
      return getCurrentTime();
    }
  }
  
  proc timeSinceEpoch() {
    var a = new TimeTmpRec();
    return a;
  }
}
