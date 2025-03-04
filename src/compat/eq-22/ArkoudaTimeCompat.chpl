module ArkoudaTimeCompat {
  public use Time;

  proc timeDelta.totalMicroseconds(): int{
    return ((days*(24*60*60) + seconds)*1_000_000 + microseconds): int;
  }  
}