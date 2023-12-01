module ArkoudaTimeCompat {
  public use Time;

  proc createFromTimestampCompat(d) {
    return dateTime.createUtcFromTimestamp(d).getDate();
  }
}
