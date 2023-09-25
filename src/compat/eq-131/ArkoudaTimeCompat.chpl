module ArkoudaTimeCompat {
  public use Time;

  proc createFromTimestampCompat(d) {
    return date.createFromTimestamp(d);
  }

  proc date.isoWeekDate() {
    return this.isoCalendar();
  }
}
