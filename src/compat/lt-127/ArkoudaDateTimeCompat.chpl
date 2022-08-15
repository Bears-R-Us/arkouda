module ArkoudaDateTimeCompat {
    // Chapel v1.26
    public use DateTime;
    proc date.isoCalendar() {
      return this.isocalendar();
    }
}