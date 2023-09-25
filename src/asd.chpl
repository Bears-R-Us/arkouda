use AryUtil;

var values: [0..10] int;
var res: [0..10] int;

coforall loc in Locales with (ref values, ref res) {
  var slice = new lowLevelLocalizingSlice(values, start..#len);
}