use List, Time, IO;

record orderedMap {
  type keyT, valueT;
  var keys: list(keyT);
  var vals: list(valueT);

  proc this(k: keyT) ref {
    if !keys.contains(k) {
      keys.append(k);
      var defValue: valueT;
      vals.append(defValue);
    }
    return vals[keys.find(k)];
  }
  iter items() {
    for (k, v) in zip(keys, vals) do yield (k, v);
  }
}

proc Timer.startStop() {
  if running then stop();
             else start();
}

var timers: orderedMap(string, Timer);
proc deinit()  {
  for (k, v) in timers.items() {
    try! writeln("%20s : %.2drs".format(k, v.elapsed()));
  }
}
