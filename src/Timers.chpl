use Time, AllLocalesBarriers;

pragma "default intent is ref"
record enumTimers {
  type enumT;
  param enabled = true;
  var timers: if enabled then [0..<enumT.size] Timer else nothing;

  proc mark(param e: enumT, param barrier=false, timingTask=true) {
    if enabled {
      if barrier {
        allLocalesBarrier.barrier();
      }
      if timingTask {
        ref t = timers[e:int];
        if t.running then t.stop();
                     else t.start();
      }
    }
  }

  inline proc writeThis(f) {
    if enabled {
      var maxSize = 0;
      for e in enumT do
        maxSize = max(maxSize, (e:string).size);
      for e in enumT do
        f.write("%s %s: %.2drs\n".format(e, (maxSize-(e:string).size)*" ", timers[e:int].elapsed()));
    }
  }

  inline proc clear() {
    if enabled {
      timers.clear();
    }
  }
}
