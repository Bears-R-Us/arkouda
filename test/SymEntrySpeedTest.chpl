use TestBase, AggDiags;

config const n = 10;
config const trials = if printDiags || printDiagsSum then 1 else 250;

proc main()  {

  var D = makeDistDom(n);
  var A: [D] int;
  var S = new SymEntry(n, int);
  for 1..trials {

    //
    // Create Array/SymEntry from scratch
    //
    diags[trials:string+" arrs from scratch"].startStop();
    {
      var A2 = makeDistArray(n, int);
    }
    diags[trials:string+" arrs from scratch"].startStop();

    diags[trials:string+" syms from scratch"].startStop();
    {
      var A2 = new SymEntry(n, int);
    }
    diags[trials:string+" syms from scratch"].startStop();


    //
    // Create Array from existing Domain/SymEntry
    //
    diags[trials:string+" arrs from dom"].startStop();
    {
      var A2: [D] int;
    }
    diags[trials:string+" arrs from dom"].startStop();

    diags[trials:string+" arrs from sym"].startStop();
    {
      var A2: [S.a.domain] int;
    }
    diags[trials:string+" arrs from sym"].startStop();


    //
    // Create Array/SymEntry from existing Array
    //
    diags[trials:string+" arrs from arr"].startStop();
    {
      var A2 = A;
    }
    diags[trials:string+" arrs from arr"].startStop();

    diags[trials:string+" syms from arr"].startStop();
    {
      var A2 = new SymEntry(A);
    }
    diags[trials:string+" syms from arr"].startStop();


    //
    // Create Array/SymEntry from existing dead Array
    //
    {
      var deadA = A;
      diags[trials:string+" arrs from dead arr"].startStop();
      {
          var A2 = deadA;
      }
      diags[trials:string+" arrs from dead arr"].startStop();
    }

    {
      var deadA = A;
      diags[trials:string+" syms from dead arr"].startStop();
      {
          var A2 = new SymEntry(deadA);
      }
      diags[trials:string+" syms from dead arr"].startStop();
    }
  }
}

module AggDiags {
  use List, TestBase;

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

  record AggDiags {
    var d: Diags;
    var dAgg: Diags;

    proc startStop() {
      if d.T.running {
        d.stop(printTime=false, printDiag=false, printDiagSum=false);
        dAgg.elapsedTime += d.elapsed();
        dAgg.D = d.D;
      } else {
        d.start();
      }
    }
  }

  var diags: orderedMap(string, AggDiags);
  proc deinit()  {
    const maxLen = max reduce for k in diags.keys do k.size;
    for (k, v) in diags.items() {
      if printTimes    then writef("%s %s: %.2drs\n", k, " "*(maxLen-k.size), v.dAgg.elapsed());
      if printDiags    then writef("%s %s: %s\n",     k, " "*(maxLen-k.size), v.dAgg.comm():string);
      if printDiagsSum then writef("%s %s: %t\n",     k, " "*(maxLen-k.size), v.dAgg.commSum());
    }
  }
}

