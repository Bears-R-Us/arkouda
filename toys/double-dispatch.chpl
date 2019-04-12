class GenSymEntry {
  proc dispatchOp(op:string, rhs:borrowed GenSymEntry): owned GenSymEntry {
    halt("pure virtual method");
  }
  proc doOp(op:string, lhs): owned GenSymEntry {
    halt("pure virtual method");
  }
}

class SymEntry : GenSymEntry {
  type etype;
  var aD;
  var a: [aD] etype;

  proc init(a:[]) {
    this.etype = a.eltType;
    this.aD = a.domain;
    this.a = a;
  }

  // Evaluates lhs <op> this
  override proc doOp(op:string, lhs:borrowed SymEntry): owned GenSymEntry {
    var rhs:borrowed SymEntry = this;

    // At this point, lhs and rhs have compile-time known types
    // (i.e. the are instantiations of SymEntry, rather than GenSymEntry)
    select op
    {
      when "+" {
        var result = lhs.a + rhs.a;
        return new owned SymEntry(result);
      }
      when "-" {
        var result = lhs.a - rhs.a;
        return new owned SymEntry(result);
      }
    }
    return nil;
  }

  // Evaluate this <op> rhs
  // through a double-dispatch to get the concrete type available
  override proc dispatchOp(op:string,
                           rhs:borrowed GenSymEntry): owned GenSymEntry {
    return rhs.doOp(op, this); // pass concrete "this", dispatch on other
  }
}

var ones: owned GenSymEntry = new owned SymEntry([1,1,1,1]);
var nums: owned GenSymEntry = new owned SymEntry([1,2,3,4]);
var rls:  owned GenSymEntry  = new owned SymEntry([0.1, 0.2, 0.3, 0.4]);

writeln(ones.dispatchOp("+", nums));
writeln(nums.dispatchOp("-", ones));
writeln(rls.dispatchOp("+", ones));
