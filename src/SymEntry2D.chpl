module SymEntry2D {

  use MultiTypeSymEntry;

  class SymEntry2D : GenSymEntry {
      type etype;

      var aD: {0..#size, 0..#size}.type;
      var a: [aD] etype;
      var m: int;
      var n: int;

      proc init(m: int, n: int, type etype) {
        super.init(etype, (m*n));
        this.etype = etype;
        this.aD = {0..#m, 0..#n};
        this.m = m;
        this.n = n;
        this.ndim = 2;
      }

      proc init(a: [?D] int) {
        super.init(int, 10);
        this.etype = int;
        this.aD = D;
        this.a = a;
        this.m = D.high[0]+1;
        this.n = D.high[1]+1;
        this.ndim = 2;
      }

      override proc __str__(thresh:int=6, prefix:string = "[", suffix:string = "]", baseFormat:string = "%t"): string throws {
        var s:string = "";

        for i in 0..#this.m {
          for j in 0..#this.n {
            s += try! baseFormat.format(this.a[i, j]) + " ";
          }
          if i != m-1 then
            s += "; ";
        }


        return prefix + s + suffix;
      }
    }
}
