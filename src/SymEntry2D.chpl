module SymEntry2D {
  use MultiTypeSymEntry;
  use SymArrayDmap;

  class SymEntry2D : GenSymEntry {
    type etype;

    var m: int;
    var n: int;
    var aD: makeDistDom2D(m,n).type;
    var a: [aD] etype;

    proc init(m: int, n: int, type etype) {
      super.init(etype, (m*n));
      this.etype = etype;
      this.m = m;
      this.n = n;
      this.aD = makeDistDom2D(m, n);
      this.ndim = 2;
    }

    proc init(a: [?D] ?etype) {
      super.init(etype, a.size);
      this.etype = etype;
      this.m = D.high[0]+1;
      this.n = D.high[1]+1;
      this.aD = D;
      this.a = a;
      this.ndim = 2;
    }

    // TODO: not using threshold, how do we want to print large
    // 2D arrays?
    override proc __str__(thresh:int=6, prefix:string = "[", suffix:string = "]", baseFormat:string = "%t"): string throws {
      var s:string = "";

      for i in 0..#this.m {
        s += "[";
        for j in 0..#this.n {
          s += try! baseFormat.format(this.a[i, j]);
          if j != n-1 then
            s += ", ";
          else if i != m-1 then
            s +=  "],\n       ";
        }
      }
      s += "]";

      if (bool == this.etype) {
        s = s.replace("true","True");
        s = s.replace("false","False");
      }

      return prefix + s + suffix;
    }
  }

  proc toSymEntry2D(e, type etype) {
    return try! e :borrowed SymEntry2D(etype);
  }

  proc makeDistDom2D(m: int, n: int) {
    select MyDmap {
        when Dmap.defaultRectangular {
          return {0..#m, 0..#n};
        }
        when Dmap.blockDist {
          if m > 0 && n > 0 {
            return {0..#m, 0..#n} dmapped Block(boundingBox={0..#m, 0..#n});
          }
          // fix the annoyance about boundingBox being enpty
          else {return {0..#0, 0..#0} dmapped Block(boundingBox={0..0, 0..0});}
        }
        when Dmap.cyclicDist {
          return {0..#m, 0..#n} dmapped Cyclic(startIdx=0);
        }
        otherwise {
          halt("Unsupported distribution " + MyDmap:string);
        }
      }
  }

  proc makeDistArray2D(m: int, n: int, type etype) {
    var a: [makeDistDom2D(m, n)] etype;
    return a;
  }
}
