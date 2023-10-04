module SegmentedArray {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use ServerConfig;
  use ReductionMsg;
  use Broadcast;
  use UniqueMsg;
  use SipHash;
  use SegmentedString;

  proc hashHelper(size, names, types, st): [] 2*uint throws {
    // Basically a duplicate of hashArrays which only operates on pdarray and strings
    // we can't just call hashArrays because the compiler complains about recursion
    overMemLimit(numBytes(uint) * size * 2);
    var dom = makeDistDom(size);
    var hashes = makeDistArray(dom, 2*uint);
    /* Hashes of subsequent arrays cannot be simply XORed
     * because equivalent values will cancel each other out.
     * Thus, a non-linear function must be applied to each array,
     * hence we do a rotation by the ordinal of the array. This
     * will only handle up to 128 arrays before rolling back around.
     */
    proc rotl(h:2*uint(64), n:int):2*uint(64) {
      use BitOps;
      // no rotation
      if (n == 0) { return h; }
      // Rotate each 64-bit word independently, then swap tails
      const (h1, h2) = h;
      // Mask for tail (right-hand portion)
      const rmask = (1 << n) - 1;
      // Mask for head (left-hand portion)
      const lmask = 2**64 - 1 - rmask;
      // Rotate each word
      var r1 = rotl(h1, n);
      var r2 = rotl(h2, n);
      // Swap tails
      r1 = (r1 & lmask) | (r2 & rmask);
      r2 = (r2 & lmask) | (r1 & rmask);
      return (r1, r2);
    }
    for (name, objtype, i) in zip(names, types, 0..) {
      select objtype.toUpper(): ObjType {
        when ObjType.PDARRAY {
          var g = getGenericTypedArrayEntry(name, st);
          select g.dtype {
            when DType.Int64 {
              var e = toSymEntry(g, int);
              ref ea = e.a;
              forall (h, x) in zip(hashes, ea) {
                h ^= rotl(sipHash128(x), i);
              }
            }
            when DType.UInt64 {
              var e = toSymEntry(g, uint);
              ref ea = e.a;
              forall (h, x) in zip(hashes, ea) {
                h ^= rotl(sipHash128(x), i);
              }
            }
            when DType.Float64 {
              var e = toSymEntry(g, real);
              ref ea = e.a;
              forall (h, x) in zip(hashes, ea) {
                h ^= rotl(sipHash128(x), i);
              }
            }
            when DType.Bool {
              var e = toSymEntry(g, bool);
              ref ea = e.a;
              forall (h, x) in zip(hashes, ea) {
                h ^= rotl((0:uint, x:uint), i);
              }
            }
          }
        }
        when ObjType.STRINGS {
          var (myNames, _) = name.splitMsgToTuple('+', 2);
          var g = getSegString(myNames, st);
          hashes ^= rotl(g.siphash(), i);
        }
      }
    }
    return hashes;
  }

  proc segarrayHash(segName: string, valName: string, valObjType: string, st: borrowed SymTab) throws {
    const segments = toSymEntry(getGenericTypedArrayEntry(segName, st), int);
    var values = getGenericTypedArrayEntry(valName, st);
    const size = values.size;
    const broadcastedSegs = broadcast(segments.a, segments.a, size);
    const valInd = makeDistDom(size);

    // calculate segment indices (we use this to prevent segXor from zeroing out arrays like [5,5,5,5])
    // see comments of issue #2459 for more details
    // temporarily add to symtab for hashArrays and then delete entry
    var segInds = [(b, i) in zip (broadcastedSegs, valInd)] i - b;
    var siName = st.nextName();
    st.addEntry(siName, createSymEntry(segInds));
    // we can't just call hashArrays because the compiler complains about recursion
    var hashes = hashHelper(size, [valName, siName], [valObjType, ObjType.PDARRAY:string], st);
    st.deleteEntry(siName);

    var upper = makeDistArray(size, uint);
    var lower = makeDistArray(size, uint);
    forall (up, low, h) in zip(upper, lower, hashes) {
      (up, low) = h;
    }

    // we have to hash twice before the XOR to avoid collisions with
    // things like ak.Segarray(ak.array([0,2],ak.array([1,1,2,2]))
    var upperName = st.nextName();
    st.addEntry(upperName, createSymEntry(upper));
    var lowerName = st.nextName();
    st.addEntry(lowerName, createSymEntry(lower));
    var rehash = hashHelper(size, [upperName, lowerName], [ObjType.PDARRAY:string, ObjType.PDARRAY:string], st);
    st.deleteEntry(upperName);
    st.deleteEntry(lowerName);

    var rehashUpper = makeDistArray(size, uint);
    var rehashLower = makeDistArray(size, uint);
    forall (up, low, h) in zip(rehashUpper, rehashLower, rehash) {
      (up, low) = h;
    }

    var xorUpper = segXor(rehashUpper, segments.a);
    var xorLower = segXor(rehashLower, segments.a);
    return (xorUpper, xorLower);
  }
}