module HashUtils {
  use SymArrayDmap;
  use ServerConfig;
  use NumPyDType;
  use SipHash;
  use GenSymIO;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use SegmentedString;
  use CommAggregation;
  use Broadcast;

  /* Hashes of subsequent arrays cannot be simply XORed
    * because equivalent values will cancel each other out.
    * Thus, a non-linear function must be applied to each array,
    * hence we do a rotation by the ordinal of the array. This
    * will only handle up to 128 arrays before rolling back around.
    */
  private proc rotl(h:2*uint(64), n:int):2*uint(64) {
    use BitOps;
    // no rotation
    if n == 0 { return h; }
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

  private proc hashArraysInnerPdarray(ref hashes, name, st, i) throws {
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

  private proc hashArraysInnerStrings(ref hashes, name, st, i) throws {
    var (myNames, _) = name.splitMsgToTuple('+', 2);
    var g = getSegString(myNames, st);
    hashes ^= rotl(g.siphash(), i);
  }

  private proc hashArraysInnerSegarray(ref hashes, name, st, i) throws {
    var segComps = jsonToMap(name);
    var (upper, lower) =
      segarrayHash(segComps["segments"],
                   segComps["values"],
                   segComps["valObjType"], st);
    forall (h, u, l) in zip(hashes, upper, lower) {
      h ^= rotl((u,l), i);
    }
  }
  private proc hashArraysInnerCategorical(ref hashes, name, st, i) throws {
    var catComps = jsonToMap(name);
    var (upper, lower) =
      categoricalHash(catComps["categories"], catComps["codes"], st);
    forall (h, u, l) in zip(hashes, upper, lower) {
      h ^= rotl((u,l), i);
    }
  }

  proc hashArrays(size, names, types, st): [] 2*uint throws {
    overMemLimit(numBytes(uint) * size * 2);
    var dom = makeDistDom(size);
    var hashes = makeDistArray(dom, 2*uint);

    for (name, objtype, i) in zip(names, types, 0..) {
      select objtype.toUpper(): ObjType {
        when ObjType.PDARRAY do
          hashArraysInnerPdarray(hashes, name, st, i);
        when ObjType.STRINGS do
          hashArraysInnerStrings(hashes, name, st, i);
        when ObjType.SEGARRAY do
          hashArraysInnerSegarray(hashes, name, st, i);
        when ObjType.CATEGORICAL do
          hashArraysInnerCategorical(hashes, name, st, i);
      }
    }
    return hashes;
  }

  // just pdarrays and strings to avoid recursion errors
  proc hashArraysMinimal(size, names, types, st): [] 2*uint throws {
    overMemLimit(numBytes(uint) * size * 2);
    var dom = makeDistDom(size);
    var hashes = makeDistArray(dom, 2*uint);

    for (name, objtype, i) in zip(names, types, 0..) {
      select objtype.toUpper(): ObjType {
        when ObjType.PDARRAY do
          hashArraysInnerPdarray(hashes, name, st, i);
        when ObjType.STRINGS do
          hashArraysInnerStrings(hashes, name, st, i);
      }
    }
    return hashes;
  }


  proc categoricalHash(categoriesName: string,
                       codesName: string, st: borrowed SymTab) throws {
    var categories = getSegString(categoriesName, st);
    var codes = toSymEntry(getGenericTypedArrayEntry(codesName, st), int);
    // hash categories first
    var hashes = categories.siphash();
    // then do expansion indexing at codes
    ref ca = codes.a;
    var expandedHashes = makeDistArray(ca.domain, (uint, uint));
    forall (eh, c) in zip(expandedHashes, ca)
    with (var agg = newSrcAggregator((uint, uint))) {
      agg.copy(eh, hashes[c]);
    }
    var hash1 = makeDistArray(ca.size, uint);
    var hash2 = makeDistArray(ca.size, uint);
    forall (h, h1, h2) in zip(expandedHashes, hash1, hash2) {
      (h1,h2) = h:(uint,uint);
    }
    return (hash1, hash2);
  }

  proc segarrayHash(segName: string, valName: string,
                    valObjType: string, st: borrowed SymTab) throws {
    const segments = toSymEntry(getGenericTypedArrayEntry(segName, st), int);
    var values = getGenericTypedArrayEntry(valName, st);
    const size = values.size;
    const broadcastedSegs = broadcast(segments.a, segments.a, size);
    const valInd = makeDistDom(size);

    // calculate segment indices (we use this to prevent
    // segXor from zeroing out arrays like [5,5,5,5])
    // see comments of issue #2459 for more details
    // temporarily add to symtab for hashArrays and then delete entry
    var segInds = [(b, i) in zip (broadcastedSegs, valInd)] i - b;
    var siName = st.nextName();
    st.addEntry(siName, createSymEntry(segInds));
    // we can't just call hashArrays because the compiler
    // complains about recursion
    var hashes =
      hashArraysMinimal(size, [valName, siName],
                        [valObjType, ObjType.PDARRAY:string], st);
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
    var rehash =
      hashArraysMinimal(size, [upperName, lowerName],
                        [ObjType.PDARRAY:string, ObjType.PDARRAY:string], st);
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

  proc segXor(values:[] ?t, segments:[?D] int) throws {
    // Because XOR has an inverse (itself), this can be
    // done with a scan like segSum
    var res = makeDistArray(D, t);
    if D.size == 0 { return res; }
    // check there's enough room to create a copy for scan and throw if
    // creating a copy would go over memory limit
    overMemLimit(numBytes(t) * values.size);
    var cumxor = ^ scan values;
    // Iterate over segments
    var rightvals = makeDistArray(D, t);
    forall (i, r) in zip(D, rightvals) with (var agg = newSrcAggregator(t)) {
      // Find the segment boundaries
      if i == D.high {
        agg.copy(r, cumxor[values.domain.high]);
      } else {
        agg.copy(r, cumxor[segments[i+1] - 1]);
      }
    }
    res[D.low] = rightvals[D.low];
    res[D.low+1..] = rightvals[D.low+1..] ^ rightvals[..D.high-1];
    return res;
  }


}
