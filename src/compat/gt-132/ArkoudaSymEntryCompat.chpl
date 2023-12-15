module ArkoudaSymEntryCompat {
  use MultiTypeSymEntry;
  use Map;

  type SymEntryAny = SymEntry(?);
  type mapAny = map(?);

  override proc SymEntry.serialize(writer, ref serializer) throws {
    use Reflection;
    var f = writer;
    proc writeField(f, param i) throws {
      if !isArray(getField(this, i)) {
        f.write(getFieldName(this.type, i), " = ", getField(this, i):string);
      } else {
        f.write(getFieldName(this.type, i), " = ", formatAry(getField(this, i)));
      }
    }

    super.writeThis(f);
    f.write(" {");
    param nFields = numFields(this.type);
    for param i in 0..nFields-2 {
      writeField(f, i);
      f.write(", ");
    }
    writeField(f, nFields-1);
    f.write("}");
  }

  implements writeSerializable(SymEntry);

  proc GenSymEntry.init(type etype, len: int = 0, ndim: int = 1) {
    this.entryType = SymbolEntryType.TypedArraySymEntry;
    assignableTypes.add(this.entryType);
    this.dtype = whichDtype(etype);
    this.itemsize = dtypeSize(this.dtype);
    this.size = len;
    this.ndim = ndim;
    init this;
    if len == 0 then
      this.shape = "[0]";
    else
      this.shape = tupShapeString(1, ndim);
  }

  /*
    This init takes length and element type

    :arg len: length of array to be allocated
    :type len: int

    :arg etype: type to be instantiated
    :type etype: type
  */
  proc SymEntry.init(args: int ...?N, type etype) {
    var len = 1;
    for param i in 0..#N {
      len *= args[i];
    }
    super.init(etype, len, N);
    this.entryType = SymbolEntryType.PrimitiveTypedArraySymEntry;
    assignableTypes.add(this.entryType);

    this.etype = etype;
    this.dimensions = N;
    this.tupShape = args;
    this.a = try! makeDistArray((...args), etype);
    init this;
    this.shape = tupShapeString(this.tupShape);
    this.ndim = N;
  }

  /*
    This init takes an array whose type matches `makeDistArray()`

    :arg a: array
    :type a: [] ?etype
  */
  proc SymEntry.init(in a: [?D] ?etype, max_bits=-1) {
    super.init(etype, D.size);
    this.entryType = SymbolEntryType.PrimitiveTypedArraySymEntry;
    assignableTypes.add(this.entryType);

    this.etype = etype;
    this.dimensions = D.rank;
    this.tupShape = D.shape;
    this.a = a;
    this.max_bits=max_bits;
    init this;
    this.shape = tupShapeString(this.tupShape);
    this.ndim = D.rank;
  }
}
