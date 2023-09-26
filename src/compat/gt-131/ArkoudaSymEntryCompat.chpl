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
}
