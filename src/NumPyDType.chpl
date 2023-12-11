
module NumPyDType
{
  use BigInteger;

  /* In chapel the types int and real default to int(64) and real(64).
    We also need other types like float32, int32, etc */
  enum DType {
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Complex64,
    Complex128,
    Bool,
    BigInt,
    Strings,
    UNDEF
  };

    /*
    Take a chapel type and return the matching DType

    :arg etype: chapel type

    :returns: DType
    */
    proc whichDtype(type etype) param : DType {
      if etype == uint(8)       then return DType.UInt8;
      if etype == uint(16)      then return DType.UInt16;
      if etype == uint(32)      then return DType.UInt32;
      if etype == uint          then return DType.UInt64;
      if etype == int(8)        then return DType.Int8;
      if etype == int(16)       then return DType.Int16;
      if etype == int(32)       then return DType.Int32;
      if etype == int           then return DType.Int64;
      if etype == real(32)      then return DType.Float32;
      if etype == real          then return DType.Float64;
      if etype == complex(64)   then return DType.Complex64;
      if etype == complex(128)  then return DType.Complex128;
      if etype == bool          then return DType.Bool;
      if etype == bigint        then return DType.BigInt;
      if etype == string        then return DType.Strings;
      return DType.UNDEF; // undefined type
    }


    /* Returns the size in bytes of a DType

    :arg dt: (pythonic) DType
    :type dt: DType

    :returns: (int)
    */
    proc dtypeSize(dt: DType): int {
      select dt {
        when DType.UInt8 do return 1;
        when DType.UInt16 do return 2;
        when DType.UInt32 do return 4;
        when DType.UInt64 do return 8;
        when DType.Int8 do return 1;
        when DType.Int16 do return 2;
        when DType.Int32 do return 4;
        when DType.Int64 do return 8;
        when DType.Float32 do return 4;
        when DType.Float64 do return 8;
        when DType.Complex64 do return 8;
        when DType.Complex128 do return 16;
        when DType.Bool do return 1;
        // TODO figure out the best way to do size estimation
        when DType.BigInt do return 16;
        otherwise do return 0;
      }
    }

    /* Turns a dtype string in pythonland into a DType

    :arg dstr: pythonic dtype to be converted
    :type dstr: string

    :returns: DType
    */
    proc str2dtype(dstr:string): DType {
      select dstr {
        when "uint8" do return DType.UInt8;
        when "uint16" do return DType.UInt16;
        when "uint32" do return DType.UInt32;
        when "uint64" do return DType.UInt64;
        when "uint" do return DType.UInt64;
        when "int8" do return DType.Int8;
        when "int16" do return DType.Int16;
        when "int32" do return DType.Int32;
        when "int64" do return DType.Int64;
        when "int" do return DType.Int64;
        when "float32" do return DType.Float32;
        when "float64" do return DType.Float64;
        when "float" do return DType.Float64;
        when "complex64" do return DType.Complex64;
        when "complex128" do return DType.Complex128;
        when "bool" do return DType.Bool;
        when "bigint" do return DType.BigInt;
        when "str" do return DType.Strings;
        otherwise do return DType.UNDEF;
      }
    }

    /* Turns a DType into a dtype string in pythonland

    :arg dtype: DType to convert to string
    :type dtype: DType

    :returns: (string)
    */
    proc dtype2str(dt: DType): string {
      select dt {
        when DType.UInt8 do return "uint8";
        when DType.UInt16 do return "uint16";
        when DType.UInt32 do return "uint32";
        when DType.UInt64 do return "uint64";
        when DType.Int8 do return "int8";
        when DType.Int16 do return "int16";
        when DType.Int32 do return "int32";
        when DType.Int64 do return "int64";
        when DType.Float32 do return "float32";
        when DType.Float64 do return "float64";
        when DType.Complex64 do return "complex64";
        when DType.Complex128 do return "complex128";
        when DType.Bool do return "bool";
        when DType.BigInt do return "bigint";
        when DType.Strings do return "str";
        otherwise do return "undef";
      }
    }

}
