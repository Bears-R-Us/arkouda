
module NumPyDType
{
  use BigInteger;
  use UInt128;

  /* In chapel the types int and real default to int(64) and real(64).
    We also need other types like float32, int32, etc */
  enum DType {
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
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
      if etype == uint128       then return DType.UInt128;
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
        when DType.UInt128 do return 16;
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

    proc typeSize(type t): int {
      return dtypeSize(whichDtype(t));
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
        when "uint128" do return DType.UInt128;
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
        when DType.UInt128 do return "uint128";
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

    proc type2str(type t): string {
      if t == uint(8) then return "uint8";
      if t == uint(16) then return "uint16";
      if t == uint(32) then return "uint32";
      if t == uint then return "uint64";
      if t == uint128 then return "uint128";
      if t == int(8) then return "int8";
      if t == int(16) then return "int16";
      if t == int(32) then return "int32";
      if t == int then return "int64";
      if t == real(32) then return "float32";
      if t == real then return "float64";
      if t == complex(64) then return "complex64";
      if t == complex(128) then return "complex128";
      if t == bool then return "bool";
      if t == bigint then return "bigint";
      if t == string then return "str";
      return "undef";
    }

    proc type2fmt(type t): string {
      if t == uint(8) then return "%u";
      if t == uint(16) then return "%u";
      if t == uint(32) then return "%u";
      if t == uint then return "%u";
      if t == int(8) then return "%i";
      if t == int(16) then return "%i";
      if t == int(32) then return "%i";
      if t == int then return "%i";
      if t == real(32) then return "%.17r";
      if t == real then return "%.17r";
      if t == complex(64) then return "%.17z%";
      if t == complex(128) then return "%.17z%";
      if t == bool then return "%s";
      if t == bigint then return "%?";
      if t == string then return "%s";
      return "%?";
    }

    proc bool2str(b: bool): string {
      if b then return "True";
      else return "False";
    }

    proc bool2str(b: ?t): t
      where t != bool
        do return b;

    /*
      Return the dtype that can store the result of
      an operation between two dtypes for the following
      operations: ``+``, ``-``, ``*``, ``**``, ``//``, ``%``,
      ``&``, ``|``, ``^``, ``<<``, ``>>``

      follows Numpy's rules for type promotion
      (of which the array-api promotion rules are a subset)
    */
    proc commonDType(a: DType, b: DType): DType {
      select (scalarDTypeKind(a), scalarDTypeKind(b)) {
        when (DTK.Integer, DTK.Integer) {
          if isSignedIntegerDType(a) == isSignedIntegerDType(b) {
            return maxDType(a, b);
          } else {
            const (s, u) = if isSignedIntegerDType(a) then (a, b) else (b, a);
            return maxDType(promoteToNextSigned(u), s);
          }
        }
        when (DTK.Integer, DTK.Float)
          do return maxDType(promoteToNextFloat(a), b);
        when (DTK.Float, DTK.Integer)
          do return maxDType(promoteToNextFloat(b), a);
        when (DTK.Integer, DTK.Complex)
          do return maxDType(promoteToNextComplex(a), b);
        when (DTK.Complex, DTK.Integer)
          do return maxDType(promoteToNextComplex(b), a);
        when (DTK.Float, DTK.Float)
          do return maxDType(a, b);
        when (DTK.Float, DTK.Complex)
          do return maxDType(promoteToNextComplex(a), b);
        when (DTK.Complex, DTK.Float)
          do return maxDType(promoteToNextComplex(b), a);
        when (DTK.Complex, DTK.Complex)
          do return maxDType(a, b);
        otherwise {
            if a == DType.Bool && b != DType.Bool then
                return b;
            else if a != DType.Bool && b == DType.Bool then
                return a;
            else return DType.Bool;
        }
      }
    }

    /*
      Return the dtype that can store the result of
      a division operation between two dtypes
      (following Numpy's rules for type promotion)
    */
    proc divDType(a: DType, b: DType): DType {
      select (scalarDTypeKind(a), scalarDTypeKind(b)) {
        when (DTK.Integer, DTK.Integer)
          do return DType.Float64;
        when (DTK.Integer, DTK.Float)
          do return if dtypeSize(a) < 4 && b == DType.Float32
            then DType.Float32
            else DType.Float64;
        when (DTK.Float, DTK.Integer)
          do return if a == DType.Float32 && dtypeSize(b) < 4
            then DType.Float32
            else DType.Float64;
        when (DTK.Integer, DTK.Complex)
          do return maxDType(promoteToNextComplex(a), b);
        when (DTK.Complex, DTK.Integer)
          do return maxDType(promoteToNextComplex(b), a);
        when (DTK.Float, DTK.Float)
          do return maxDType(a, b);
        when (DTK.Float, DTK.Complex)
          do return maxDType(promoteToNextComplex(a), b);
        when (DTK.Complex, DTK.Float)
          do return maxDType(promoteToNextComplex(b), a);
        when (DTK.Complex, DTK.Complex)
          do return maxDType(a, b);
        when (DTK.Bool, DTK.Float)
            do return b;
        when (DTK.Float, DTK.Bool)
            do return a;
        when (DTK.Bool, DTK.Complex)
            do return b;
        when (DTK.Complex, DTK.Bool)
            do return a;
        otherwise do return DType.Float64;
      }
    }

    private proc maxDType(a: DType, b: DType): DType {
      if dtypeSize(a) >= dtypeSize(b)
          then return a;
          else return b;
    }

    enum DTK {
      Integer,
      Float,
      Complex,
      Bool,
      Other
    }

    private proc scalarDTypeKind(dt: DType): DTK {
      select dt {
        when DType.UInt8 do return DTK.Integer;
        when DType.UInt16 do return DTK.Integer;
        when DType.UInt32 do return DTK.Integer;
        when DType.UInt64 do return DTK.Integer;
        when DType.UInt128 do return DTK.Integer;
        when DType.Int8 do return DTK.Integer;
        when DType.Int16 do return DTK.Integer;
        when DType.Int32 do return DTK.Integer;
        when DType.Int64 do return DTK.Integer;
        when DType.Float32 do return DTK.Float;
        when DType.Float64 do return DTK.Float;
        when DType.Complex64 do return DTK.Complex;
        when DType.Complex128 do return DTK.Complex;
        when DType.Bool do return DTK.Bool;
        otherwise do return DTK.Other;
      }
    }

    private proc isSignedIntegerDType(dt: DType): bool {
        select dt {
            when DType.Int8 do return true;
            when DType.Int16 do return true;
            when DType.Int32 do return true;
            when DType.Int64 do return true;
            otherwise do return false;
        }
    }

    private proc promoteToNextSigned(dt: DType): DType {
      select dt {
        when DType.Bool do return DType.Int8;
        when DType.UInt8 do return DType.Int16;
        when DType.UInt16 do return DType.Int32;
        when DType.UInt32 do return DType.Int64;
        when DType.UInt64 do return DType.Float64;
        when DType.Int8 do return DType.Int16;
        when DType.Int16 do return DType.Int32;
        when DType.Int32 do return DType.Int64;
        when DType.Int64 do return DType.Float64;
        when DType.Float32 do return DType.Float64;
        when DType.Float64 do return DType.Float64;
        when DType.Complex64 do return DType.Complex128;
        when DType.Complex128 do return DType.Complex128;
        otherwise do return DType.UNDEF;
      }
    }

    private proc promoteToNextFloat(dt: DType): DType {
      select dt {
        when DType.Bool do return DType.Float32;
        when DType.UInt8 do return DType.Float32;
        when DType.UInt16 do return DType.Float32;
        when DType.UInt32 do return DType.Float64;
        when DType.UInt64 do return DType.Float64;
        when DType.Int8 do return DType.Float32;
        when DType.Int16 do return DType.Float32;
        when DType.Int32 do return DType.Float64;
        when DType.Int64 do return DType.Float64;
        when DType.Float32 do return DType.Float64;
        when DType.Float64 do return DType.Float64;
        when DType.Complex64 do return DType.Complex128;
        when DType.Complex128 do return DType.Complex128;
        otherwise do return DType.UNDEF;
      }
    }

    private proc promoteToNextComplex(dt: DType): DType {
      select dt {
        when DType.Bool do return DType.Complex64;
        when DType.UInt8 do return DType.Complex64;
        when DType.UInt16 do return DType.Complex64;
        when DType.UInt32 do return DType.Complex128;
        when DType.UInt64 do return DType.Complex128;
        when DType.Int8 do return DType.Complex64;
        when DType.Int16 do return DType.Complex64;
        when DType.Int32 do return DType.Complex128;
        when DType.Int64 do return DType.Complex128;
        when DType.Float32 do return DType.Complex64;
        when DType.Float64 do return DType.Complex128;
        when DType.Complex64 do return DType.Complex128;
        when DType.Complex128 do return DType.Complex128;
        otherwise do return DType.UNDEF;
      }
    }
}
