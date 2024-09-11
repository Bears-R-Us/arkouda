
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

    proc type2str(type t): string {
      if t == uint(8) then return "uint8";
      if t == uint(16) then return "uint16";
      if t == uint(32) then return "uint32";
      if t == uint then return "uint64";
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
      if t == uint(8) then return "%i";
      if t == uint(16) then return "%i";
      if t == uint(32) then return "%i";
      if t == uint then return "%i";
      if t == int(8) then return "%i";
      if t == int(16) then return "%i";
      if t == int(32) then return "%i";
      if t == int then return "%i";
      if t == real(32) then return "%.17r";
      if t == real then return "%.17r";
      if t == complex(64) then return "%.17z%"; // "
      if t == complex(128) then return "%.17z%"; // "
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
      Numpy's type promotion rules

      (including some additional rules for bigint)

      see: https://numpy.org/doc/2.1/reference/arrays.promotion.html#numerical-promotion
  */
  proc promotedType(type a, type b) type {
      if getKind(a) == getKind(b) {
      return maxSizedType(a, b);
      } else {
          if getKind(a) == DTK.BOOL {
              return b;
          } else if getKind(b) == DTK.BOOL {
              return a;
          } else if getKind(a) == DTK.SINT && getKind(b) == DTK.UINT {
              if typeSize(a) > typeSize(b)
              then return a;
              else return promoteUIntToSInt(b);
          } else if getKind(a) == DTK.UINT && getKind(b) == DTK.SINT {
              if typeSize(b) > typeSize(a)
              then return b;
              else return promoteUIntToSInt(a);
          } else if getKind(a) == DTK.REAL && (getKind(b) == DTK.UINT || getKind(b) == DTK.SINT) {
              if typeSize(a) > typeSize(b)
              then return a;
              else return promoteIntToReal(b);
          } else if (getKind(a) == DTK.UINT || getKind(a) == DTK.SINT) && getKind(b) == DTK.REAL {
              if typeSize(b) > typeSize(a)
              then return b;
              else return promoteIntToReal(a);
          } else if getKind(a) == DTK.COMPLEX && (getKind(b) == DTK.UINT || getKind(b) == DTK.SINT) {
              if typeSize(b) < 4
              then return a;
              else return complex(128);
          } else if (getKind(a) == DTK.UINT || getKind(a) == DTK.SINT) && getKind(b) == DTK.COMPLEX {
              if typeSize(a) < 4
              then return b;
              else return complex(128);
          } else if getKind(a) == DTK.COMPLEX && getKind(b) == DTK.REAL {
              if typeSize(b) <= 4
              then return a;
              else return complex(128);
          } else if getKind(a) == DTK.REAL && getKind(b) == DTK.COMPLEX {
              if typeSize(a) <= 4
              then return b;
              else return complex(128);
          } else if getKind(a) == DTK.BIGINT || getKind(b) == DTK.BIGINT {
              // type promotions between bigint and non-integer types don't make
              // sense; however, prohibiting such cases needs to be handled outside
              // this procedure
              return bigint;
          } else {
              // should be unreachable
              return int;
          }
      }
  }

  enum DTK {
      SINT,
      UINT,
      REAL,
      COMPLEX,
      BOOL,
      BIGINT,
  }

  proc getKind(type t) param: DTK {
      if isIntType(t) then return DTK.SINT;
      if isUintType(t) then return DTK.UINT;
      if isRealType(t) then return DTK.REAL;
      if isComplexType(t) then return DTK.COMPLEX;
      if t == bigint then return DTK.BIGINT;
      return DTK.BOOL;
  }

  proc maxSizedType(type a, type b) type {
      if typeSize(a) >= typeSize(b)
      then return a;
      else return b;
  }

  proc promoteUIntToSInt(type t) type {
      if t == uint(8) then return int(16);
      if t == uint(16) then return int(32);
      if t == uint(32) then return int(64);
      return real(64);
  }

  proc promoteIntToReal(type t) type {
      if t == int(8) || t == uint(8) ||
          t == int(16) || t == uint(16)
          then return real(32);
          else return real(64);
  }

  proc typeSize(type t) param: int {
      if t == bigint then return 16;
      if t == bool then return 1;
      return numBytes(t);
  }

}
