module CastMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Reflection;
  use SegmentedString;
  use ServerErrors;
  use Logging;
  use Message;
  use ServerErrorStrings;
  use ServerConfig;
  use Cast;
  use BigInteger;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const castLogger = new Logger(logLevel, logChannel);

  proc castMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    const name = msgArgs.getValueOf("name"),
          objtype = msgArgs.getValueOf("objType").toUpper(): ObjType,
          targetDtype = str2dtype(msgArgs.getValueOf("targetDtype")),
          opt = msgArgs.getValueOf("opt");
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
          "name: %s obgtype: %? targetDtype: %? opt: %?".doFormat(
                                                 name,objtype,targetDtype,opt));

    select objtype {
      when ObjType.PDARRAY {
        var gse: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);

        proc doScalarCast(type from, type to): MsgTuple throws
          where isSupportedType(from) && isSupportedType(to)
        {
          const (success, msg) = castGenSymEntry(gse, st, from, to);
          return new MsgTuple(msg, if success then MsgType.NORMAL else MsgType.ERROR);
        }

        proc doScalarCast(type from, type to): MsgTuple throws
          where !isSupportedType(from) || !isSupportedType(to)
        {
          const errorMsg = unsupportedTypeError(from, pn);
          castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        proc doBigintCast(type from): MsgTuple throws
          where isSupportedType(from)
        {
          const (success, msg) = castGenSymEntryToBigInt(gse, st, from);
          return new MsgTuple(msg, if success then MsgType.NORMAL else MsgType.ERROR);
        }

        proc doBigintCast(type from): MsgTuple throws
          where !isSupportedType(from)
        {
          const errorMsg = unsupportedTypeError(from, pn);
          castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        proc doStringCast(type from): MsgTuple throws
          where isSupportedType(from)
        {
          const (success, msg) = castGenSymEntryToString(gse, st, from);
          return new MsgTuple(msg, if success then MsgType.NORMAL else MsgType.ERROR);
        }

        proc doStringCast(type from): MsgTuple throws
          where !isSupportedType(from)
        {
          const errorMsg = unsupportedTypeError(from, pn);
          castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }

        select (gse.dtype, targetDtype) {
            when (DType.Int64, DType.Int8) do return doScalarCast(int, int(8));
            when (DType.Int64, DType.Int16) do return doScalarCast(int, int(16));
            when (DType.Int64, DType.Int32) do return doScalarCast(int, int(32));
            when (DType.Int64, DType.Int64) do return doScalarCast(int, int);
            when (DType.Int64, DType.UInt8) do return doScalarCast(int, uint(8));
            when (DType.Int64, DType.UInt16) do return doScalarCast(int, uint(16));
            when (DType.Int64, DType.UInt32) do return doScalarCast(int, uint(32));
            when (DType.Int64, DType.UInt64) do return doScalarCast(int, uint);
            when (DType.Int64, DType.Float32) do return doScalarCast(int, real(32));
            when (DType.Int64, DType.Float64) do return doScalarCast(int, real(64));
            when (DType.Int64, DType.Complex64) do return doScalarCast(int, complex(64));
            when (DType.Int64, DType.Complex128) do return doScalarCast(int, complex(128));
            when (DType.Int64, DType.Bool) do return doScalarCast(int, bool);
            when (DType.Int64, DType.Strings) do return doStringCast(int);
            when (DType.Int64, DType.BigInt) do return doBigintCast(int);

            when (DType.Int32, DType.Int8) do return doScalarCast(int(32), int(8));
            when (DType.Int32, DType.Int16) do return doScalarCast(int(32), int(16));
            when (DType.Int32, DType.Int32) do return doScalarCast(int(32), int(32));
            when (DType.Int32, DType.Int64) do return doScalarCast(int(32), int(64));
            when (DType.Int32, DType.UInt8) do return doScalarCast(int(32), uint(8));
            when (DType.Int32, DType.UInt16) do return doScalarCast(int(32), uint(16));
            when (DType.Int32, DType.UInt32) do return doScalarCast(int(32), uint(32));
            when (DType.Int32, DType.UInt64) do return doScalarCast(int(32), uint(64));
            when (DType.Int32, DType.Float32) do return doScalarCast(int(32), real(32));
            when (DType.Int32, DType.Float64) do return doScalarCast(int(32), real(64));
            when (DType.Int32, DType.Complex64) do return doScalarCast(int(32), complex(64));
            when (DType.Int32, DType.Complex128) do return doScalarCast(int(32), complex(128));
            when (DType.Int32, DType.Bool) do return doScalarCast(int(32), bool);
            when (DType.Int32, DType.Strings) do return doStringCast(int(32));
            when (DType.Int32, DType.BigInt) do return doBigintCast(int(32));

            when (DType.Int16, DType.Int8) do return doScalarCast(int(16), int(8));
            when (DType.Int16, DType.Int16) do return doScalarCast(int(16), int(16));
            when (DType.Int16, DType.Int32) do return doScalarCast(int(16), int(32));
            when (DType.Int16, DType.Int64) do return doScalarCast(int(16), int(64));
            when (DType.Int16, DType.UInt8) do return doScalarCast(int(16), uint(8));
            when (DType.Int16, DType.UInt16) do return doScalarCast(int(16), uint(16));
            when (DType.Int16, DType.UInt32) do return doScalarCast(int(16), uint(32));
            when (DType.Int16, DType.UInt64) do return doScalarCast(int(16), uint(64));
            when (DType.Int16, DType.Float32) do return doScalarCast(int(16), real(32));
            when (DType.Int16, DType.Float64) do return doScalarCast(int(16), real(64));
            when (DType.Int16, DType.Complex64) do return doScalarCast(int(16), complex(64));
            when (DType.Int16, DType.Complex128) do return doScalarCast(int(16), complex(128));
            when (DType.Int16, DType.Bool) do return doScalarCast(int(16), bool);
            when (DType.Int16, DType.Strings) do return doStringCast(int(16));
            when (DType.Int16, DType.BigInt) do return doBigintCast(int(16));

            when (DType.Int8, DType.Int8) do return doScalarCast(int(8), int(8));
            when (DType.Int8, DType.Int16) do return doScalarCast(int(8), int(16));
            when (DType.Int8, DType.Int32) do return doScalarCast(int(8), int(32));
            when (DType.Int8, DType.Int64) do return doScalarCast(int(8), int);
            when (DType.Int8, DType.UInt8) do return doScalarCast(int(8), uint(8));
            when (DType.Int8, DType.UInt16) do return doScalarCast(int(8), uint(16));
            when (DType.Int8, DType.UInt32) do return doScalarCast(int(8), uint(32));
            when (DType.Int8, DType.UInt64) do return doScalarCast(int(8), uint);
            when (DType.Int8, DType.Float32) do return doScalarCast(int(8), real(32));
            when (DType.Int8, DType.Float64) do return doScalarCast(int(8), real(64));
            when (DType.Int8, DType.Complex64) do return doScalarCast(int(8), complex(64));
            when (DType.Int8, DType.Complex128) do return doScalarCast(int(8), complex(128));
            when (DType.Int8, DType.Bool) do return doScalarCast(int(8), bool);
            when (DType.Int8, DType.Strings) do return doStringCast(int(8));
            when (DType.Int8, DType.BigInt) do return doBigintCast(int(8));

            when (DType.UInt64, DType.Int8) do return doScalarCast(uint, int(8));
            when (DType.UInt64, DType.Int16) do return doScalarCast(uint, int(16));
            when (DType.UInt64, DType.Int32) do return doScalarCast(uint, int(32));
            when (DType.UInt64, DType.Int64) do return doScalarCast(uint, int);
            when (DType.UInt64, DType.UInt8) do return doScalarCast(uint, uint(8));
            when (DType.UInt64, DType.UInt16) do return doScalarCast(uint, uint(16));
            when (DType.UInt64, DType.UInt32) do return doScalarCast(uint, uint(32));
            when (DType.UInt64, DType.UInt64) do return doScalarCast(uint, uint);
            when (DType.UInt64, DType.Float32) do return doScalarCast(uint, real(32));
            when (DType.UInt64, DType.Float64) do return doScalarCast(uint, real(64));
            when (DType.UInt64, DType.Complex64) do return doScalarCast(uint, complex(64));
            when (DType.UInt64, DType.Complex128) do return doScalarCast(uint, complex(128));
            when (DType.UInt64, DType.Bool) do return doScalarCast(uint, bool);
            when (DType.UInt64, DType.Strings) do return doStringCast(uint);
            when (DType.UInt64, DType.BigInt) do return doBigintCast(uint);

            when (DType.UInt32, DType.Int8) do return doScalarCast(uint(32), int(8));
            when (DType.UInt32, DType.Int16) do return doScalarCast(uint(32), int(16));
            when (DType.UInt32, DType.Int32) do return doScalarCast(uint(32), int(32));
            when (DType.UInt32, DType.Int64) do return doScalarCast(uint(32), int(64));
            when (DType.UInt32, DType.UInt8) do return doScalarCast(uint(32), uint(8));
            when (DType.UInt32, DType.UInt16) do return doScalarCast(uint(32), uint(16));
            when (DType.UInt32, DType.UInt32) do return doScalarCast(uint(32), uint(32));
            when (DType.UInt32, DType.UInt64) do return doScalarCast(uint(32), uint(64));
            when (DType.UInt32, DType.Float32) do return doScalarCast(uint(32), real(32));
            when (DType.UInt32, DType.Float64) do return doScalarCast(uint(32), real(64));
            when (DType.UInt32, DType.Complex64) do return doScalarCast(uint(32), complex(64));
            when (DType.UInt32, DType.Complex128) do return doScalarCast(uint(32), complex(128));
            when (DType.UInt32, DType.Bool) do return doScalarCast(uint(32), bool);
            when (DType.UInt32, DType.Strings) do return doStringCast(uint(32));
            when (DType.UInt32, DType.BigInt) do return doBigintCast(uint(32));

            when (DType.UInt16, DType.Int8) do return doScalarCast(uint(16), int(8));
            when (DType.UInt16, DType.Int16) do return doScalarCast(uint(16), int(16));
            when (DType.UInt16, DType.Int32) do return doScalarCast(uint(16), int(32));
            when (DType.UInt16, DType.Int64) do return doScalarCast(uint(16), int(64));
            when (DType.UInt16, DType.UInt8) do return doScalarCast(uint(16), uint(8));
            when (DType.UInt16, DType.UInt16) do return doScalarCast(uint(16), uint(16));
            when (DType.UInt16, DType.UInt32) do return doScalarCast(uint(16), uint(32));
            when (DType.UInt16, DType.UInt64) do return doScalarCast(uint(16), uint(64));
            when (DType.UInt16, DType.Float32) do return doScalarCast(uint(16), real(32));
            when (DType.UInt16, DType.Float64) do return doScalarCast(uint(16), real(64));
            when (DType.UInt16, DType.Complex64) do return doScalarCast(uint(16), complex(64));
            when (DType.UInt16, DType.Complex128) do return doScalarCast(uint(16), complex(128));
            when (DType.UInt16, DType.Bool) do return doScalarCast(uint(16), bool);
            when (DType.UInt16, DType.Strings) do return doStringCast(uint(16));
            when (DType.UInt16, DType.BigInt) do return doBigintCast(uint(16));

            when (DType.UInt8, DType.Int8) do return doScalarCast(uint(8), int(8));
            when (DType.UInt8, DType.Int16) do return doScalarCast(uint(8), int(16));
            when (DType.UInt8, DType.Int32) do return doScalarCast(uint(8), int(32));
            when (DType.UInt8, DType.Int64) do return doScalarCast(uint(8), int);
            when (DType.UInt8, DType.UInt8) do return doScalarCast(uint(8), uint(8));
            when (DType.UInt8, DType.UInt16) do return doScalarCast(uint(8), uint(16));
            when (DType.UInt8, DType.UInt32) do return doScalarCast(uint(8), uint(32));
            when (DType.UInt8, DType.UInt64) do return doScalarCast(uint(8), uint);
            when (DType.UInt8, DType.Float32) do return doScalarCast(uint(8), real(32));
            when (DType.UInt8, DType.Float64) do return doScalarCast(uint(8), real(64));
            when (DType.UInt8, DType.Complex64) do return doScalarCast(uint(8), complex(64));
            when (DType.UInt8, DType.Complex128) do return doScalarCast(uint(8), complex(128));
            when (DType.UInt8, DType.Bool) do return doScalarCast(uint(8), bool);
            when (DType.UInt8, DType.Strings) do return doStringCast(uint(8));
            when (DType.UInt8, DType.BigInt) do return doBigintCast(uint(8));

            when (DType.Float64, DType.Int8) do return doScalarCast(real, int(8));
            when (DType.Float64, DType.Int16) do return doScalarCast(real, int(16));
            when (DType.Float64, DType.Int32) do return doScalarCast(real, int(32));
            when (DType.Float64, DType.Int64) do return doScalarCast(real, int(64));
            when (DType.Float64, DType.UInt8) do return doScalarCast(real, uint(8));
            when (DType.Float64, DType.UInt16) do return doScalarCast(real, uint(16));
            when (DType.Float64, DType.UInt32) do return doScalarCast(real, uint(32));
            when (DType.Float64, DType.UInt64) do return doScalarCast(real, uint(64));
            when (DType.Float64, DType.Float32) do return doScalarCast(real, real(32));
            when (DType.Float64, DType.Float64) do return doScalarCast(real, real(64));
            when (DType.Float64, DType.Complex64) do return doScalarCast(real, complex(64));
            when (DType.Float64, DType.Complex128) do return doScalarCast(real, complex(128));
            when (DType.Float64, DType.Bool) do return doScalarCast(real, bool);
            when (DType.Float64, DType.Strings) do return doStringCast(real);

            when (DType.Float32, DType.Int8) do return doScalarCast(real(32), int(8));
            when (DType.Float32, DType.Int16) do return doScalarCast(real(32), int(16));
            when (DType.Float32, DType.Int32) do return doScalarCast(real(32), int(32));
            when (DType.Float32, DType.Int64) do return doScalarCast(real(32), int(64));
            when (DType.Float32, DType.UInt8) do return doScalarCast(real(32), uint(8));
            when (DType.Float32, DType.UInt16) do return doScalarCast(real(32), uint(16));
            when (DType.Float32, DType.UInt32) do return doScalarCast(real(32), uint(32));
            when (DType.Float32, DType.UInt64) do return doScalarCast(real(32), uint(64));
            when (DType.Float32, DType.Float32) do return doScalarCast(real(32), real(32));
            when (DType.Float32, DType.Float64) do return doScalarCast(real(32), real(64));
            when (DType.Float32, DType.Complex64) do return doScalarCast(real(32), complex(64));
            when (DType.Float32, DType.Complex128) do return doScalarCast(real(32), complex(128));
            when (DType.Float32, DType.Bool) do return doScalarCast(real(32), bool);
            when (DType.Float32, DType.Strings) do return doStringCast(real(32));

            when (DType.Complex128, DType.Int8) do return doScalarCast(complex(128), int(8));
            when (DType.Complex128, DType.Int16) do return doScalarCast(complex(128), int(16));
            when (DType.Complex128, DType.Int32) do return doScalarCast(complex(128), int(32));
            when (DType.Complex128, DType.Int64) do return doScalarCast(complex(128), int(64));
            when (DType.Complex128, DType.UInt8) do return doScalarCast(complex(128), uint(8));
            when (DType.Complex128, DType.UInt16) do return doScalarCast(complex(128), uint(16));
            when (DType.Complex128, DType.UInt32) do return doScalarCast(complex(128), uint(32));
            when (DType.Complex128, DType.UInt64) do return doScalarCast(complex(128), uint(64));
            when (DType.Complex128, DType.Float32) do return doScalarCast(complex(128), real(32));
            when (DType.Complex128, DType.Float64) do return doScalarCast(complex(128), real(64));
            when (DType.Complex128, DType.Complex64) do return doScalarCast(complex(128), complex(64));
            when (DType.Complex128, DType.Complex128) do return doScalarCast(complex(128), complex(128));
            when (DType.Complex128, DType.Bool) do return doScalarCast(complex(128), bool);
            when (DType.Complex128, DType.Strings) do return doStringCast(complex(128));

            when (DType.Complex64, DType.Int8) do return doScalarCast(complex(64), int(8));
            when (DType.Complex64, DType.Int16) do return doScalarCast(complex(64), int(16));
            when (DType.Complex64, DType.Int32) do return doScalarCast(complex(64), int(32));
            when (DType.Complex64, DType.Int64) do return doScalarCast(complex(64), int(64));
            when (DType.Complex64, DType.UInt8) do return doScalarCast(complex(64), uint(8));
            when (DType.Complex64, DType.UInt16) do return doScalarCast(complex(64), uint(16));
            when (DType.Complex64, DType.UInt32) do return doScalarCast(complex(64), uint(32));
            when (DType.Complex64, DType.UInt64) do return doScalarCast(complex(64), uint(64));
            when (DType.Complex64, DType.Float32) do return doScalarCast(complex(64), real(32));
            when (DType.Complex64, DType.Float64) do return doScalarCast(complex(64), real(64));
            when (DType.Complex64, DType.Complex64) do return doScalarCast(complex(64), complex(64));
            when (DType.Complex64, DType.Complex128) do return doScalarCast(complex(64), complex(128));
            when (DType.Complex64, DType.Bool) do return doScalarCast(complex(64), bool);
            when (DType.Complex64, DType.Strings) do return doStringCast(complex(64));

            when (DType.Bool, DType.Int8) do return doScalarCast(bool, int(8));
            when (DType.Bool, DType.Int16) do return doScalarCast(bool, int(16));
            when (DType.Bool, DType.Int32) do return doScalarCast(bool, int(32));
            when (DType.Bool, DType.Int64) do return doScalarCast(bool, int(64));
            when (DType.Bool, DType.UInt8) do return doScalarCast(bool, uint(8));
            when (DType.Bool, DType.UInt16) do return doScalarCast(bool, uint(16));
            when (DType.Bool, DType.UInt32) do return doScalarCast(bool, uint(32));
            when (DType.Bool, DType.UInt64) do return doScalarCast(bool, uint(64));
            when (DType.Bool, DType.Float32) do return doScalarCast(bool, real(32));
            when (DType.Bool, DType.Float64) do return doScalarCast(bool, real(64));
            when (DType.Bool, DType.Complex64) do return doScalarCast(bool, complex(64));
            when (DType.Bool, DType.Complex128) do return doScalarCast(bool, complex(128));
            when (DType.Bool, DType.Bool) do return doScalarCast(bool, bool);
            when (DType.Bool, DType.Strings) do return doStringCast(bool);
            when (DType.Bool, DType.BigInt) do return doBigintCast(bool);

            when (DType.BigInt, DType.UInt8) do return doScalarCast(bigint, uint(8));
            when (DType.BigInt, DType.UInt16) do return doScalarCast(bigint, uint(16));
            when (DType.BigInt, DType.UInt32) do return doScalarCast(bigint, uint(32));
            when (DType.BigInt, DType.UInt64) do return doScalarCast(bigint, uint(64));
            when (DType.BigInt, DType.Int8) do return doScalarCast(bigint, int(8));
            when (DType.BigInt, DType.Int16) do return doScalarCast(bigint, int(16));
            when (DType.BigInt, DType.Int32) do return doScalarCast(bigint, int(32));
            when (DType.BigInt, DType.Int64) do return doScalarCast(bigint, int(64));
            when (DType.BigInt, DType.Float32) do return doScalarCast(bigint, real(32));
            when (DType.BigInt, DType.Float64) do return doScalarCast(bigint, real(64));
            when (DType.BigInt, DType.Complex64) do return doScalarCast(bigint, complex(64));
            when (DType.BigInt, DType.Complex128) do return doScalarCast(bigint, complex(128));
            when (DType.BigInt, DType.Strings) do return doStringCast(bigint);
            when (DType.BigInt, DType.BigInt) do return doBigintCast(bigint);

            otherwise {
                var errorMsg = notImplementedError(pn,gse.dtype,":",targetDtype);
                castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
      }
      when ObjType.STRINGS {
          const strings = getSegString(name, st);
          const errors = opt.toLower() : ErrorMode;

          proc doCastString(type to): MsgTuple throws
            where isSupportedType(to)
          {
            const msg = castStringToSymEntry(strings, st, to, errors);
            return new MsgTuple(msg, MsgType.NORMAL);
          }

          proc doCastString(type to): MsgTuple throws
            where !isSupportedType(to)
          {
            const errorMsg = unsupportedTypeError(to, pn);
            castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
          }

          select targetDtype {
            when DType.Int64 do return doCastString(int);
            when DType.Int32 do return doCastString(int(32));
            when DType.Int16 do return doCastString(int(16));
            when DType.Int8 do return doCastString(int(8));
            when DType.UInt8 do return doCastString(uint(8));
            when DType.UInt16 do return doCastString(uint(16));
            when DType.UInt32 do return doCastString(uint(32));
            when DType.UInt64 do return doCastString(uint);
            when DType.Float64 do return doCastString(real);
            when DType.Float32 do return doCastString(real(32));
            when DType.Complex128 do return doCastString(complex(128));
            when DType.Complex64 do return doCastString(complex(64));
            when DType.Bool do return doCastString(bool);
            when DType.BigInt do {
              const msg = castStringToBigInt(strings, st, errors);
              return new MsgTuple(msg, MsgType.NORMAL);
            }
            when DType.Strings {
              const oname = st.nextName();
              const vname = st.nextName();
              var offsets = st.addEntry(oname, createSymEntry(strings.offsets.a));
              var values = st.addEntry(vname, createSymEntry(strings.values.a));
              return new MsgTuple("created " + st.attrib(oname) + "+created " + st.attrib(vname), MsgType.NORMAL);
            }
            otherwise {
              var errorMsg = notImplementedError(pn,DType.Strings,":",targetDtype);
              castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
            }
          }
      }
      otherwise {
        var errorMsg = notImplementedError(pn,objtype:string);
        castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
  }

  proc transmuteFloatMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    var name = msgArgs.getValueOf("name");
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"name: %s".doFormat(name));
    var e = toSymEntry(getGenericTypedArrayEntry(name, st), real);
    var transmuted = makeDistArray(e.a.domain, uint);
    transmuted = [ei in e.a] ei.transmute(uint(64));
    var transmuteName = st.nextName();
    st.addEntry(transmuteName, createSymEntry(transmuted));
    var repMsg = "created " + st.attrib(transmuteName);
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  use CommandMap;
  registerFunction("cast", castMsg, getModuleName());
  registerFunction("transmuteFloat", transmuteFloatMsg, getModuleName());
}
