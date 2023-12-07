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
    var name = msgArgs.getValueOf("name");
    var objtype = msgArgs.getValueOf("objType").toUpper(): ObjType;
    var targetDtype = msgArgs.getValueOf("targetDtype");
    var opt = msgArgs.getValueOf("opt");
    castLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
          "name: %s obgtype: %? targetDtype: %? opt: %?".doFormat(
                                                 name,objtype,targetDtype,opt));
    select objtype {
      when ObjType.PDARRAY {
        var gse: borrowed GenSymEntry = getGenericTypedArrayEntry(name, st);
        select (gse.dtype, targetDtype) {
            when (DType.Int64, "int8") do return castGenSymEntry(gse, st, int, int(8));
            when (DType.Int64, "int16") do return castGenSymEntry(gse, st, int, int(16));
            when (DType.Int64, "int32") do return castGenSymEntry(gse, st, int, int(32));
            when (DType.Int64, "int64") do return castGenSymEntry(gse, st, int, int);
            when (DType.Int64, "uint8") do return castGenSymEntry(gse, st, int, uint(8));
            when (DType.Int64, "uint16") do return castGenSymEntry(gse, st, int, uint(16));
            when (DType.Int64, "uint32") do return castGenSymEntry(gse, st, int, uint(32));
            when (DType.Int64, "uint64") do return castGenSymEntry(gse, st, int, uint);
            when (DType.Int64, "float32") do return castGenSymEntry(gse, st, int, real(32));
            when (DType.Int64, "float64") do return castGenSymEntry(gse, st, int, real(64));
            when (DType.Int64, "complex64") do return castGenSymEntry(gse, st, int, complex(64));
            when (DType.Int64, "complex128") do return castGenSymEntry(gse, st, int, complex(128));
            when (DType.Int64, "bool") do return castGenSymEntry(gse, st, int, bool);
            when (DType.Int64, "str") do return castGenSymEntryToString(gse, st, int);
            when (DType.Int64, "bigint") do return castGenSymEntryToBigInt(gse, st, int);

            when (DType.Int32, "int8") do return castGenSymEntry(gse, st, int(32), int(8));
            when (DType.Int32, "int16") do return castGenSymEntry(gse, st, int(32), int(16));
            when (DType.Int32, "int32") do return castGenSymEntry(gse, st, int(32), int(32));
            when (DType.Int32, "int64") do return castGenSymEntry(gse, st, int(32), int(64));
            when (DType.Int32, "uint8") do return castGenSymEntry(gse, st, int(32), uint(8));
            when (DType.Int32, "uint16") do return castGenSymEntry(gse, st, int(32), uint(16));
            when (DType.Int32, "uint32") do return castGenSymEntry(gse, st, int(32), uint(32));
            when (DType.Int32, "uint64") do return castGenSymEntry(gse, st, int(32), uint(64));
            when (DType.Int32, "float32") do return castGenSymEntry(gse, st, int(32), real(32));
            when (DType.Int32, "float64") do return castGenSymEntry(gse, st, int(32), real(64));
            when (DType.Int32, "complex64") do return castGenSymEntry(gse, st, int(32), complex(64));
            when (DType.Int32, "complex128") do return castGenSymEntry(gse, st, int(32), complex(128));
            when (DType.Int32, "bool") do return castGenSymEntry(gse, st, int(32), bool);
            when (DType.Int32, "str") do return castGenSymEntryToString(gse, st, int(32));
            when (DType.Int32, "bigint") do return castGenSymEntryToBigInt(gse, st, int(32));

            when (DType.Int16, "int8") do return castGenSymEntry(gse, st, int(16), int(8));
            when (DType.Int16, "int16") do return castGenSymEntry(gse, st, int(16), int(16));
            when (DType.Int16, "int32") do return castGenSymEntry(gse, st, int(16), int(32));
            when (DType.Int16, "int64") do return castGenSymEntry(gse, st, int(16), int(64));
            when (DType.Int16, "uint8") do return castGenSymEntry(gse, st, int(16), uint(8));
            when (DType.Int16, "uint16") do return castGenSymEntry(gse, st, int(16), uint(16));
            when (DType.Int16, "uint32") do return castGenSymEntry(gse, st, int(16), uint(32));
            when (DType.Int16, "uint64") do return castGenSymEntry(gse, st, int(16), uint(64));
            when (DType.Int16, "float32") do return castGenSymEntry(gse, st, int(16), real(32));
            when (DType.Int16, "float64") do return castGenSymEntry(gse, st, int(16), real(64));
            when (DType.Int16, "complex64") do return castGenSymEntry(gse, st, int(16), complex(64));
            when (DType.Int16, "complex128") do return castGenSymEntry(gse, st, int(16), complex(128));
            when (DType.Int16, "bool") do return castGenSymEntry(gse, st, int(16), bool);
            when (DType.Int16, "str") do return castGenSymEntryToString(gse, st, int(16));
            when (DType.Int16, "bigint") do return castGenSymEntryToBigInt(gse, st, int(16));

            when (DType.Int8, "int8") do return castGenSymEntry(gse, st, int(8), int(8));
            when (DType.Int8, "int16") do return castGenSymEntry(gse, st, int(8), int(16));
            when (DType.Int8, "int32") do return castGenSymEntry(gse, st, int(8), int(32));
            when (DType.Int8, "int64") do return castGenSymEntry(gse, st, int(8), int);
            when (DType.Int8, "uint8") do return castGenSymEntry(gse, st, int(8), uint(8));
            when (DType.Int8, "uint16") do return castGenSymEntry(gse, st, int(8), uint(16));
            when (DType.Int8, "uint32") do return castGenSymEntry(gse, st, int(8), uint(32));
            when (DType.Int8, "uint64") do return castGenSymEntry(gse, st, int(8), uint);
            when (DType.Int8, "float32") do return castGenSymEntry(gse, st, int(8), real(32));
            when (DType.Int8, "float64") do return castGenSymEntry(gse, st, int(8), real(64));
            when (DType.Int8, "complex64") do return castGenSymEntry(gse, st, int(8), complex(64));
            when (DType.Int8, "complex128") do return castGenSymEntry(gse, st, int(8), complex(128));
            when (DType.Int8, "bool") do return castGenSymEntry(gse, st, int(8), bool);
            when (DType.Int8, "str") do return castGenSymEntryToString(gse, st, int(8));
            when (DType.Int8, "bigint") do return castGenSymEntryToBigInt(gse, st, int(8));

            when (DType.UInt64, "int8") do return castGenSymEntry(gse, st, uint, int(8));
            when (DType.UInt64, "int16") do return castGenSymEntry(gse, st, uint, int(16));
            when (DType.UInt64, "int32") do return castGenSymEntry(gse, st, uint, int(32));
            when (DType.UInt64, "int64") do return castGenSymEntry(gse, st, uint, int);
            when (DType.UInt64, "uint8") do return castGenSymEntry(gse, st, uint, uint(8));
            when (DType.UInt64, "uint16") do return castGenSymEntry(gse, st, uint, uint(16));
            when (DType.UInt64, "uint32") do return castGenSymEntry(gse, st, uint, uint(32));
            when (DType.UInt64, "uint64") do return castGenSymEntry(gse, st, uint, uint);
            when (DType.UInt64, "float32") do return castGenSymEntry(gse, st, uint, real(32));
            when (DType.UInt64, "float64") do return castGenSymEntry(gse, st, uint, real(64));
            when (DType.UInt64, "complex64") do return castGenSymEntry(gse, st, uint, complex(64));
            when (DType.UInt64, "complex128") do return castGenSymEntry(gse, st, uint, complex(128));
            when (DType.UInt64, "bool") do return castGenSymEntry(gse, st, uint, bool);
            when (DType.UInt64, "str") do return castGenSymEntryToString(gse, st, uint);
            when (DType.UInt64, "bigint") do return castGenSymEntryToBigInt(gse, st, uint);

            when (DType.UInt32, "int8") do return castGenSymEntry(gse, st, uint(32), int(8));
            when (DType.UInt32, "int16") do return castGenSymEntry(gse, st, uint(32), int(16));
            when (DType.UInt32, "int32") do return castGenSymEntry(gse, st, uint(32), int(32));
            when (DType.UInt32, "int64") do return castGenSymEntry(gse, st, uint(32), int(64));
            when (DType.UInt32, "uint8") do return castGenSymEntry(gse, st, uint(32), uint(8));
            when (DType.UInt32, "uint16") do return castGenSymEntry(gse, st, uint(32), uint(16));
            when (DType.UInt32, "uint32") do return castGenSymEntry(gse, st, uint(32), uint(32));
            when (DType.UInt32, "uint64") do return castGenSymEntry(gse, st, uint(32), uint(64));
            when (DType.UInt32, "float32") do return castGenSymEntry(gse, st, uint(32), real(32));
            when (DType.UInt32, "float64") do return castGenSymEntry(gse, st, uint(32), real(64));
            when (DType.UInt32, "complex64") do return castGenSymEntry(gse, st, uint(32), complex(64));
            when (DType.UInt32, "complex128") do return castGenSymEntry(gse, st, uint(32), complex(128));
            when (DType.UInt32, "bool") do return castGenSymEntry(gse, st, uint(32), bool);
            when (DType.UInt32, "str") do return castGenSymEntryToString(gse, st, uint(32));
            when (DType.UInt32, "bigint") do return castGenSymEntryToBigInt(gse, st, uint(32));

            when (DType.UInt16, "int8") do return castGenSymEntry(gse, st, uint(16), int(8));
            when (DType.UInt16, "int16") do return castGenSymEntry(gse, st, uint(16), int(16));
            when (DType.UInt16, "int32") do return castGenSymEntry(gse, st, uint(16), int(32));
            when (DType.UInt16, "int64") do return castGenSymEntry(gse, st, uint(16), int(64));
            when (DType.UInt16, "uint8") do return castGenSymEntry(gse, st, uint(16), uint(8));
            when (DType.UInt16, "uint16") do return castGenSymEntry(gse, st, uint(16), uint(16));
            when (DType.UInt16, "uint32") do return castGenSymEntry(gse, st, uint(16), uint(32));
            when (DType.UInt16, "uint64") do return castGenSymEntry(gse, st, uint(16), uint(64));
            when (DType.UInt16, "float32") do return castGenSymEntry(gse, st, uint(16), real(32));
            when (DType.UInt16, "float64") do return castGenSymEntry(gse, st, uint(16), real(64));
            when (DType.UInt16, "complex64") do return castGenSymEntry(gse, st, uint(16), complex(64));
            when (DType.UInt16, "complex128") do return castGenSymEntry(gse, st, uint(16), complex(128));
            when (DType.UInt16, "bool") do return castGenSymEntry(gse, st, uint(16), bool);
            when (DType.UInt16, "str") do return castGenSymEntryToString(gse, st, uint(16));
            when (DType.UInt16, "bigint") do return castGenSymEntryToBigInt(gse, st, uint(16));

            when (DType.UInt8, "int8") do return castGenSymEntry(gse, st, uint(8), int(8));
            when (DType.UInt8, "int16") do return castGenSymEntry(gse, st, uint(8), int(16));
            when (DType.UInt8, "int32") do return castGenSymEntry(gse, st, uint(8), int(32));
            when (DType.UInt8, "int64") do return castGenSymEntry(gse, st, uint(8), int);
            when (DType.UInt8, "uint8") do return castGenSymEntry(gse, st, uint(8), uint(8));
            when (DType.UInt8, "uint16") do return castGenSymEntry(gse, st, uint(8), uint(16));
            when (DType.UInt8, "uint32") do return castGenSymEntry(gse, st, uint(8), uint(32));
            when (DType.UInt8, "uint64") do return castGenSymEntry(gse, st, uint(8), uint);
            when (DType.UInt8, "float32") do return castGenSymEntry(gse, st, uint(8), real(32));
            when (DType.UInt8, "float64") do return castGenSymEntry(gse, st, uint(8), real(64));
            when (DType.UInt8, "complex64") do return castGenSymEntry(gse, st, uint(8), complex(64));
            when (DType.UInt8, "complex128") do return castGenSymEntry(gse, st, uint(8), complex(128));
            when (DType.UInt8, "bool") do return castGenSymEntry(gse, st, uint(8), bool);
            when (DType.UInt8, "str") do return castGenSymEntryToString(gse, st, uint(8));
            when (DType.UInt8, "bigint") do return castGenSymEntryToBigInt(gse, st, uint(8));

            when (DType.Float64, "int8") do return castGenSymEntry(gse, st, real, int(8));
            when (DType.Float64, "int16") do return castGenSymEntry(gse, st, real, int(16));
            when (DType.Float64, "int32") do return castGenSymEntry(gse, st, real, int(32));
            when (DType.Float64, "int64") do return castGenSymEntry(gse, st, real, int(64));
            when (DType.Float64, "uint8") do return castGenSymEntry(gse, st, real, uint(8));
            when (DType.Float64, "uint16") do return castGenSymEntry(gse, st, real, uint(16));
            when (DType.Float64, "uint32") do return castGenSymEntry(gse, st, real, uint(32));
            when (DType.Float64, "uint64") do return castGenSymEntry(gse, st, real, uint(64));
            when (DType.Float64, "float32") do return castGenSymEntry(gse, st, real, real(32));
            when (DType.Float64, "float64") do return castGenSymEntry(gse, st, real, real(64));
            when (DType.Float64, "complex64") do return castGenSymEntry(gse, st, real, complex(64));
            when (DType.Float64, "complex128") do return castGenSymEntry(gse, st, real, complex(128));
            when (DType.Float64, "bool") do return castGenSymEntry(gse, st, real, bool);
            when (DType.Float64, "str") do return castGenSymEntryToString(gse, st, real);

            when (DType.Float32, "int8") do return castGenSymEntry(gse, st, real(32), int(8));
            when (DType.Float32, "int16") do return castGenSymEntry(gse, st, real(32), int(16));
            when (DType.Float32, "int32") do return castGenSymEntry(gse, st, real(32), int(32));
            when (DType.Float32, "int64") do return castGenSymEntry(gse, st, real(32), int(64));
            when (DType.Float32, "uint8") do return castGenSymEntry(gse, st, real(32), uint(8));
            when (DType.Float32, "uint16") do return castGenSymEntry(gse, st, real(32), uint(16));
            when (DType.Float32, "uint32") do return castGenSymEntry(gse, st, real(32), uint(32));
            when (DType.Float32, "uint64") do return castGenSymEntry(gse, st, real(32), uint(64));
            when (DType.Float32, "float32") do return castGenSymEntry(gse, st, real(32), real(32));
            when (DType.Float32, "float64") do return castGenSymEntry(gse, st, real(32), real(64));
            when (DType.Float32, "complex64") do return castGenSymEntry(gse, st, real(32), complex(64));
            when (DType.Float32, "complex128") do return castGenSymEntry(gse, st, real(32), complex(128));
            when (DType.Float32, "bool") do return castGenSymEntry(gse, st, real(32), bool);
            when (DType.Float32, "str") do return castGenSymEntryToString(gse, st, real(32));

            when (DType.Complex128, "int8") do return castGenSymEntry(gse, st, complex(128), int(8));
            when (DType.Complex128, "int16") do return castGenSymEntry(gse, st, complex(128), int(16));
            when (DType.Complex128, "int32") do return castGenSymEntry(gse, st, complex(128), int(32));
            when (DType.Complex128, "int64") do return castGenSymEntry(gse, st, complex(128), int(64));
            when (DType.Complex128, "uint8") do return castGenSymEntry(gse, st, complex(128), uint(8));
            when (DType.Complex128, "uint16") do return castGenSymEntry(gse, st, complex(128), uint(16));
            when (DType.Complex128, "uint32") do return castGenSymEntry(gse, st, complex(128), uint(32));
            when (DType.Complex128, "uint64") do return castGenSymEntry(gse, st, complex(128), uint(64));
            when (DType.Complex128, "float32") do return castGenSymEntry(gse, st, complex(128), real(32));
            when (DType.Complex128, "float64") do return castGenSymEntry(gse, st, complex(128), real(64));
            when (DType.Complex128, "complex64") do return castGenSymEntry(gse, st, complex(128), complex(64));
            when (DType.Complex128, "complex128") do return castGenSymEntry(gse, st, complex(128), complex(128));
            when (DType.Complex128, "bool") do return castGenSymEntry(gse, st, complex(128), bool);
            when (DType.Complex128, "str") do return castGenSymEntryToString(gse, st, complex(128));

            when (DType.Complex64, "int8") do return castGenSymEntry(gse, st, complex(64), int(8));
            when (DType.Complex64, "int16") do return castGenSymEntry(gse, st, complex(64), int(16));
            when (DType.Complex64, "int32") do return castGenSymEntry(gse, st, complex(64), int(32));
            when (DType.Complex64, "int64") do return castGenSymEntry(gse, st, complex(64), int(64));
            when (DType.Complex64, "uint8") do return castGenSymEntry(gse, st, complex(64), uint(8));
            when (DType.Complex64, "uint16") do return castGenSymEntry(gse, st, complex(64), uint(16));
            when (DType.Complex64, "uint32") do return castGenSymEntry(gse, st, complex(64), uint(32));
            when (DType.Complex64, "uint64") do return castGenSymEntry(gse, st, complex(64), uint(64));
            when (DType.Complex64, "float32") do return castGenSymEntry(gse, st, complex(64), real(32));
            when (DType.Complex64, "float64") do return castGenSymEntry(gse, st, complex(64), real(64));
            when (DType.Complex64, "complex64") do return castGenSymEntry(gse, st, complex(64), complex(64));
            when (DType.Complex64, "complex128") do return castGenSymEntry(gse, st, complex(64), complex(128));
            when (DType.Complex64, "bool") do return castGenSymEntry(gse, st, complex(64), bool);
            when (DType.Complex64, "str") do return castGenSymEntryToString(gse, st, complex(64));

            when (DType.Bool, "int8") do return castGenSymEntry(gse, st, bool, int(8));
            when (DType.Bool, "int16") do return castGenSymEntry(gse, st, bool, int(16));
            when (DType.Bool, "int32") do return castGenSymEntry(gse, st, bool, int(32));
            when (DType.Bool, "int64") do return castGenSymEntry(gse, st, bool, int(64));
            when (DType.Bool, "uint8") do return castGenSymEntry(gse, st, bool, uint(8));
            when (DType.Bool, "uint16") do return castGenSymEntry(gse, st, bool, uint(16));
            when (DType.Bool, "uint32") do return castGenSymEntry(gse, st, bool, uint(32));
            when (DType.Bool, "uint64") do return castGenSymEntry(gse, st, bool, uint(64));
            when (DType.Bool, "float32") do return castGenSymEntry(gse, st, bool, real(32));
            when (DType.Bool, "float64") do return castGenSymEntry(gse, st, bool, real(64));
            when (DType.Bool, "complex64") do return castGenSymEntry(gse, st, bool, complex(64));
            when (DType.Bool, "complex128") do return castGenSymEntry(gse, st, bool, complex(128));
            when (DType.Bool, "bool") do return castGenSymEntry(gse, st, bool, bool);
            when (DType.Bool, "str") do return castGenSymEntryToString(gse, st, bool);
            when (DType.Bool, "bigint") do return castGenSymEntryToBigInt(gse, st, bool);

            when (DType.BigInt, "uint8") do return castGenSymEntry(gse, st, bigint, uint(8));
            when (DType.BigInt, "uint16") do return castGenSymEntry(gse, st, bigint, uint(16));
            when (DType.BigInt, "uint32") do return castGenSymEntry(gse, st, bigint, uint(32));
            when (DType.BigInt, "uint64") do return castGenSymEntry(gse, st, bigint, uint(64));
            when (DType.BigInt, "int8") do return castGenSymEntry(gse, st, bigint, int(8));
            when (DType.BigInt, "int16") do return castGenSymEntry(gse, st, bigint, int(16));
            when (DType.BigInt, "int32") do return castGenSymEntry(gse, st, bigint, int(32));
            when (DType.BigInt, "int64") do return castGenSymEntry(gse, st, bigint, int(64));
            when (DType.BigInt, "float32") do return castGenSymEntry(gse, st, bigint, real(32));
            when (DType.BigInt, "float64") do return castGenSymEntry(gse, st, bigint, real(64));
            when (DType.BigInt, "complex64") do return castGenSymEntry(gse, st, bigint, complex(64));
            when (DType.BigInt, "complex128") do return castGenSymEntry(gse, st, bigint, complex(128));
            when (DType.BigInt, "str") do return castGenSymEntryToString(gse, st, bigint);
            when (DType.BigInt, "bigint") do return castGenSymEntryToBigInt(gse, st, bigint);

            otherwise {
                var errorMsg = notImplementedError(pn,gse.dtype:string,":",targetDtype);
                castLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
      }
      when ObjType.STRINGS {
          const strings = getSegString(name, st);
          const errors = opt.toLower() : ErrorMode;
          select targetDtype {
            when "int64" do return castStringToSymEntry(strings, st, int, errors);
            when "int32" do return castStringToSymEntry(strings, st, int(32), errors);
            when "int16" do return castStringToSymEntry(strings, st, int(16), errors);
            when "int8" do return castStringToSymEntry(strings, st, int(8), errors);
            when "uint8" do return castStringToSymEntry(strings, st, uint(8), errors);
            when "uint16" do return castStringToSymEntry(strings, st, uint(16), errors);
            when "uint32" do return castStringToSymEntry(strings, st, uint(32), errors);
            when "uint64" do return castStringToSymEntry(strings, st, uint, errors);
            when "float64" do return castStringToSymEntry(strings, st, real, errors);
            when "float32" do return castStringToSymEntry(strings, st, real(32), errors);
            when "complex64" do return castStringToSymEntry(strings, st, complex(64), errors);
            when "complex128" do return castStringToSymEntry(strings, st, complex(128), errors);
            when "bool" do return castStringToSymEntry(strings, st, bool, errors);
            when "bigint" do return castStringToBigInt(strings, st, errors);
            when "str" {
              const oname = st.nextName();
              const vname = st.nextName();
              var offsets = st.addEntry(oname, createSymEntry(strings.offsets.a));
              var values = st.addEntry(vname, createSymEntry(strings.values.a));
              return new MsgTuple("created " + st.attrib(oname) + "+created " + st.attrib(vname), MsgType.NORMAL);
            }
            otherwise {
              var errorMsg = notImplementedError(pn,"str",":",targetDtype);
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

  use CommandMap;
  registerFunction("cast", castMsg, getModuleName());
}
