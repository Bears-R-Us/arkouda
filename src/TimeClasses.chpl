module TimeClasses {
  use MultiTypeSymEntry;
  use MultiTypeSymbolTable;
  use Reflection;
  use Logging;
  use ServerConfig;

  private config const logLevel = ServerConfig.logLevel;
  const tcLogger = new Logger(logLevel);

  enum TimeUnit {
    Weeks,
    Days,
    Hours,
    Minutes,
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds
  }
  
  private const get_factor = [TimeUnit.Weeks => 7*24*60*60*10**9,
                              TimeUnit.Days => 24*60*60*10**9,
                              TimeUnit.Hours => 60*60*10**9,
                              TimeUnit.Minutes => 60*10**9,
                              TimeUnit.Seconds => 10**9,
                              TimeUnit.Milliseconds => 10**6,
                              TimeUnit.Microseconds => 10**3,
                              TimeUnit.Nanoseconds => 1];
  
  proc getTimeEntry(name: string, st: borrowed SymTab): borrowed TimeEntry throws {
    var abstractEntry = st.lookup(name);
    if !abstractEntry.isAssignableTo(SymbolEntryType.TimeEntry) {
      var errorMsg = "Error: Cannot interpret %s as TimeEntry".format(abstractEntry.entryType);
      tcLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
      throw new Error(errorMsg);
    }
    var entry:TimeEntry = abstractEntry: borrowed TimeEntry;
    return entry;
  }

  class TimeEntry:AbstractSymEntry {
    const entryType = SymbolEntryType.TimeEntry;
    const dtype: DType;
    const itemsize = 8;
    var size: int = 0;
    var ndim: int = 1;
    var shape: 1*int = (0,);
    var unit: TimeUnit = TimeUnit.Nanoseconds;
    var factor: int = 1;
    var aD: makeDistDom(size).type;
    var a: [aD] int;
    
    proc init(len: int = 0, dtype: DType, unit: TimeUnit = TimeUnit.Nanoseconds) {
      this.entryType = SymbolEntryType.TimeEntry;
      this.assignableTypes.add(this.entryType);
      this.assignableTypes.add(SymbolEntryType.ComplexTypedArraySymEntry);
      this.dtype = dtype;
      this.size = len;
      this.shape = (len,);
      this.unit = unit;
      this.factor = get_factor(unit);
      this.aD = makeDistDom(len);
    }

    proc init(array: [?D] int, dtype: DType, unit: TimeUnit = TimeUnit.Nanoseconds) {
      this.entryType = SymbolEntryType.TimeEntry;
      this.assignableTypes.add(this.entryType);
      this.assignableTypes.add(SymbolEntryType.ComplexTypedArraySymEntry);
      this.dtype = dtype;
      this.size = D.size;
      this.shape = (D.size,);
      this.unit = unit;
      this.factor = get_factor(unit);
      this.aD = D;
      // Store as a datetime64[ns] array
      this.a = this.factor * array;
    }

    proc postinit() throws {
      if (this.dtype != DType.Datetime64) && (this.dtype != DType.Timedelta64) {
        var errorMsg = "Error: dtype must be Datetime64 or Timedelta64, not %s".format(this.dtype);
        tcLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
        throw new Error(errorMsg);
      }
    }

    override proc getSizeEstimate(): int {
      return this.size * this.itemsize;
    }

    proc floor(freq: TimeUnit) {
      const f = get_factor(freq);
      var newa = this.a / f;
      return new shared TimeEntry(newa, this.dtype, unit=freq);
    }

    proc ceil(freq: TimeUnit) {
      const f = get_factor(freq);
      var newa = (this.a + (f - 1)) / f;
      return new shared TimeEntry(newa, this.dtype, unit=freq);
    }

    proc round(freq: TimeUnit) {
      const f = get_factor(freq);
      var newa: [this.aD] int;
      forall (x, y) in zip(this.a, newa) {
        const offset = x + ((f + 1) / 2);
        const rounded = offset / f;
        /* Halfway values should round to nearest even integer.
         * Halfway values are multiples of f, so if one of those
         * gets rounded to an odd number, decrement it.
         */
        if ((offset % f) == 0) && ((rounded % 2) == 1) {
          y = rounded - 1;
        } else {
          y = rounded;
        }
      }
      return new shared TimeEntry(newa, this.dtype, unit=freq);
    }
  }

}