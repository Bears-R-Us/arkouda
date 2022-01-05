module TimeClasses {
  use MultiTypeSymEntry;
  use MultiTypeSymbolTable;
  use Reflection;
  use Logging;
  use ServerConfig;

  private config const logLevel = ServerConfig.logLevel;
  const tcLogger = new Logger(logLevel);

  // Supported units for time objects
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

  // Factors for converting units to nanoseconds (pandas standard)
  private const get_factor = [TimeUnit.Weeks => 7*24*60*60*10**9,
                              TimeUnit.Days => 24*60*60*10**9,
                              TimeUnit.Hours => 60*60*10**9,
                              TimeUnit.Minutes => 60*10**9,
                              TimeUnit.Seconds => 10**9,
                              TimeUnit.Milliseconds => 10**6,
                              TimeUnit.Microseconds => 10**3,
                              TimeUnit.Nanoseconds => 1];

  /*
   * Shortcut to get a TimeEntry directly from the symbol table by name.
   */
  proc getTimeEntry(name: string, st: borrowed SymTab): borrowed TimeEntry throws {
    var abstractEntry = st.lookup(name);
    if !abstractEntry.isAssignableTo(SymbolEntryType.TimeEntry) {
      var errorMsg = "Error: Cannot interpret %s as TimeEntry".format(abstractEntry.entryType);
      tcLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
      throw new Error(errorMsg);
    }
    var entry = toSymEntry(toGenSymEntry(abstractEntry), int): borrowed TimeEntry(int);
    return entry;
  }

  /*
   * Both Datetime and Timedelta are specializations of SymEntry(int). They store
   * the same data as a normal Int64 array, but with three differences:
   *   - They have units (e.g. nanoseconds)
   *   - They have restrictions on which arithmetic ops they support with other arrays
   *   - They have special methods
   * This class inherits from SymEntry and supplies the units, a time-related dtype
   * (Datetime64 or Timedelta64), and the special methods. This inheritance allows
   * for the reuse of a lot of Int64 code.
   */
  class TimeEntry: SymEntry {
    var unit: TimeUnit = TimeUnit.Nanoseconds;
    var factor: int = 1;
    
    proc init(len: int = 0, dtype: DType, unit: TimeUnit = TimeUnit.Nanoseconds) {
      super.init(len, int);
      this.entryType = SymbolEntryType.TimeEntry;
      this.assignableTypes.add(SymbolEntryType.TimeEntry);
      this.dtype = dtype;
      this.unit = unit;
      this.factor = get_factor(unit);
    }

    proc init(array: [?D] int, dtype: DType, unit: TimeUnit = TimeUnit.Nanoseconds) {
      super.init(array);
      this.entryType = SymbolEntryType.TimeEntry;
      this.assignableTypes.add(SymbolEntryType.TimeEntry);
      this.dtype = dtype;
      this.unit = unit;
      this.factor = get_factor(unit);
      // The underlying data should always correspond to a datetime64[ns] array,
      // following the implementation of pandas.
      if (this.factor != 1) {
        this.a *= this.factor;
      }
    }

    proc postinit() throws {
      // init methods cannot throw, so add dtype error handling to postinit
      if (this.dtype != DType.Datetime64) && (this.dtype != DType.Timedelta64) {
        var errorMsg = "Error: dtype must be Datetime64 or Timedelta64, not %s".format(this.dtype);
        tcLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
        throw new Error(errorMsg);
      }
    }

    /*
     * Round times downwards to the nearest specified unit.
     */
    proc floor(freq: TimeUnit) {
      const f = get_factor(freq);
      var newa = this.a / f;
      return new shared TimeEntry(newa, this.dtype, unit=freq);
    }
    
    /*
     * Round times upwards to the nearest specified unit.
     */
    proc ceil(freq: TimeUnit) {
      const f = get_factor(freq);
      var newa = (this.a + (f - 1)) / f;
      return new shared TimeEntry(newa, this.dtype, unit=freq);
    }

    /*
     * Round times to the nearest specified unit (values exactly
     * halfway between round to the nearest even unit).
     */
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