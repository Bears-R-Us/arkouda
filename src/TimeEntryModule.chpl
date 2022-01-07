module TimeEntryModule {
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

  // Defines supported operations with time-related dtypes

  private const supportedTimeOps =
      [(DType.Datetime64, '==', DType.Datetime64) => DType.Bool,
       (DType.Datetime64, '!=', DType.Datetime64) => DType.Bool,
       (DType.Datetime64, '<', DType.Datetime64) => DType.Bool,
       (DType.Datetime64, '<=', DType.Datetime64) => DType.Bool,
       (DType.Datetime64, '>', DType.Datetime64) => DType.Bool,
       (DType.Datetime64, '>=', DType.Datetime64) => DType.Bool,
       (DType.Datetime64, '-', DType.Datetime64) => DType.Timedelta64,
       (DType.Datetime64, '+', DType.Timedelta64) => DType.Datetime64,
       (DType.Datetime64, '-', DType.Timedelta64) => DType.Datetime64,
       (DType.Datetime64, '/', DType.Timedelta64) => DType.Float64,
       (DType.Datetime64, '//', DType.Timedelta64) => DType.Int64,
       (DType.Datetime64, '%', DType.Timedelta64) => DType.Timedelta64,
       (DType.Datetime64, '+=', DType.Timedelta64) => DType.Datetime64,
       (DType.Datetime64, '-=', DType.Timedelta64) => DType.Datetime64,
       (DType.Timedelta64, '==', DType.Timedelta64) => DType.Bool,
       (DType.Timedelta64, '!=', DType.Timedelta64) => DType.Bool,
       (DType.Timedelta64, '<', DType.Timedelta64) => DType.Bool,
       (DType.Timedelta64, '<=', DType.Timedelta64) => DType.Bool,
       (DType.Timedelta64, '>', DType.Timedelta64) => DType.Bool,
       (DType.Timedelta64, '>=', DType.Timedelta64) => DType.Bool,
       (DType.Timedelta64, '+', DType.Timedelta64) => DType.Timedelta64,
       (DType.Timedelta64, '+=', DType.Timedelta64) => DType.Timedelta64,
       (DType.Timedelta64, '-', DType.Timedelta64) => DType.Timedelta64,
       (DType.Timedelta64, '-=', DType.Timedelta64) => DType.Timedelta64,
       (DType.Timedelta64, '/', DType.Timedelta64) => DType.Float64,
       (DType.Timedelta64, '//', DType.Timedelta64) => DType.Int64,
       (DType.Timedelta64, '%', DType.Timedelta64) => DType.Timedelta64,
       (DType.Timedelta64, '%=', DType.Timedelta64) => DType.Timedelta64,
       (DType.Timedelta64, '+', DType.Datetime64) => DType.Datetime64,
       (DType.Timedelta64, '*', DType.Int64) => DType.Timedelta64,
       (DType.Timedelta64, '*=', DType.Int64) => DType.Timedelta64,
       (DType.Timedelta64, '/', DType.Int64) => DType.Timedelta64,
       // (DType.Timedelta64, '/=', DType.Int64) => DType.Timedelta64,
       // (DType.Timedelta64, '//', DType.Int64) => DType.Int64,
       (DType.Timedelta64, '*', DType.Float64) => DType.Timedelta64,
       (DType.Timedelta64, '*=', DType.Float64) => DType.Timedelta64,
       (DType.Timedelta64, '/', DType.Float64) => DType.Timedelta64,
       (DType.Timedelta64, '/=', DType.Float64) => DType.Timedelta64,
       // (DType.Timedelta64, '//', DType.Float64) => DType.Float64,
       (DType.Int64, '*', DType.Timedelta64) => DType.Timedelta64,
       (DType.Float64, '*', DType.Timedelta64) => DType.Timedelta64];
    
  /* 
   * Get the return dtype of "<leftDType> <op> <rightDType>". If not supported, 
   * throw an error. If neither type is time-related, return UNDEF.
   */
  proc checkTimeOpCompatibility(leftDType: DType, op: string, rightDType: DType): DType throws {
    if (leftDType == DType.Datetime64) || (leftDType == DType.Timedelta64) || (rightDType == DType.Datetime64) || (rightDType == DType.Timedelta64) {
      if supportedTimeOps.domain.contains((leftDType, op, rightDType)) {
        return supportedTimeOps[(leftDType, op, rightDType)];
      } else {
        var errorMsg = "Error: operation (%s %s %s) not implemented".format(leftDType:string, rightDType:string, op);
        tcLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
        throw new Error(errorMsg);
      }
    } else {
      return DType.UNDEF;
    }
  }

  proc SymTab.addTimeEntry(name: string, len: int, dtype: DType): borrowed TimeEntry throws {
    var entry = new shared TimeEntry(len, dtype);
    this.addEntry(name, entry);
    // The lookup is necessary because if the return of this.addEntry is used,
    // the compiler throws an error about being unable to return a scoped variable
    var abstract = this.lookup(name);
    var se = toSymEntry(toGenSymEntry(abstract), int);
    return se: borrowed TimeEntry(int);
  }

  proc getTimeEntry(name, st: borrowed SymTab): borrowed TimeEntry throws {
    var abstract = st.lookup(name);
    if !abstract.isAssignableTo(SymbolEntryType.TimeEntry) {
      var errorMsg = "Error: SymbolEntryType %s is not assignable to TimeEntry".format(abstract.entryType);
        tcLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
        throw new Error(errorMsg);
    }
    var se = toSymEntry(toGenSymEntry(abstract), int);
    return se: borrowed TimeEntry(int);
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