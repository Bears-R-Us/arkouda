module TimeClassMsg {
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use BinOp;

    use ArkoudaDateTimeCompat;
    use Map;

    private config const logLevel = ServerConfig.logLevel;
    const tLogger = new Logger(logLevel);

    // The first 13 entries give the month days elapsed as of the first of month N
    // (or the total number of days in the year for N=13) in non-leap years.
    // The remaining 13 entries give the days elapsed in leap years.
    const MONTHOFFSET = [
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365,
        0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366
    ];

    // To get UNITS do (values // (PROD_PREVIOUS_FACTORS)) % CURRENT_FACTOR
    // for example seconds = (values // (1000*1000*1000)) % 60
    const FACTORS = [
        1000,  // nanosecond
        1000,  // microsecond
        1000,  // millisecond
        60,  // second
        60,  // minute
        24,  // hour
        7,  // day
    ];

    const UNITS = ["nanosecond", "microsecond", "millisecond", "second", "minute", "hour", "day"];

    proc dateTimeAttributesMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var values: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("values"), st);
        var valuesEntry = toSymEntry(values, int);
        var attributesDict = simpleAttributesHelper(valuesEntry.a, st);

        const valDom = valuesEntry.a.domain;
        var year: [valDom] int;
        var month: [valDom] int;
        var day: [valDom] int;
        var isoYear: [valDom] int;
        var weekOfYear: [valDom] int;
        var dayOfWeek: [valDom] int;
        forall (v, y, m, d, iso_y, woy, dow) in zip(valuesEntry.a, year, month, day, isoYear, weekOfYear, dayOfWeek) {
            // convert to seconds and create date
            var t = date.fromTimestamp(floorDivisionHelper(v, 10**9):int);
            (y, m, d, (iso_y, woy, dow)) = (t.year, t.month, t.day, t.isoCalendar());
            dow -= 1;
        }
        const is_leap_year: [valDom] bool = isLeapYear(year);
        const dayOfYear: [valDom] int = MONTHOFFSET[is_leap_year * 13 + month - 1] + day;

        var retname = st.nextName();
        st.addEntry(retname, new shared SymEntry(day));
        attributesDict.addOrSet("day", "created %s".format(st.attrib(retname)));
        retname = st.nextName();
        st.addEntry(retname, new shared SymEntry(month));
        attributesDict.add("month", "created %s".format(st.attrib(retname)));
        retname = st.nextName();
        st.addEntry(retname, new shared SymEntry(year));
        attributesDict.add("year", "created %s".format(st.attrib(retname)));
        retname = st.nextName();
        st.addEntry(retname, new shared SymEntry(is_leap_year));
        attributesDict.add("isLeapYear", "created %s".format(st.attrib(retname)));
        retname = st.nextName();
        st.addEntry(retname, new shared SymEntry(dayOfYear));
        attributesDict.add("dayOfYear", "created %s".format(st.attrib(retname)));
        retname = st.nextName();
        st.addEntry(retname, new shared SymEntry(isoYear));
        attributesDict.add("isoYear", "created %s".format(st.attrib(retname)));
        retname = st.nextName();
        st.addEntry(retname, new shared SymEntry(weekOfYear));
        attributesDict.add("weekOfYear", "created %s".format(st.attrib(retname)));
        retname = st.nextName();
        st.addEntry(retname, new shared SymEntry(dayOfWeek));
        attributesDict.add("dayOfWeek", "created %s".format(st.attrib(retname)));

        var repMsg: string = "%jt".format(attributesDict);

        tLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc timeDeltaAttributesMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        var values: borrowed GenSymEntry = getGenericTypedArrayEntry(msgArgs.getValueOf("values"), st);
        var repMsg: string = "%jt".format(simpleAttributesHelper(toSymEntry(values, int).a, st));

        tLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    proc simpleAttributesHelper(values: [?aD] ?t, st: borrowed SymTab): map throws {
        var attributesDict = new map(keyType=string, valType=string);
        var denominator = 1;
        for (u, f) in zip(UNITS, FACTORS) {
            // handles all the attributes in UNITS
            // UNIT = (values // PROD_PREVIOUS_FACTORS) % CURRENT_FACTOR
            var retname = st.nextName();
            var e = st.addEntry(retname, aD.size, int);
            if u != "day" {
                [(ei,vi) in zip(e.a,values)] ei = floorDivisionHelper(vi, denominator):int % f;
            }
            else {
                // "day" is not modded, because it's the last unit
                [(ei,vi) in zip(e.a,values)] ei = floorDivisionHelper(vi, denominator):int;
            }
            attributesDict.add(u, "created %s".format(st.attrib(retname)));
            denominator *= f;  //denominator is product of previous factors
        }
        return attributesDict;
    }

    use CommandMap;
    registerFunction("dateTimeAttributes",  dateTimeAttributesMsg, getModuleName());
    registerFunction("timeDeltaAttributes",  timeDeltaAttributesMsg, getModuleName());
}
