module StatsMsg {
    use ServerConfig;

    use AryUtil;
    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use Stats;
    use IOUtils;
    use List;
    use Map;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const sLogger = new Logger(logLevel, logChannel);

    @arkouda.registerCommand()
    proc meanAll(const ref x: [?d] ?t, skipNan: bool): real throws {
      if canBeNan(t) && skipNan
          then return meanSkipNan(x, d);
          else return (+ reduce x:real) / x.size:real;
    }

    @arkouda.registerCommand()
    proc mean(const ref x: [?d] ?t, skipNan: bool, axis: list(int)): [] real throws {
      const (valid, axes_) = validateNegativeAxes (axis, d.rank);
      if !valid {
        throw new Error("Invalid axis value(s) '%?' in mean reduction".format(axis));
      } else {
        var meanArr = makeDistArray(domOffAxis(d, axes_), real);
        forall (slice, sliceIdx) in axisSlices(d, axes_) {
            if canBeNan(t) && skipNan
            then meanArr[sliceIdx] = meanSkipNan(x, slice);
            else meanArr[sliceIdx] = meanOver(x, slice);
        }
        return meanArr;
      }
    }

    @arkouda.registerCommand()
    proc varAll(const ref x: [?d] ?t, skipNan: bool, ddof: real): real throws {
      if canBeNan(t) && skipNan
        then return varianceSkipNan(x, d, ddof);
        else return variance(x, ddof);
    }

    // Note that this proc is named varReduce instead of just "var", because "var"
    // is a reserved Chapel keyword.  Because var and std are part of the same
    // reduction grouping python-side, the corresponding std function is also
    // named stdReduce.

    @arkouda.registerCommand()
    proc varReduce(const ref x: [?d] ?t, skipNan: bool, ddof: real, axis: list(int)): [] real throws {
      const (valid, axes_) = validateNegativeAxes (axis, d.rank);
      if !valid {
        throw new Error("Invalid axis value(s) '%?' in var reduction".format(axis));
      } else {
        var varArr = makeDistArray(domOffAxis(d, axes_), real);
        forall (slice, sliceIdx) in axisSlices(d, axes_) {
            if canBeNan(t) && skipNan
              then varArr[sliceIdx] = varianceSkipNan(x, slice, ddof);
              else varArr[sliceIdx] = varianceOver(x, slice, ddof);
        }
        return varArr;
      }
    }

    @arkouda.registerCommand()
    proc stdAll(const ref x: [?d] ?t, skipNan: bool, ddof: real): real throws {
      if canBeNan(t) && skipNan
        then return stdSkipNan(x, d, ddof);
        else return std(x, ddof);
    }

    @arkouda.registerCommand()
    proc stdReduce(const ref x: [?d] ?t, skipNan: bool, ddof: real, axis: list(int)): [] real throws {
      const (valid, axes_) = validateNegativeAxes (axis, d.rank);
      if !valid {
        throw new Error("Invalid axis value(s) '%?' in std reduction".format(axis));
      } else {
        var stdArr = makeDistArray(domOffAxis(d, axes_), real);
        forall (slice, sliceIdx) in axisSlices(d, axes_) {
            if canBeNan(t) && skipNan
                then stdArr[sliceIdx] = stdSkipNan(x, slice, ddof);
                else stdArr[sliceIdx] = stdOver(x, slice, ddof);
        }
        return stdArr;
      }
    }

    @arkouda.registerCommand()
    proc cov(const ref x: [?dx] ?tx, const ref y: [?dy] ?ty): real throws
      where dx.rank == dy.rank
    {
      if dx.shape != dy.shape then
        throw new Error("x and y must have the same shape");

      const mx = mean(x),
            my = mean(y);

      return (+ reduce ((x:real - mx) * (y:real - my))) / (dx.size - 1):real;
    }

    @arkouda.registerCommand()
    proc corr(const ref x: [?dx] ?tx, const ref y: [?dy] ?ty): real throws
      where dx.rank == dy.rank
    {
      if dx.shape != dy.shape then
        throw new Error("x and y must have the same shape");

      const mx = mean(x),
            my = mean(y);

      return cov(x, y) / (std(x, 1) * std(y, 1));
    }

    @arkouda.registerCommand()
    proc cumSum(const ref x: [?d] ?t, axis: int, includeInitial: bool): [] t throws {
      if d.rank == 1 {
        var cs = makeDistArray(if includeInitial then x.size+1 else x.size, t);
        cs[if includeInitial then 1.. else 0..] = (+ scan x):t;
        return cs;
      } else {
        var cs = makeDistArray(if includeInitial then expandedDomain(d, axis) else d, t);

        forall (slice, _) in axisSlices(d, new list([axis])) {
          const xSlice = removeDegenRanks(x[slice], 1),
                csSlice = (+ scan xSlice):t;

          for idx in slice {
            var csIdx = idx;
            if includeInitial then csIdx[axis] += 1;
            cs[csIdx] = csSlice[idx[axis]];
          }
        }
        return cs;
      }
    }

    @arkouda.registerCommand()
    proc cumProd(const ref x: [?d] ?t, axis: int, includeInitial: bool): [] t throws
        where t != bool {
      if d.rank == 1 {
        var cs = makeDistArray(if includeInitial then x.size+1 else x.size, t);
        if includeInitial {
            cs[0] = 1:t;
            cs[1..] = (* scan x):t;
        } else {
            cs[0..] = (* scan x):t;
        }
        return cs;
      } else {    // fill with 1s so that if includeInitial is set, answer starts with 1
        var cs = makeDistArray(if includeInitial then expandedDomain(d, axis) else d, 1:t);

        forall (slice, _) in axisSlices(d, new list([axis])) {
          const xSlice = removeDegenRanks(x[slice], 1),
                csSlice = (* scan xSlice):t;

          for idx in slice {
            var csIdx = idx;
            if includeInitial then csIdx[axis] += 1;
            cs[csIdx] = csSlice[idx[axis]];
          }
        }
        return cs;
      }
    }

    private proc expandedDomain(d: domain(?), axis: int): domain(?) {
      var rngs: d.rank*range;
      for param i in 0..<d.rank do rngs[i] = if i == axis
        then d.dim(i).low..(d.dim(i).high + 1)
        else d.dim(i);
      return makeDistDom({(...rngs)});
    }
}
