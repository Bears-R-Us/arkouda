module Stats {
    use AryUtil;

    // TODO: cast to real(32) instead of real(64) for arrays of
    // real(32) or smaller integer types

    proc canBeNan(type t) param: bool do
        return isRealType(t) || isImagType(t) || isComplexType(t);

    proc mean(const ref ar: [?aD] ?t): real throws {
        return (+ reduce ar:real) / aD.size:real;
    }

    proc variance(const ref ar: [?aD] ?t, ddof: real): real throws {
        const m = mean(ar);
        return (+ reduce ((ar:real - m) ** 2)) / (aD.size - ddof);
    }

    proc std(const ref ar: [?aD] ?t, ddof: real): real throws {
        return sqrt(variance(ar, ddof));
    }

    proc meanOver(const ref ar: [], slice): real throws {
        var sum = 0.0;
        forall i in slice with (+ reduce sum) do sum += ar[i]:real;
        return sum / slice.size;
    }

    proc varianceOver(const ref ar: [], slice, ddof: real): real throws {
        const mean = meanOver(ar, slice);
        var sum = 0.0;
        forall i in slice with (+ reduce sum) do sum += (ar[i]:real - mean) ** 2;
        return sum / (slice.size - ddof);
    }

    proc stdOver(const ref ar: [], slice, ddof: real): real throws {
        return sqrt(varianceOver(ar, slice, ddof));
    }

    proc meanSkipNan(const ref arr: [?aD] ?t, slice = aD): real throws
        where canBeNan(t)
    {
        var sum = 0.0,
            count = 0;
        forall i in slice with(+ reduce sum, + reduce count) {
            if isNan(arr[i]) then continue;
            sum += arr[i]:real;
            count += 1;
        }
        return sum / count;
    }

    proc varianceSkipNan(const ref arr: [?aD] ?t, slice = aD, ddof: real): real throws
        where canBeNan(t)
    {
        const mean = meanSkipNan(arr, slice);
        var sum = 0.0,
            count = 0;
        forall i in slice with(+ reduce sum, + reduce count) {
            if isNan(arr[i]) then continue;
            sum += (arr[i]:real - mean) ** 2;
            count += 1;
        }
        return sum / (count - ddof);
    }

    proc stdSkipNan(const ref arr: [?aD] ?t, slice = aD, ddof: real): real throws {
        return sqrt(varianceSkipNan(arr, slice, ddof));
    }
}
