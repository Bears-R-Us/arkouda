module Stats {
    use AryUtil;

    // TODO: cast to real(32) instead of real(64) for arrays of
    // real(32) or smaller integer types

    proc canBeNan(type t) param: bool do
        return isRealType(t) || isImagType(t) || isComplexType(t);

    proc mean(ref ar: [?aD] ?t): real throws {
        return (+ reduce ar:real) / aD.size:real;
    }

    proc variance(ref ar: [?aD] ?t, ddof: real): real throws {
        const m = mean(ar);
        return (+ reduce ((ar:real - m) ** 2)) / (aD.size - ddof);
    }

    proc std(ref ar: [?aD] ?t, ddof: real): real throws {
        return sqrt(variance(ar, ddof));
    }

    proc cov(ref ar1: [?aD1] ?t1, ref ar2: [?aD2] ?t2): real throws {
        const m1 = mean(ar1), m2 = mean(ar2);
        return (+ reduce ((ar1:real - m1) * (ar2:real - m2))) / (aD1.size - 1):real;
    }

    proc corr(ref ar1: [?aD1] ?t1, ref ar2: [?aD2] ?t2): real throws {
        return cov(ar1, ar2) / (std(ar1, 1) * std(ar2, 1));
    }

    proc meanOver(ref ar: [], slice): real throws {
        var sum = 0.0;
        forall i in slice with (+ reduce sum) do sum += ar[i]:real;
        return sum / slice.size;
    }

    proc varianceOver(ref ar: [], slice, ddof: real): real throws {
        const mean = meanOver(ar, slice);
        var sum = 0.0;
        forall i in slice with (+ reduce sum) do sum += (ar[i]:real - mean) ** 2;
        return sum / (slice.size - ddof);
    }

    proc stdOver(ref ar: [], slice, ddof: real): real throws {
        return sqrt(varianceOver(ar, slice, ddof));
    }

    proc meanSkipNan(const ref arr: [?d] ?t): real throws {
        return meanSkipNan(arr, d);
    }

    proc meanSkipNan(const ref arr: [?aD] ?t, slice): real throws {
        var sum = 0.0,
            count = 0;
        forall i in slice with(+ reduce sum, + reduce count) {
            if canBeNan(t) { if isNan(arr[i]) then continue; }
            sum += arr[i]:real;
            count += 1;
        }
        return sum / count;
    }

    proc varianceSkipNan(ref arr: [?d] ?t, ddof: real): real throws {
        return varianceSkipNan(arr, d, ddof);
    }

    proc varianceSkipNan(ref arr: [?aD] ?t, slice, ddof: real): real throws {
        const mean = meanSkipNan(arr, slice);
        var sum = 0.0,
            count = 0;
        forall i in slice with(+ reduce sum, + reduce count) {
            if canBeNan(t) { if isNan(arr[i]) then continue; }
            sum += (arr[i]:real - mean) ** 2;
            count += 1;
        }
        return sum / (count - ddof);
    }

    proc stdSkipNan(ref arr: [?d] ?t, ddof: real): real throws {
        return stdSkipNan(arr, d, ddof);
    }

    proc stdSkipNan(ref arr: [?aD] ?t, slice, ddof: real): real throws {
        return sqrt(varianceSkipNan(arr, slice, ddof));
    }
}
