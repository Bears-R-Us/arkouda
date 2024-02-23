module Stats {
    use AryUtil;

    // TODO: cast to real(32) instead of real(64) for arrays of
    // real(32) or smaller integer types

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
}
