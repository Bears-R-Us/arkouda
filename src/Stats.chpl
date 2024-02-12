module Stats {
    use AryUtil;

    // TODO: cast to real(32) instead of real(64) for arrays of
    // real(32) or smaller integer types

    proc mean(ref ar: [?aD] ?t): real throws {
        return (+ reduce ar:real) / aD.size:real;
    }

    proc variance(ref ar: [?aD] ?t, ddof: int): real throws {
        return (+ reduce ((ar:real - mean(ar)) ** 2)) / (aD.size - ddof):real;
    }

    proc std(ref ar: [?aD] ?t, ddof: int): real throws {
        return sqrt(variance(ar, ddof));
    }

    proc cov(ref ar1: [?aD1] ?t1, ref ar2: [?aD2] ?t2): real throws {
        return (+ reduce ((ar1:real - mean(ar1)) * (ar2:real - mean(ar2)))) / (aD1.size - 1):real;
    }

    proc corr(ref ar1: [?aD1] ?t1, ref ar2: [?aD2] ?t2): real throws {
        return cov(ar1, ar2) / (std(ar1, 1) * std(ar2, 1));
    }

    proc meanOver(ref ar: [], d: domain): real throws {
        var sum = 0.0;
        forall i in d with (+ reduce sum) do sum += ar[i]:real;
        return sum / d.size;
    }

    proc varianceOver(ref ar: [], d: domain, ddof: int): real throws {
        const mean = meanOver(ar, d);
        var sum = 0.0;
        forall i in d with (+ reduce sum) do sum += (ar[i]:real - mean) ** 2;
        return sum / (d.size - ddof):real;
    }

    proc stdOver(ref ar: [], d: domain, ddof: int): real throws {
        return sqrt(varianceOver(ar, d, ddof));
    }
}
