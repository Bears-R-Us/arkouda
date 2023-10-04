module Stats {
    use AryUtil;

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
}
