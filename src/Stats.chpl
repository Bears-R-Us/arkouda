module Stats {
    use AryUtil;

    proc mean(ar: [?aD] ?t): real throws {
        return (+ reduce ar:real) / aD.size:real;
    }

    proc variance(ar: [?aD] ?t, ddof: int): real throws {
        return (+ reduce ((ar:real - mean(ar)) ** 2)) / (aD.size - ddof):real;
    }

    proc std(ar: [?aD] ?t, ddof: int): real throws {
        return sqrt(variance(ar, ddof));
    }

    proc cov(ar1: [?aD1] ?t1, ar2: [?aD2] ?t2): real throws {
        return (+ reduce ((ar1:real - mean(ar1)) * (ar2:real - mean(ar2)))) / (aD1.size - 1):real;
    }

    proc corr(ar1: [?aD1] ?t1, ar2: [?aD2] ?t2): real throws {
        return cov(ar1, ar2) / (std(ar1, 1) * std(ar2, 1));
    }
}
