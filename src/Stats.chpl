module Stats {
    use AryUtil;

    proc mean(ar1: [?aD1] ?t): real throws {
        return (+ reduce ar1:real) / aD1.size:real;
    }

    proc variance(ar1: [?aD1] ?t, ddof: int): real throws {
        return (+ reduce ((ar1:real - mean(ar1)) ** 2)) / (aD1.size - ddof):real;
    }

    proc std(ar1: [?aD1] ?t, ddof: int): real throws {
        return sqrt(variance(ar1, ddof));
    }

    proc cov(ar1: [?aD1] ?t, ar2: [?aD2] ?t2): real throws {
        return (+ reduce ((ar1:real - mean(ar1)) * (ar2:real - mean(ar2)))) / (aD1.size - 1):real;
    }

    proc corr(ar1: [?aD1] ?t, ar2: [?aD2] ?t2): real throws {
        return cov(ar1, ar2) / (std(ar1, 1) * std(ar2, 1));
    }
}
