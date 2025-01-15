module Histogram
{
    use ServerConfig;

    use PrivateDist;
    use SymArrayDmap;
    use Logging;
    use Reflection;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const hgLogger = new Logger(logLevel, logChannel);

    /* Helper: is this value within the requested range? */
    inline proc histValWithinRange(val, aMin, aMax): bool {
      return aMin <= val && val <= aMax;
    }

    /* Helper: computes the 1-d bin number for the value, assuming it is within the range. */
    inline proc histValToBin(val, aMin, aMax, numBins, binWidth: real): int {
      return if val == aMax then numBins-1 else ((val - aMin) / binWidth): int;
    }

    /* Helper: converts (x,y) into a bin number for a 2-d histogram.
       Returns -1 if (x,y) is outside the requested range. */
    private proc valsToBin(xi, yi, xMin, xMax, yMin, yMax,
                           numXBins, numYBins, xBinWidth, yBinWidth): int {
        if ! histValWithinRange(xi, xMin, xMax) then return -1;
        if ! histValWithinRange(yi, yMin, yMax) then return -1;
        const xiBin = histValToBin(xi, xMin, xMax, numXBins, xBinWidth);
        const yiBin = histValToBin(yi, yMin, yMax, numYBins, yBinWidth);
        return (xiBin * numYBins) + yiBin;
    }

    /*
        Takes the data in array a, creates an atomic histogram in parallel,
        and copies the result of the histogram operation into a distributed int array

        Returns the histogram (distributed int array).

        :arg a: array of data to be histogrammed
        :type a: [] ?etype

        :arg aMin: Min value to count

        :arg aMax: Max value to count

        :arg bins: allocate size of the histogram's distributed domain
        :type bins: int

        :arg binWidth: set value for either 1:1 unique value counts, or multiple unique values per bin.
        :type binWidth: real

        :returns: [] int
    */
    proc histogramGlobalAtomic(a: [?aD] ?etype, aMin, aMax, bins: int, binWidth: real) throws {

        var hD = makeDistDom(bins);
        var atomicHist: [hD] atomic int;

        // count into atomic histogram
        forall v in a {
          if histValWithinRange(v, aMin, aMax) {
            const vBin = histValToBin(v, aMin, aMax, bins, binWidth);
            atomicHist[vBin].add(1);
          }
        }

        var hist = makeDistArray(bins,int);
        // copy from atomic histogram to normal histogram
        [(e,ae) in zip(hist, atomicHist)] e = ae.read();
        try! hgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                             "hist = %?".format(hist));

        return hist;
    }

    proc histogramGlobalAtomic(x: [?aD] ?etype1, y: [aD] ?etype2, xMin, xMax, yMin, yMax, numXBins: int, numYBins: int, xBinWidth: real, yBinWidth: real) throws {
        const totNumBins = numXBins * numYBins;
        var hD = makeDistDom(totNumBins);
        var atomicHist: [hD] atomic real;

        // count into atomic histogram
        forall (xi, yi) in zip(x, y) {
            const bin = valsToBin(xi, yi, xMin, xMax, yMin, yMax,
                                  numXBins, numYBins, xBinWidth, yBinWidth);
            if bin < 0 then continue;
            atomicHist[bin].add(1);
        }

        var hist = makeDistArray(totNumBins,real);
        // copy from atomic histogram to normal histogram
        [(e,ae) in zip(hist, atomicHist)] e = ae.read();
        try! hgLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                             "hist = %?".format(hist));

        return hist;
    }

    /*
        Takes the data in array a, creates an atomic histogram in each locale,
        + reduces each locale's histogram computations into a distributed int array

        Returns the histogram (distributed int array).

        :arg a: array of data to be histogrammed
        :type a: [] ?etype

        :arg aMin: Min value to count

        :arg aMax: Max value to count

        :arg bins: allocate size of the histogram's distributed domain
        :type bins: int

        :arg binWidth: set value for either 1:1 unique value counts, or multiple unique values per bin.
        :type binWidth: real

        :returns: [] int
    */
    proc histogramLocalAtomic(a: [?aD] ?etype, aMin, aMax, bins: int, binWidth: real) throws {

        // allocate per-locale atomic histogram
        var atomicHist: [PrivateSpace] [0..#bins] atomic int;

        // count into per-locale private atomic histogram
        forall v in a {
          if histValWithinRange(v, aMin, aMax) {
            const vBin = histValToBin(v, aMin, aMax, bins, binWidth);
            atomicHist[here.id][vBin].add(1);
          }
        }

        // +reduce across per-locale histograms to get counts
        var lHist: [0..#bins] int;
        forall i in PrivateSpace with (+ reduce lHist) do
          lHist reduce= atomicHist[i].read();

        var hist = makeDistArray(bins,int);
        hist = lHist;
        return hist;
    }

    proc histogramLocalAtomic(x: [?aD] ?etype1, y: [aD] ?etype2, xMin, xMax, yMin, yMax, numXBins: int, numYBins: int, xBinWidth: real, yBinWidth: real) throws {
        const totNumBins = numXBins * numYBins;
        // allocate per-locale atomic histogram
        var atomicHist: [PrivateSpace] [0..#totNumBins] atomic real;

        // count into per-locale private atomic histogram
        forall (xi, yi) in zip(x, y) {
            const bin = valsToBin(xi, yi, xMin, xMax, yMin, yMax,
                                  numXBins, numYBins, xBinWidth, yBinWidth);
            if bin < 0 then continue;
            atomicHist[here.id][bin].add(1);
        }

        // +reduce across per-locale histograms to get counts
        var lHist: [0..#totNumBins] real;
        forall i in PrivateSpace with (+ reduce lHist) do
          lHist reduce= atomicHist[i].read();

        var hist = makeDistArray(totNumBins,real);
        hist = lHist;
        return hist;
    }

    /*
        Iterates in parallel over all values of a, histogramming into a new array as each value is processed.
        This new array is returned as the histogram.

        Returns the histogram (distributed int array).

        :arg a: array of data to be histogrammed
        :type a: [] ?etype

        :arg aMin: Min value to count

        :arg aMax: Max value to count

        :arg bins: allocate size of the histogram's distributed domain
        :type bins: int

        :arg binWidth: set value for either 1:1 unique value counts, or multiple unique values per bin.
        :type binWidth: real

        :returns: [] int
    */
    proc histogramReduceIntent(a: [?aD] ?etype, aMin, aMax, bins: int, binWidth: real) throws {

        var gHist: [0..#bins] int;

        // count into per-task/per-locale histogram and then reduce as tasks complete
        forall v in a with (+ reduce gHist) {
          if histValWithinRange(v, aMin, aMax) {
            const vBin = histValToBin(v, aMin, aMax, bins, binWidth);
            gHist[vBin] += 1;
          }
        }

        var hist = makeDistArray(bins,int);
        hist = gHist;
        return hist;
    }

    proc histogramReduceIntent(x: [?aD] ?etype1, y: [aD] ?etype2, xMin, xMax, yMin, yMax, numXBins: int, numYBins: int, xBinWidth: real, yBinWidth: real) throws {
        const totNumBins = numXBins * numYBins;
        var gHist: [0..#totNumBins] real;

        // count into per-task/per-locale histogram and then reduce as tasks complete
        forall (xi, yi) in zip(x, y) with (+ reduce gHist) {
            const bin = valsToBin(xi, yi, xMin, xMax, yMin, yMax,
                                  numXBins, numYBins, xBinWidth, yBinWidth);
            if bin < 0 then continue;
            gHist[bin] += 1;
        }

        var hist = makeDistArray(totNumBins,real);
        hist = gHist;
        return hist;
    }
}
