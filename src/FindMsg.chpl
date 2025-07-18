module FindMsg
{
  use ServerConfig;

  use Time;
  use Math only;
  use Reflection;
  use ServerErrors;
  use Logging;
  use Message;
  use BigInteger;
  
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use SegmentedString;
  use ServerErrorStrings;
  use CommAggregation;
  use PrivateDist;
  
  use AryUtil;
  use CTypes;
  use Set;
  use List;
  use ArkoudaSortCompat;
  use Map;

  use Repartition;

  private config const logLevel = ServerConfig.logLevel;
  private config const logChannel = ServerConfig.logChannel;
  const fmLogger = new Logger(logLevel, logChannel);

  use CommandMap;

  @chplcheck.ignore("UnusedFormal")
  proc findStrMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
    param pn = Reflection.getRoutineName();
    var sendWarning = false;
    var repMsg: string;
    var allOccurrences = msgArgs.get("allOccurrences").toScalar(bool);
    var removeMissing = msgArgs.get("removeMissing").toScalar(bool);
    var qName = msgArgs.getValueOf("query");
    var sName = msgArgs.getValueOf("space");
    var querySegString = getSegString(qName, st);
    var spaceSegString = getSegString(sName, st);
    var numQuery = querySegString.offsets.a.size;
    var numSpace = spaceSegString.offsets.a.size;

    const queryLocRanges = [loc in Locales] querySegString.offsets.a.domain.localSubdomain(loc).dim(0);
    
    fmLogger.debug(getModuleName(),getRoutineName(), getLineNumber(), 
          "number of queries: %i, size of space: %i".format(numQuery,numSpace));

    var queryStrOffsetInLocale: [PrivateSpace] list(int);
    var queryStrBytesInLocale: [PrivateSpace] list(uint(8));
    var spaceStrOffsetInLocale: [PrivateSpace] list(int);
    var spaceStrBytesInLocale: [PrivateSpace] list(uint(8));

    var queryOrigIndices: [PrivateSpace] list(int);
    var spaceOrigIndices: [PrivateSpace] list(int);

    coforall loc in Locales do on loc {

      ref queryGlobalOffsets = querySegString.offsets.a;
      const queryOffsetsDom = querySegString.offsets.a.domain.localSubdomain();
      const querySize = queryOffsetsDom.size;
      const queryOffsets = querySegString.offsets.a.localSlice[queryOffsetsDom].reindex(0..#querySize);
      const queryTopEnd = if queryOffsetsDom.high >= querySegString.offsets.a.domain.high
        then querySegString.values.a.size
        else queryGlobalOffsets[queryOffsetsDom.high + 1];
      const queryBytes = querySegString.values.a[queryOffsets[0]..<queryTopEnd];
      const queryOffsetsRelative = queryOffsets - queryOffsets[0];

      var myQueryOrigIndices: [0..#queryOffsets.size] int = queryOffsetsDom.low..;

      ref spaceGlobalOffsets = spaceSegString.offsets.a;
      const spaceOffsetsDom = spaceSegString.offsets.a.domain.localSubdomain();
      const spaceSize = spaceOffsetsDom.size;
      const spaceOffsets = spaceSegString.offsets.a.localSlice[spaceOffsetsDom].reindex(0..#spaceSize);
      const spaceTopEnd = if spaceOffsetsDom.high >= spaceSegString.offsets.a.domain.high
        then spaceSegString.values.a.size
        else spaceGlobalOffsets[spaceOffsetsDom.high + 1];
      const spaceBytes = spaceSegString.values.a[spaceOffsets[0]..<spaceTopEnd];
      const spaceOffsetsRelative = spaceOffsets - spaceOffsets[0];

      var mySpaceOrigIndices: [0..#spaceOffsets.size] int = spaceOffsetsDom.low..;

      queryStrOffsetInLocale[here.id] = new list(queryOffsetsRelative);
      queryStrBytesInLocale[here.id] = new list(queryBytes);
      spaceStrOffsetInLocale[here.id] = new list(spaceOffsetsRelative);
      spaceStrBytesInLocale[here.id] = new list(spaceBytes);

      queryOrigIndices[here.id] = new list(myQueryOrigIndices);
      spaceOrigIndices[here.id] = new list(mySpaceOrigIndices);
      
    }

    var (queryRecvOffsets, queryRecvBytes, queryDestLocales) = repartitionByHashStringWithDestLocales(
                                                               queryStrOffsetInLocale,
                                                               queryStrBytesInLocale
                                                             );

    var (spaceRecvOffsets, spaceRecvBytes, spaceDestLocales) = repartitionByHashStringWithDestLocales(
                                                               spaceStrOffsetInLocale,
                                                               spaceStrBytesInLocale
                                                             );

    var queryRecvInds = repartitionByLocale(int, queryDestLocales, queryOrigIndices);
    var spaceRecvInds = repartitionByLocale(int, spaceDestLocales, spaceOrigIndices);

    var queryRespVals: [PrivateSpace] list(int);
    var queryRespValLocales: [PrivateSpace] list(int);
    var queryRespOffsets: [PrivateSpace] list(int);
    var queryRespOffsetLocales: [PrivateSpace] list(int);

    var queryRespNumRecv: [0..<numLocales] [0..<numLocales] int;
    var queryRespSizeRecv: [0..<numLocales] [0..<numLocales] int;

    @chplcheck.ignore("UnusedTaskIntent")
    coforall loc in Locales with (|| reduce sendWarning) do on loc {

      const sOffsets = spaceRecvOffsets[here.id];
      var sBytes = spaceRecvBytes[here.id].toArray();
      const sInds = spaceRecvInds[here.id];

      const N = sOffsets.size;

      var strIndPairs: [0..<N] (string, int);

      forall i in 0..<N {
        const start = sOffsets[i];
        const end = if i+1 < N then sOffsets[i + 1] else sBytes.size;
        const s = interpretAsString(sBytes, start..<end);
        strIndPairs[i] = (s, sInds[i]);
      }

      const qOffsets = queryRecvOffsets[here.id];
      var qBytes = queryRecvBytes[here.id].toArray();
      const qInds = queryRecvInds[here.id];

      const queryN = qOffsets.size;

      var queryStrIndPairs: [0..<queryN] (string, int);

      forall i in 0..<queryN {
        const start = qOffsets[i];
        const end = if i+1 < queryN then qOffsets[i + 1] else qBytes.size;
        const s = interpretAsString(qBytes, start..<end);
        queryStrIndPairs[i] = (s, qInds[i]);
      }

      if N == 0 || queryN == 0 {
        if queryN == 0  || (!allOccurrences && !removeMissing){

          queryRespVals[here.id] = new list(int);
          queryRespValLocales[here.id] = new list(int);
          queryRespOffsets[here.id] = new list(int);
          queryRespOffsetLocales[here.id] = new list(int);
          
        } else {

          var queryLocs: [0..<queryN] int;

          forall i in 0..<queryN {

            var (currQuery, queryInd) = queryStrIndPairs[i];

            for (qlr, locID) in zip(queryLocRanges, 0..<numLocales) {
              if qlr.contains(queryInd) {
                queryLocs[i] = locID;
                break;
              }
            }

          }

          if allOccurrences {

            queryRespVals[here.id] = new list(int);
            queryRespValLocales[here.id] = new list(int);
            queryRespOffsets[here.id] = new list([0..#queryN] 0);
            queryRespOffsetLocales[here.id] = new list(queryLocs);

            for i in 0..<numLocales {

              var onCurrLoc: [0..<queryN] bool = queryLocs == i;
              queryRespNumRecv[i][here.id] = + reduce onCurrLoc;

            }

          } else {

            var fillNegOnes: [0..#queryN] int = -1;
            queryRespVals[here.id] = new list(fillNegOnes);
            queryRespValLocales[here.id] = new list(queryLocs);
            queryRespOffsets[here.id] = new list([i in 0..#queryN] i);
            queryRespOffsetLocales[here.id] = new list(queryLocs);

            for i in 0..<numLocales {

              var onCurrLoc: [0..<queryN] bool = queryLocs == i;
              queryRespNumRecv[i][here.id] = + reduce onCurrLoc;
              queryRespSizeRecv[i][here.id] = + reduce onCurrLoc;

            }

          }
        }

      } else { // I do not like this else. Essentially, if the situation is normal, proceed as normal.

        record tupKeyComparator: keyComparator {
          proc key(a: (string, int)) {
            return a(0);
          }
        }

        compatSort(strIndPairs, comparator=new tupKeyComparator());

        var sortedInds: [0..<N] int;
        var allStrings: [0..<N] string;
        var allStringsShifted: [0..<N] string;
        forall i in 0..<N {
          sortedInds[i] = strIndPairs[i][1];
          allStrings[i] = strIndPairs[i][0];
          allStringsShifted[i] = if i == 0 then strIndPairs[i][0] else strIndPairs[i - 1][0];
        }

        var lowStringIndicator: [0..<N] int = allStrings != allStringsShifted;
        lowStringIndicator[0] = 1;
        var lowStringIndexer = (+ scan lowStringIndicator) - 1;
        const nUnique = lowStringIndexer[lowStringIndexer.size - 1] + 1;
        var lowStringIndices: [0..<nUnique] int;
        var stringToInd = new map(string, int, parSafe = true);
        var stringGroupToInd: [0..<nUnique] int;
        forall i in 0..<N with (ref stringToInd) {
          if lowStringIndicator[i] != 0 {
            stringToInd.add(allStrings[i], i);
            stringGroupToInd[lowStringIndexer[i]] = i;
          }
        }
        var groupSizes: [0..<nUnique] int;
        forall i in 0..<nUnique {
          groupSizes[i] = if i == nUnique - 1 
                          then N - stringGroupToInd[i]
                          else stringGroupToInd[i + 1] - stringGroupToInd[i];
        }

        var respSizes: [0..<queryN] int;

        forall i in 0..<queryN with (|| reduce sendWarning) {
          if stringToInd.contains(queryStrIndPairs[i][0]) {
            if allOccurrences {
              respSizes[i] = groupSizes[lowStringIndexer[stringToInd.get(queryStrIndPairs[i][0], -1)]];
            } else {
              respSizes[i] = 1;
              if groupSizes[lowStringIndexer[stringToInd.get(queryStrIndPairs[i][0], -1)]] > 1 {
                sendWarning = true;
              }
            }
          } else {
            if allOccurrences || removeMissing {
              respSizes[i] = 0;
            } else {
              respSizes[i] = 1;
            }
          }
        }

        var respInds = (+ scan respSizes) - respSizes;
        var totalRespSize = + reduce respSizes;

        var resp: [0..<totalRespSize] int;

        var queryLocs: [0..<queryN] int;

        forall i in 0..<queryN {

          var (currQuery, queryInd) = queryStrIndPairs[i];

          if stringToInd.contains(currQuery) {
            const stringGroupInd = stringToInd.get(currQuery, -1);
            const groupInd = lowStringIndexer[stringGroupInd];
            for j in 0..<respSizes[i] {
              resp[respInds[i] + j] = sortedInds[stringGroupInd + j];
            }
          } else {
            if !removeMissing && !allOccurrences {
              resp[respInds[i]] = -1;
            }
          }

          for (qlr, locID) in zip(queryLocRanges, 0..<numLocales) {
            if qlr.contains(queryInd) {
              queryLocs[i] = locID;
              break;
            }
          }

        }

        // Send the data back

        var myQueryRespOffsets: [0..<queryN] int;
        var myQueryRespValLocales: [0..<totalRespSize] int;

        for i in 0..<numLocales {

          var onCurrLoc: [0..<queryN] bool = queryLocs == i;
          var currLocSizes: [0..<queryN] int = (onCurrLoc: int) * respSizes;
          myQueryRespOffsets += ((+ scan currLocSizes) - currLocSizes) * (onCurrLoc: int);
          queryRespNumRecv[i][here.id] = + reduce onCurrLoc;
          queryRespSizeRecv[i][here.id] = + reduce currLocSizes;

          forall j in 0..<queryN {
            if onCurrLoc[j] {
              myQueryRespValLocales[respInds[j]..#respSizes[j]] = i;
            }
          }

        }

        queryRespVals[here.id] = new list(resp);
        queryRespValLocales[here.id] = new list(myQueryRespValLocales);
        queryRespOffsets[here.id] = new list(myQueryRespOffsets);
        queryRespOffsetLocales[here.id] = new list(queryLocs);

      }

    }

    var queryRecvRespVals = repartitionByLocale(int, queryRespValLocales, queryRespVals);
    var queryRecvRespOffsets = repartitionByLocale(int, queryRespOffsetLocales, queryRespOffsets);

    var queryRespByLoc: [0..<numLocales] int;
    var queryRespSizeByLoc: [0..<numLocales] int;

    forall i in 0..<numLocales {
      queryRespByLoc[i] = + reduce queryRespNumRecv[i];
      queryRespSizeByLoc[i] = + reduce queryRespSizeRecv[i];
    }

    var queryRespValsInd = (+ scan queryRespSizeByLoc) - queryRespSizeByLoc;
    var queryRespOffsetInd = (+ scan queryRespByLoc) - queryRespByLoc;

    var queryResp = makeDistArray(+ reduce queryRespSizeByLoc, int);
    var queryOffsets = makeDistArray(+ reduce queryRespByLoc, int);

    coforall loc in Locales do on loc {
      const numRespByLoc = queryRespNumRecv[here.id];
      const sizeRespByLoc = queryRespSizeRecv[here.id];
      const myQueryRespValsInd = queryRespValsInd[here.id];
      const myQueryRespOffsetInd = queryRespOffsetInd[here.id];
      
      var myRecvVals = queryRecvRespVals[here.id].toArray();
      var myRecvOffsets = queryRecvRespOffsets[here.id].toArray();

      var myQueryDestLocales = queryDestLocales[here.id].toArray();
      var myQueryOrigIndices = queryOrigIndices[here.id].toArray();

      // Here's the idea - the offsets will be relative to the locale that sent them. So every so often,
      // it resets to 0. So I think I need like a + scan sizeRespByLoc - sizeRespByLoc
      // and then indexing a certain amount in and just adding sizeRespByLoc[i] in. So that way we get
      // true offsets.

      var respValOffsetsBySendingLocale: [0..<numLocales] int = (+ scan sizeRespByLoc) - sizeRespByLoc;
      var queryOffsetOffsetsByLocale: [0..<numLocales] int = (+ scan numRespByLoc) - numRespByLoc;

      if (+ reduce numRespByLoc) > 0 {

        for sourceLoc in 0..<numLocales {
          const offsetStart = queryOffsetOffsetsByLocale[sourceLoc];
          const offsetCount = numRespByLoc[sourceLoc];
          myRecvOffsets[offsetStart..#offsetCount] += respValOffsetsBySendingLocale[sourceLoc];
        }

        // Then we can use queryOrigIndices and queryDestLocales to figure out what goes where.
        // I want to say that I start by doing this on the offsets, and create a new offset array based
        // on that. Then when I have the offsets, I can put things in the original array asynchronously.

        var recvRespQueryIndices: [0..#myRecvOffsets.size] int;

        for sourceLoc in 0..<numLocales {
          var temp = (myQueryDestLocales == sourceLoc): int;
          var temp2 = (+ scan temp) - temp;
          const offsetStart = queryOffsetOffsetsByLocale[sourceLoc];
          forall idx in 0..#myRecvOffsets.size {
            if temp[idx] != 0 {
              recvRespQueryIndices[temp2[idx] + offsetStart] = myQueryOrigIndices[idx];
            }
          }
        }

        var finalLocalSizes: [0..#myRecvOffsets.size] int;
        var localIndices = recvRespQueryIndices - myQueryOrigIndices[0];

        forall i in 0..#myRecvOffsets.size {
          const localOffset = localIndices[i];
          const localStart = myRecvOffsets[i];
          const localEnd = if i == myRecvOffsets.size - 1 then myRecvVals.size else myRecvOffsets[i + 1];
          finalLocalSizes[localOffset] = localEnd - localStart;
        }

        var finalLocalOffsets = (+ scan finalLocalSizes) - finalLocalSizes;

        var finalRespVals: [0..#(+ reduce sizeRespByLoc)] int;

        forall i in 0..#myRecvOffsets.size {
          const localIndex = localIndices[i];
          const localOffset = finalLocalOffsets[localIndex];
          const localSize = finalLocalSizes[localIndex];
          const recvOffset = myRecvOffsets[i];
          finalRespVals[localOffset..#localSize] = myRecvVals[recvOffset..#localSize];
        }

        // Finally, I can create a global offset array using myQueryRespValsInd and myQueryRespOffsetInd.
        // And I think I use the vals ind to shift the whole local offset array, the the offsetInd to put
        // it in the right place.

        var finalGlobalOffsets = finalLocalOffsets + myQueryRespValsInd;

        queryResp[myQueryRespValsInd..#finalRespVals.size] = finalRespVals;
        queryOffsets[myQueryRespOffsetInd..#finalGlobalOffsets.size] = finalGlobalOffsets;
      }
    }

    var respName = st.nextName();
    var offsetName = st.nextName();

    st.addEntry(respName, createSymEntry(queryResp));
    st.addEntry(offsetName, createSymEntry(queryOffsets));

    repMsg = "created " + st.attrib(respName) + "+created " + st.attrib(offsetName) +
             "+sendWarning " + sendWarning:string;

    return new MsgTuple(repMsg, MsgType.NORMAL);
    
  }
  registerFunction("findStr", findStrMsg, getModuleName());
}