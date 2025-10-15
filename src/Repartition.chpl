module Repartition
{

  // The goal of this module is to provide helper functions to facilitate redistributing data between
  // locales.

  use PrivateDist;
  use SegmentedString;
  use List;
  use BigInteger;

  record innerArray {
    type t;
    var Dom: domain(1);
    var Arr: [Dom] t;

    proc init(in Dom: domain(1), type t) {
      this.t = t;
      this.Dom = Dom;
    }

    proc init(type t) {
      this.t = t;
    }
  }

  // Note, the arrays passed here must have PrivateSpace domains.
  proc repartitionByHashString(const ref strOffsets: [] list(int),
                                 const ref strBytes: [] list(uint(8))):
    ([PrivateSpace] list(int), [PrivateSpace] list(uint(8)))
  {
    var destLocales: [PrivateSpace] list(int);

    coforall loc in Locales do on loc {

      var myStrOffsets = strOffsets[here.id].toArray();
      var myStrBytes = strBytes[here.id].toArray();
      var myDestLocales: [0..#myStrOffsets.size] int;

      forall i in myStrOffsets.domain {
        var start = myStrOffsets[i];
        var end = if i == myStrOffsets.size - 1 then myStrBytes.size else myStrOffsets[i + 1];
        var str = interpretAsString(myStrBytes, start..<end);
        myDestLocales[i] = (str.hash() % numLocales): int;
      }

      destLocales[here.id] = new list(myDestLocales);
    }

    return repartitionByLocaleString(destLocales, strOffsets, strBytes);
  }

  // Note, the arrays passed here must have PrivateSpace domains. With Chapel
  // 2.5, distribution equality with PrivateSpace doesn't work.
  // https://github.com/chapel-lang/chapel/pull/27397 is the upstream PR to
  // fix it.
  proc repartitionByLocaleString(const ref destLocales: [] list(int),
                                 const ref strOffsets: [] list(int),
                                 const ref strBytes: [] list(uint(8))):
    ([PrivateSpace] list(int), [PrivateSpace] list(uint(8)))
  {
    var maxBytesPerLocale: int;
    var maxStringsPerLocale: int;
    var numBytesReceivingByLocale: [PrivateSpace] [0..#numLocales] int;
    var numStringsReceivingByLocale: [PrivateSpace] [0..#numLocales] int;
    var allStrSizes: [PrivateSpace] list(int);

    // First we need to figure out how many bytes and strings are getting transferred.
    // Also calculating the sizes of each string so that indexing is easier down the road.

    coforall loc in Locales 
      with (max reduce maxBytesPerLocale, max reduce maxStringsPerLocale) 
      do on loc
    {
      const myDestLocales = destLocales[here.id].toArray();
      const myStrOffsets = strOffsets[here.id].toArray();
      const myStrBytesSize = strBytes[here.id].size;
      var bytesPerLocale: [0..#numLocales] int = 0;
      var stringsPerLocale: [0..#numLocales] int = 0;
      var sizes: [0..#myDestLocales.size] int = 0;

      forall idx in myDestLocales.domain with (+ reduce bytesPerLocale, + reduce stringsPerLocale) {
        var destLoc = myDestLocales[idx];
        const start = myStrOffsets[idx];
        const end = if idx == myDestLocales.size - 1 then myStrBytesSize else myStrOffsets[idx + 1];
        const size = end - start;

        sizes[idx] = size;
        bytesPerLocale[destLoc] += size;
        stringsPerLocale[destLoc] += 1;
      }

      maxBytesPerLocale = max reduce bytesPerLocale;
      maxStringsPerLocale = max reduce stringsPerLocale;

      forall i in 0..#numLocales {
        numBytesReceivingByLocale[i][here.id] = bytesPerLocale[i];
        numStringsReceivingByLocale[i][here.id] = stringsPerLocale[i];
      }

      allStrSizes[here.id] = new list(sizes);
    }

    var recvOffsets: [PrivateSpace] [0..#numLocales] [0..#maxStringsPerLocale] int;
    var recvBytes: [PrivateSpace] [0..#numLocales] [0..#maxBytesPerLocale] uint(8);

    // Now we're going to fill the receiving buffers
    // with the data that needs to get transferred from another locale

    coforall loc in Locales do on loc {
      const myDestLocales = destLocales[here.id].toArray();
      const myStrOffsets = strOffsets[here.id].toArray();
      const myStrBytes = strBytes[here.id].toArray();
      const mySizes = allStrSizes[here.id].toArray();
      var idxInDestLoc: [0..#myDestLocales.size] int = 0;
      var offsetInDestLoc: [0..#myDestLocales.size] int = 0;
      var bytesPerLocale: [0..#numLocales] int = 0;
      var numStringsPerLocale: [0..#numLocales] int = 0;

      // First we need to figure out what the destination index will be
      // for each offset and for the bytes

      for i in 0..#numLocales {
        var onCurrLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then 1 else 0;
        var currLocSizes = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then mySizes[j] else 0;
        
        var idxInCurrLoc = (+ scan onCurrLoc) - onCurrLoc;
        idxInDestLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then idxInCurrLoc[j] 
                                                     else idxInDestLoc[j];

        var offsetInCurrLoc = (+ scan currLocSizes) - currLocSizes;
        offsetInDestLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then offsetInCurrLoc[j]
                                                        else offsetInDestLoc[j];

        bytesPerLocale[i] = + reduce currLocSizes;
        numStringsPerLocale[i] = + reduce onCurrLoc;
      }

      var sendOffsets: [0..#numLocales] [0..#maxStringsPerLocale] int;
      var sendBytes: [0..#numLocales] [0..#maxBytesPerLocale] uint(8);

      forall idx in 0..#myDestLocales.size {

        var destLoc = myDestLocales[idx];
        var idxInOffsetArr = idxInDestLoc[idx];
        var offset = offsetInDestLoc[idx];
        var size = mySizes[idx];
        
        sendOffsets[destLoc][idxInOffsetArr] = offset;
        sendBytes[destLoc][offset..#size] = myStrBytes[myStrOffsets[idx]..#size];

      }

      // Maybe could be a forall but I don't know how that plays with the bulk transfer.
      for i in 0..#numLocales {
        recvOffsets[i][here.id][0..#numStringsPerLocale[i]] = sendOffsets[i][0..#numStringsPerLocale[i]];
        recvBytes[i][here.id][0..#bytesPerLocale[i]] = sendBytes[i][0..#bytesPerLocale[i]];
      }

    }

    var returnedOffsets: [PrivateSpace] list(int);
    var returnedBytes: [PrivateSpace] list(uint(8));

    // Now that the buffers have been filled, we're going to group them together into a single list.
    // Strictly speaking, this probably isn't necessary, but it does make it more friendly to work with

    coforall loc in Locales do on loc {
      const ref numBytesReceivedByLoc = numBytesReceivingByLocale[here.id];
      const ref numStringsReceivedByLoc = numStringsReceivingByLocale[here.id];
      const ref myRecvOffsets = recvOffsets[here.id];
      const ref myRecvBytes = recvBytes[here.id];
      var numBytesReceived = + reduce numBytesReceivedByLoc;
      var numStringsReceived = + reduce numStringsReceivedByLoc;
      var myOffsets: [0..#numStringsReceived] int = 0;
      var myBytes: [0..#numBytesReceived] uint(8) = 0;
      var byteOffsetAdjuster = (+ scan numBytesReceivedByLoc) - numBytesReceivedByLoc;
      var idxOffsetAdjuster = (+ scan numStringsReceivedByLoc) - numStringsReceivedByLoc;

      for i in 0..#numLocales {

        var byteOffsetThisLoc = byteOffsetAdjuster[i];
        var idxOffsetThisLoc = idxOffsetAdjuster[i];
        myOffsets[idxOffsetThisLoc..#numStringsReceivedByLoc[i]]
          = myRecvOffsets[i][0..#numStringsReceivedByLoc[i]] + byteOffsetThisLoc;
        myBytes[byteOffsetThisLoc..#numBytesReceivedByLoc[i]]
          = myRecvBytes[i][0..#numBytesReceivedByLoc[i]];

      }

      returnedOffsets[here.id] = new list(myOffsets);
      returnedBytes[here.id] = new list(myBytes);

    }

    return (returnedOffsets, returnedBytes);

  }

  proc repartitionByLocaleStringArray(const ref destLocales: [] innerArray(int),
                                 const ref strOffsets: [] innerArray(int),
                                 const ref strBytes: [] innerArray(uint(8))):
    ([PrivateSpace] innerArray(int), [PrivateSpace] innerArray(uint(8)))
  {
    var numBytesSendingByLocale: [PrivateSpace] [0..#numLocales] int;
    var numStringsSendingByLocale: [PrivateSpace] [0..#numLocales] int;
    var allStrSizes: [PrivateSpace] innerArray(int);
    var sendOffsets: [PrivateSpace] [0..#numLocales] innerArray(int);
    var sendBytes: [PrivateSpace] [0..#numLocales] innerArray(uint(8));

    // First we need to figure out how many bytes and strings are getting transferred.
    // Also calculating the sizes of each string so that indexing is easier down the road.

    coforall loc in Locales do on loc
    {
      const ref myDestLocales = destLocales[here.id].Arr;
      const ref myStrOffsets = strOffsets[here.id].Arr;
      const ref myStrBytes = strBytes[here.id].Arr;
      const myStrBytesSize = myStrBytes.size;
      var bytesPerLocale: [0..#numLocales] int = 0;
      var stringsPerLocale: [0..#numLocales] int = 0;
      allStrSizes[here.id] = new innerArray(myDestLocales.domain, int);
      ref sizes = allStrSizes[here.id].Arr;
      const topEnd = myDestLocales.domain.high;

      forall idx in myDestLocales.domain with (+ reduce bytesPerLocale, + reduce stringsPerLocale) {
        var destLoc = myDestLocales[idx];
        const start = myStrOffsets[idx];
        const end = if idx == topEnd then myStrBytesSize else myStrOffsets[idx + 1];
        const size = end - start;

        sizes[idx] = size;
        bytesPerLocale[destLoc] += size;
        stringsPerLocale[destLoc] += 1;
      }

      numBytesSendingByLocale[here.id] = bytesPerLocale;
      numStringsSendingByLocale[here.id] = stringsPerLocale;

      var currLocIndAllLocales: [myDestLocales.domain] int;
      var currLocOffsetAllLocales: [myDestLocales.domain] int;

      /*
      // It would be very cool if we could do things this way:

      for i in 0..#numLocales {
        sendOffsets[here.id][i] = new innerArray({0..#stringsPerLocale[i]}, int);
        sendBytes[here.id][i] = new innerArray({0..#bytesPerLocale[i]}, uint(8));

        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;
        const currLocSizes = doCurrLoc * sizes;
        const currLocOffsets = (+ scan currLocSizes) - currLocSizes;

        currLocIndAllLocales += doCurrLoc * currLocInd;
        currLocOffsetAllLocales += doCurrLoc * currLocOffsets;
      }

      ref currSendOffsets = [i in 0..#numLocales] sendOffsets[here.id][i].Arr;
      ref currSendBytes = [i in 0..#numLocales] sendBytes[here.id][i].Arr;

      forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
        const currSize = sizes[j];
        currSendOffsets[dl][currLocIndAllLocales[j]] = currLocOffsetAllLocales[j];
        currSendBytes[dl][currLocOffsetAllLocales[j]..#currSize]
                      = myStrBytes[myStrOffsets[j]..#currSize];
      }

      // Notice that there's only one forall at the end. This would potentially be faster.
      // But because we can't do an array of refs (currently, at least), there's going to be a forall
      // inside the for, and we do a forall for each of the locales. With all the vectorized stuff
      // we're kind of already doing that so I'm not convinced what follows is significantly slower.
      */

      for i in 0..#numLocales {
        sendOffsets[here.id][i] = new innerArray({0..#stringsPerLocale[i]}, int);
        sendBytes[here.id][i] = new innerArray({0..#bytesPerLocale[i]}, uint(8));

        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;
        const currLocSizes = doCurrLoc * sizes;
        const currLocOffsets = (+ scan currLocSizes) - currLocSizes;

        currLocIndAllLocales += doCurrLoc * currLocInd;
        currLocOffsetAllLocales += doCurrLoc * currLocOffsets;

        ref currSendOffsets = sendOffsets[here.id][i].Arr;
        ref currSendBytes = sendBytes[here.id][i].Arr;

        forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
          if dl == i {
            const currSize = sizes[j];
            currSendOffsets[currLocIndAllLocales[j]] = currLocOffsetAllLocales[j];
            currSendBytes[currLocOffsetAllLocales[j]..#currSize]
                        = myStrBytes[myStrOffsets[j]..#currSize];
          }
        }
      }
      
    }

    var recvOffsets: [PrivateSpace] innerArray(int);
    var recvBytes: [PrivateSpace] innerArray(uint(8));

    // Now we're going to fill the receiving buffers
    // with the data that needs to get transferred from another locale

    coforall loc in Locales do on loc {
      
      const numStringsReceivingByLocale = [i in 0..#numLocales] numStringsSendingByLocale[i][here.id];
      const numBytesReceivingByLocale = [i in 0..#numLocales] numBytesSendingByLocale[i][here.id];
      const stringOffsetByLocale = (+ scan numStringsReceivingByLocale) - numStringsReceivingByLocale;
      const byteOffsetByLocale = (+ scan numBytesReceivingByLocale) - numBytesReceivingByLocale;

      recvBytes[here.id] = new innerArray({0..#(+ reduce numBytesReceivingByLocale)}, uint(8));
      recvOffsets[here.id] = new innerArray({0..#(+ reduce numStringsReceivingByLocale)}, int);
      ref myRecvBytes = recvBytes[here.id].Arr;
      ref myRecvOffsets = recvOffsets[here.id].Arr;

      for i in 0..#numLocales {

        // myRecvOffsets[stringOffsetByLocale[i]..#numStringsReceivingByLocale[i]]
        //              = sendOffsets[i][here.id].Arr + byteOffsetByLocale[i];
        myRecvOffsets[stringOffsetByLocale[i]..#numStringsReceivingByLocale[i]]
                    = sendOffsets[i][here.id].Arr;
        myRecvOffsets[stringOffsetByLocale[i]..#numStringsReceivingByLocale[i]] += byteOffsetByLocale[i];
        myRecvBytes[byteOffsetByLocale[i]..#numBytesReceivingByLocale[i]] = sendBytes[i][here.id].Arr;

      }

    }

    return (recvOffsets, recvBytes);

  }

  // Note, the arrays passed here must have PrivateSpace domains.
  proc repartitionByLocale(type t,
                           const ref destLocales: [] list(int),
                           const ref vals: [] list(t))
  {
    type eltType = vals.eltType.eltType;

    var maxValsPerLocale: int;
    var numValsReceivingByLocale: [PrivateSpace] [0..#numLocales] int;

    coforall loc in Locales 
      with (max reduce maxValsPerLocale) 
      do on loc
    {
      const ref myDestLocales = destLocales[here.id];
      const ref myVals = vals[here.id];
      var valsPerLocale: [0..#numLocales] int = 0;

      forall idx in 0..#myDestLocales.size with (+ reduce valsPerLocale) {
        var destLoc = myDestLocales[idx];
        valsPerLocale[destLoc] += 1;
      }

      maxValsPerLocale = max reduce valsPerLocale;

      forall i in 0..#numLocales {
        numValsReceivingByLocale[i][here.id] = valsPerLocale[i];
      }

    }

    var recvVals: [PrivateSpace] [0..#numLocales] [0..#maxValsPerLocale] eltType;

    // Now we're going to fill the receiving buffers
    // with the data that needs to get transferred from another locale

    coforall loc in Locales do on loc {
      const ref myDestLocales = destLocales[here.id];
      const ref myVals = vals[here.id];
      var idxInDestLoc: [0..#myDestLocales.size] int = 0;
      var numValsPerLocale: [0..#numLocales] int = 0;

      // First we need to figure out what the destination index will be for each value

      for i in 0..#numLocales {
        var onCurrLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then 1 else 0;
        
        var idxInCurrLoc = (+ scan onCurrLoc) - onCurrLoc;
        idxInDestLoc = [j in 0..#myDestLocales.size] if myDestLocales[j] == i then idxInCurrLoc[j] 
                                                     else idxInDestLoc[j];

        numValsPerLocale[i] = + reduce onCurrLoc;
      }

      var sendVals: [0..#numLocales] [0..#maxValsPerLocale] eltType;

      forall idx in 0..#myDestLocales.size {

        var destLoc = myDestLocales[idx];
        var idxInValArr = idxInDestLoc[idx];
        
        sendVals[destLoc][idxInValArr] = myVals[idx];

      }

      // Maybe could be a forall but I don't know how that plays with the bulk transfer.
      for i in 0..#numLocales {
        recvVals[i][here.id][0..#numValsPerLocale[i]] = sendVals[i][0..#numValsPerLocale[i]];
      }

    }

    var returnedVals: [PrivateSpace] list(eltType);

    // Now that the buffers have been filled, we're going to group them together into a single list.
    // Strictly speaking, this probably isn't necessary, but it does make it more friendly to work with

    coforall loc in Locales do on loc {
      const ref numValsReceivedByLoc = numValsReceivingByLocale[here.id];
      const ref myRecvVals = recvVals[here.id];
      var numValsReceived = + reduce numValsReceivedByLoc;
      var myVals: [0..#numValsReceived] eltType;
      var idxOffsetAdjuster = (+ scan numValsReceivedByLoc) - numValsReceivedByLoc;

      for i in 0..#numLocales {

        var idxOffsetThisLoc = idxOffsetAdjuster[i];
        myVals[idxOffsetThisLoc..#numValsReceivedByLoc[i]] = myRecvVals[i][0..#numValsReceivedByLoc[i]];

      }

      returnedVals[here.id] = new list(myVals);

    }

    return returnedVals;

  }

  proc repartitionByLocaleArray(type t,
                                const ref destLocales: [] innerArray(int),
                                const ref vals: [] innerArray(t))
  {
    type eltType = vals.eltType.t;

    var numValsSendingByLocale: [PrivateSpace] [0..#numLocales] int;
    var sendVals: [PrivateSpace] [0..#numLocales] innerArray(t);

    coforall loc in Locales do on loc
    {

      const ref myDestLocales = destLocales[here.id].Arr;
      const ref myVals = vals[here.id].Arr;
      var valsPerLocale: [0..#numLocales] int = 0;

      forall idx in myDestLocales.domain with (+ reduce valsPerLocale) {
        var destLoc = myDestLocales[idx];
        valsPerLocale[destLoc] += 1;
      }

      numValsSendingByLocale[here.id] = valsPerLocale;
      
      var currLocIndAllLocales: [myDestLocales.domain] int;

      /*
      // It would be very cool if we could do things this way:

      for i in 0..#numLocales {
        sendVals[here.id][i] = new innerArray({0..#valsPerLocale[i]}, eltType);

        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;
        
        currLocIndAllLocales += doCurrLoc * currLocInd;
      }

      ref currSendVals = [i in 0..#numLocales] sendVals[here.id][i].Arr;

      forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
        currSendVals[dl][currLocIndAllLocales[j]] = myVals[j];
      }

      // Notice that there's only one forall at the end. This would potentially be faster.
      // But because we can't do an array of refs (currently, at least), there's going to be a forall
      // inside the for, and we do a forall for each of the locales. With all the vectorized stuff
      // we're kind of already doing that so I'm not convinced what follows is significantly slower.
      */

      for i in 0..#numLocales {
        sendVals[here.id][i] = new innerArray({0..#valsPerLocale[i]}, eltType);

        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;
        
        currLocIndAllLocales += doCurrLoc * currLocInd;

        ref currSendVals = sendVals[here.id][i].Arr;

        forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
          if dl == i {
            currSendVals[currLocIndAllLocales[j]] = myVals[j];
          }
        }
      }

    }

    var recvVals: [PrivateSpace] innerArray(eltType);

    // Now we're going to fill the receiving buffers
    // with the data that needs to get transferred from another locale

    coforall loc in Locales do on loc {
      
      const numValsReceivingByLocale = [i in 0..#numLocales] numValsSendingByLocale[i][here.id];
      const valOffsetByLocale = (+ scan numValsReceivingByLocale) - numValsReceivingByLocale;

      recvVals[here.id] = new innerArray({0..#(+ reduce numValsReceivingByLocale)}, eltType);

      ref myRecvVals = recvVals[here.id].Arr;

      for i in 0..#numLocales {

        myRecvVals[valOffsetByLocale[i]..#numValsReceivingByLocale[i]] = sendVals[i][here.id].Arr;
        
      }

    }

    return recvVals;

  }

  proc repartitionByLocaleMultiArray(type t,
                                    const ref destLocales: [] innerArray(int),
                                    const ref vals: [] [] innerArray(t))
  {
    // Notes:
    // - We assume `vals` is indexed as [field][PrivateSpace], i.e., vals[k][here.id]
    // - `innerArray(t)` is your existing wrapper with `.Arr` and
    //   a ctor like new innerArray(dom, eltType)

    type eltType = t;

    const Fields = vals.domain.dim(0);        // range over the "multi" dimension
    const LocaleRange = 0..#numLocales;       // 0..numLocales-1

    var numValsSendingByLocale: [PrivateSpace] [LocaleRange] int;
    var sendVals: [Fields] [PrivateSpace] [LocaleRange] innerArray(eltType);

    // 1) Build per-locale send buffers on each locale (for all fields)
    coforall loc in Locales do on loc {

      const ref myDestLocales = destLocales[here.id].Arr;

      // Count how many elements this locale will send to each destination locale.
      var valsPerLocale: [LocaleRange] int = 0;

      forall idx in myDestLocales.domain with (+ reduce valsPerLocale) {
        const destLoc = myDestLocales[idx];
        valsPerLocale[destLoc] += 1;
      }

      numValsSendingByLocale[here.id] = valsPerLocale;

      // We'll reuse the computed positions for each destination locale.
      var currLocIndAllLocales: [myDestLocales.domain] int = 0;

      // For each destination locale, compute its local indices and fill all fields.
      for i in LocaleRange {
        // Allocate one send buffer per (field, destLocale)
        for k in Fields {
          sendVals[k][here.id][i] = new innerArray({0..#valsPerLocale[i]}, eltType);
        }

        // Boolean mask of entries headed to locale i; then 0-based indices within that subset.
        const doCurrLoc = [j in myDestLocales.domain] myDestLocales[j] == i;
        const currLocInd = (+ scan doCurrLoc) - doCurrLoc;

        // Accumulate the per-position index (same for all fields)
        currLocIndAllLocales += doCurrLoc * currLocInd;

        // Fill every field's send buffer for this dest locale
        for k in Fields {
          const ref myValsK = vals[k][here.id].Arr;
          ref currSendValsK = sendVals[k][here.id][i].Arr;

          forall (j, dl) in zip(myDestLocales.domain, myDestLocales) {
            if dl == i {
              currSendValsK[currLocIndAllLocales[j]] = myValsK[j];
            }
          }
        }
      }
    }

    // 2) Build per-locale receive buffers and splice in segments from every source locale
    var recvVals: [Fields] [PrivateSpace] innerArray(eltType);

    coforall loc in Locales do on loc {
      // For this receiving locale, how many values will arrive from each source locale?
      const numValsReceivingByLocale = [i in LocaleRange] numValsSendingByLocale[i][here.id];
      const valOffsetByLocale = (+ scan numValsReceivingByLocale) - numValsReceivingByLocale;
      const totalIncoming = + reduce numValsReceivingByLocale;

      // Allocate one receive buffer per field
      for k in Fields {
        recvVals[k][here.id] = new innerArray({0..#totalIncoming}, eltType);
      }

      // Splice in, locale by locale, for every field
      for i in LocaleRange {
        const count = numValsReceivingByLocale[i];
        const off   = valOffsetByLocale[i];

        for k in Fields {
          ref myRecvValsK = recvVals[k][here.id].Arr;
          myRecvValsK[off..#count] = sendVals[k][i][here.id].Arr;
        }
      }
    }

    return recvVals;
  }

}
