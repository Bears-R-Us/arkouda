module Repartition
{

  // The goal of this module is to provide helper functions to facilitate redistributing data between
  // locales.

  use PrivateDist;
  use List;

  proc repartitionByLocaleString(const ref destLocales: [PrivateSpace] list(int),
                                 const ref strOffsets: [PrivateSpace] list(int),
                                 const ref strBytes: [PrivateSpace] list(uint(8))):
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
      const ref myDestLocales = destLocales[here.id];
      const ref myStrOffsets = strOffsets[here.id];
      const ref myStrBytes = strBytes[here.id];
      var bytesPerLocale: [0..#numLocales] int = 0;
      var stringsPerLocale: [0..#numLocales] int = 0;
      var sizes: [0..#myDestLocales.size] int = 0;

      forall idx in 0..#myDestLocales.size with (+ reduce bytesPerLocale, + reduce stringsPerLocale) {
        var destLoc = myDestLocales[idx];
        var size = if idx == myDestLocales.size - 1 then myStrBytes.size - myStrOffsets[idx]
                   else myStrOffsets[idx + 1] - myStrOffsets[idx];
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
      const ref myDestLocales = destLocales[here.id];
      const ref myStrOffsets = strOffsets[here.id];
      const ref myStrBytes = strBytes[here.id];
      const ref mySizes = allStrSizes[here.id];
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

        forall j in 0..#numStringsReceivedByLoc[i] {
          myOffsets[j + idxOffsetThisLoc] = myRecvOffsets[i][j] + byteOffsetThisLoc;
          var size = if j == numStringsReceivedByLoc[i] - 1
                     then numBytesReceivedByLoc[i] - myRecvOffsets[i][j]
                     else myRecvOffsets[i][j + 1] - myRecvOffsets[i][j];
          myBytes[(myRecvOffsets[i][j] + byteOffsetThisLoc)..#size]
            = myRecvBytes[i][myRecvOffsets[i][j]..#size];
        }

      }

      returnedOffsets[here.id] = new list(myOffsets);
      returnedBytes[here.id] = new list(myBytes);

    }

    return (returnedOffsets, returnedBytes);

  }
}
