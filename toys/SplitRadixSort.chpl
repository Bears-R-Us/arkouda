module SplitRadixSort
{

    proc reverseOrder (a: [?aD] ?etype): [aD] etype {
        var result: [aD] etype;
        forall i in aD {
            result[aD.high - i] = a[i]; // this could give wrong result unless range is 0..high
        }
        return result;
    }

    // exclusize-scan 
    proc prescan_add(a: [?aD] bool): [aD] int {
        var s = + scan a;
        var result: [aD] int;
        result[0] = 0; // additive identity in first element
        forall i in aD {
            if i > 0 then result[i] = s[i-1]; // all other elements shifted up one position
        }
        return result;
    }
    
    proc split(a: [?aD] int, iv: [aD] int, flags: [aD] bool): ([aD] int, [aD] int) {
        var nF = !flags;
        var iDown = prescan_add(nF); // needs to be a prescan(exclusive-scan)
        //writeln("iDown = ",iDown);
        var iUp = aD.size - reverseOrder((+ scan (reverseOrder(flags)))); // scan(inclusive-scan)
        //writeln("iUp = ", iUp);

        var perm: [aD] int;
        forall i in aD {
            if flags[i]
            { perm[i] = iUp[i]; }
            else
            { perm[i] = iDown[i]; } // needed to be a prescan(exclusive-scan) so -1
        }
        //writeln("perm = ", perm);
        var result_a: [aD] int;
        result_a[perm]  = a;  // permute a
        var result_iv: [aD] int;
        result_iv[perm] = iv; // permute iv

        return (result_a, result_iv);
    }

    proc splitRadixSort(a: [?aD] int, nBits: int): ([aD] int, [aD] int) {
        //writeln("nBits = ",nBits);
        var a1: [aD] int = a;
        var a2: [aD] int;
        var iv1: [aD] int = [i in aD] i;
        var iv2: [aD] int;

        var flags: [aD] bool;

        for b in {0..#nBits} {
            flags = (((a1 >> b) & 1) == 1); // test bit for this iteration
            //writeln("flags = ", flags);
            (a2, iv2) = split(a1, iv1, flags); // split and shift 1's up and 0's down
            //writeln("a2  = ",  a2);
            //writeln("iv2 = ", iv2);
            a1 = a2;
            iv1 = iv2;
        }

        return (a2, iv2);
    }

}
