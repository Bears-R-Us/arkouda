module AryUtil
{
    var printThresh = 30;
    
    proc printAry(name:string, A) {
        if A.size <= printThresh {writeln(name,A);}
        else {writeln(name,[i in A.domain.low..A.domain.low+2] A[i],
                      " ... ", [i in A.domain.high-2..A.domain.high] A[i]);}
    }

    proc isSorted(A:[?D] ?t): bool {
        var truth: bool;
        truth = true;
        forall i in {D.low .. D.high-1} with (& reduce truth) {
            truth &= (A[i] <= A[i+1]);
        }
        return truth;
    }

    proc isSorted_1(A:[?D] ?t): bool {
        return (& reduce ([i in {D.low .. D.high-1}] A[i] <= A[i+1])); 
    }

    proc isSorted_3(A:[?D] ?t): bool {
        var truth: [D] bool;
        truth[D.high] = true;
        [i in {D.low .. D.high-1}] truth[i] = A[i] <= A[i+1];
        return (& reduce truth); 
    }

    proc aStats(a: [?D] int): (int,int,real,real,real) {
        var a_min:int = min reduce a;
        var a_max:int = max reduce a;
        var a_mean:real = (+ reduce a:real) / a.size:real;
        var a_vari:real = (+ reduce (a:real **2) / a.size:real) - a_mean**2;
        var a_std:real = sqrt(a_vari);
        return (a_min,a_max,a_mean,a_vari,a_std);
    }
}
