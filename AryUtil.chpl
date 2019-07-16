module AryUtil
{
    var printThresh = 30;
    /* print out and array with a name and threshold how many elements are printed */
    proc printAry(name:string, A) {
        if A.size <= printThresh {writeln(name,A);}
        else {writeln(name,[i in A.domain.low..A.domain.low+2] A[i],
                      " ... ", [i in A.domain.high-2..A.domain.high] A[i]);}
    }

    /* test to see if an array is sorted */
    proc isSorted(A:[?D] ?t): bool {
        var sorted: bool;
        sorted = true;
        forall (a,i) in zip(A,D) with (&& reduce sorted) {
            if i > D.low {
                sorted &&= (A[i-1] <= a);
            }
        }
        return sorted;
    }

    /* calculates the (min, max, mean, variance, standard deviation) of an array */
    proc aStats(a: [?D] int): (int,int,real,real,real) {
        var a_min:int = min reduce a;
        var a_max:int = max reduce a;
        var a_mean:real = (+ reduce a:real) / a.size:real;
        var a_vari:real = (+ reduce (a:real **2) / a.size:real) - a_mean**2;
        var a_std:real = sqrt(a_vari);
        return (a_min,a_max,a_mean,a_vari,a_std);
    }
}
