module AryUtil
{
    use Random;
    
    var printThresh = 30;
    
    proc printAry(name:string, A) {
        if A.size <= printThresh {writeln(name,A);}
        else {writeln(name,[i in A.domain.low..A.domain.low+2] A[i],
                      " ... ", [i in A.domain.high-2..A.domain.high] A[i]);}
    }

    proc fillUniform(A:[?D] int, a_min:int ,a_max:int, seed:int=241) {
        // random numer generator
        var R = new owned RandomStream(real, seed); R.getNext();
        [a in A] a = (R.getNext() * (a_max - a_min) + a_min):int;
    }

    proc isSorted(A:[?D] int): bool {
        return (& reduce ([i in zip(A.domain.low .. A.domain.high-1)] A[i] <= A[i+1])); 
    }
}
