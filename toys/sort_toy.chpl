use Random;
use Sort;
use BlockDist;

config const N = 10;

proc print_ary(name:string, A) {
    if A.size <= 10 {writeln(name,A);}
    else {writeln(name,[i in 0..2] A[i]," ... ", [i in A.size-3..A.size-1] A[i]);}
}

record MyComparator
{
    proc key(e: (?etype, int) ): etype {return e[1];}
}
var comp : MyComparator;

var R = new owned RandomStream(real, 241); R.getNext();

var D = newBlockDom({0..#N});
var A: [D] real;
for a in A { a = R.getNext(); }
print_ary("A = ",A);

var EI: [D] (real, int) = [(e,i) in zip(A, A.domain)] (e,i);
print_ary("EI = ",EI);

mergeSort(EI, comparator=comp);
print_ary("sorted EI = ",EI);

var S: [D] real = [(e,i) in EI] A[i];
var IV: [D] int = [(e,i) in EI] i;
print_ary("S = ",S);
print_ary("IV = ",IV);
