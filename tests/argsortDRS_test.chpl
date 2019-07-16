use ArgsortDRS;
use Sort only sort, isSorted;
use BlockDist;
use Random;
use Time;
use Math only abs;

config const LEN = 1_000_000;

writeln("Initializing...");
var D = {0..#LEN};
var DD: domain(1) dmapped Block(boundingBox=D) = D;
var A: [D] int;
fillRandom(A);
A = abs(A) % 2**32;
var aMin = min reduce A;
var aMax = max reduce A;
writeln("A = ", A[0..#5]);
writeln("aMin = ", aMin, ", aMax = ", aMax);

writeln("Sorting with argsortDRS...");
var iv = argsortDRS(A, aMin, aMax);
var B = A[iv];
writeln("Result is sorted? ", isSorted(B));

writeln("Using builtin sort...");
var t = new Timer();
t.start();
var AI = [(a, i) in zip(A, DD)] (a, i);
sort(AI);
var IV = [(a, i) in AI] i;
var C = A[IV];
t.stop();
writeln(t.elapsed(), " seconds");
writeln("Array is sorted? ", isSorted(C));