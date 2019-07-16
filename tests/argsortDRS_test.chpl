use ArgsortDRS;
use Sort only isSorted;
use BlockDist;
use Random;
use Math only abs;

config const LEN = 1_000_000;

writeln("Initializing...");
var D = {0..#LEN};
var DD: domain(1) dmapped Block(boundingBox=D) = D;
var A: [DD] int;
fillRandom(A);
// Squeeze values into [0, 2**32)
A = abs(A) % 2**32;
var aMin = min reduce A;
var aMax = max reduce A;
writeln("A = ", A[0..#5]);
writeln("aMin = ", aMin, ", aMax = ", aMax);

writeln("Sorting with argsortDRS...");
var iv = argsortDRS(A, aMin, aMax);
var B = A[iv];
writeln("Result is sorted? ", isSorted(B));