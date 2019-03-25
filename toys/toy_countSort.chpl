// Number of samples
config const N = 20;
// Minimum Value
config const M = 0;
// ValueRange
config const VR = 20; // 20 possible values are in [M,M+VR)

use CyclicDist;
use CountingSort;
use AryUtil;
use Memory;  // for physicalMemory()

config const printLocaleInfo = true;  // permit testing to turn this off

if printLocaleInfo {
  for loc in Locales do
    on loc {
      writeln("locale #", here.id, "...");
      writeln("  ...is named: ", here.name);
      writeln("  ...has ", here.numPUs(), " processor cores");
      writeln("  ...has ", here.physicalMemory(unit=MemUnits.GB, retType=real),
              " GB of memory");
      writeln("  ...has ", here.maxTaskPar, " maximum parallelism");
    }
}

writeln();

// domain for A
var D = {0..#N} dmapped Cyclic(startIdx=0);
// create and fill A with values from interval [M,M+VR)
var A: [D] int;
fillUniform(A,M,M+VR);
printAry("A = ",A);

// sort A and return the permutation in VI
var IV = argCountSort(A);
printAry("IV = ",IV);
// permute A with IV
var S = A[IV];
printAry("S = ",S);

writeln("passed = ",isSorted(S));

