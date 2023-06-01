/*
  This file contains the code for the examples in the associated guide, `CHAPEL_TUTORIAL.md`.
  To start, uncomment the relevant example and navigate to the directory containng this file.

  The compilation command is:
    chpl Tutorial.chpl -o tutorial
  The execution command is:
    ./tutorial
*/

/*
  Example 1: Range 5..10
*/
// for i in 5..10 {
//   writeln(i);
// }

/*
  Example 2: Range 5..#10
*/
// for i in 5..#10 {
//   writeln(i);
// }

/*
  Example 3: Initial Factorial
*/
// proc factorial(n: int) {
//   var fact: int = 1;
//   for i in 1..n {
//     fact *= i;
//   }
//   return fact;
// }
// writeln(factorial(5));

/*
  Example 4: Parallel Factorial Attempt
  just replace `for` with `forall`... what's the worst that can happen
*/
// proc factorial(n: int) {
//   var fact: int = 1;
//   forall i in 1..n {
//     fact *= i;
//   }
//   return fact;
// }
// writeln(factorial(5));

/*
  Example 5: `reduce` and `scan`
*/
// var a: [0..#5] int = [1, 2, 3, 4, 5];
// writeln("a = ", a, "\n");
// writeln("+ scan a = ", + scan a);
// writeln("+ reduce a = ", + reduce a, "\n");
// writeln("* scan a = ", * scan a);
// writeln("* reduce a = ", * reduce a, "\n");
// writeln("min scan a = ", min scan a);
// writeln("min reduce a = ", min reduce a, "\n");
// writeln("max scan a = ", max scan a);
// writeln("max reduce a = ", max reduce a);

/*
  Example 6: Factorial with must-parallel `forall` and Reduction
*/
// proc factorial(n: int) {
//   var fact: int = 1;
//   forall i in 1..n with (* reduce fact) {
//     fact *= i;
//   }
//   return fact;
// }
// writeln(factorial(5));

/*
  Example 7: may-parallel `forall`
*/
// [i in 0..10] {
//   writeln(i);
// }

/*
  Example 8: Forall Expressions
*/
// // must-parallel forall expression
// var tens = forall i in 1..10 do i*10;
// writeln(tens);
// // may-parallel forall expression
// var negativeTens = [i in tens] -i;
// writeln(negativeTens);

/*
  Example 9: Factorial with may-parallel `forall` Expression and Reduction
*/
// proc factorial(n: int) {
//   return * reduce [i in 1..n] i;
// }
// writeln(factorial(5));

/*
  Try It Yourself 1: Perfect Squares <=25
  Compute and print out all perfect squares less than or equal to `25`
  Bonus points if you can do it in one line using `forall` expressions and reductions!

  Expected output:
  0 1 4 9 16 25
*/

/*
  Example 10: `coforall` loop
*/
// const numTasks = 8;

// coforall tid in 1..numTasks do
//   writeln("Hello from task ", tid, " of ", numTasks);

// writeln("Signing off...");

/*
  Example 11: Absolute Value Ternary
*/
// proc absoluteVal(n:int) {
//   return if n >= 0 then n else -n;
// }
// writeln(absoluteVal(-15));
// writeln(absoluteVal(7));

/*
  Example 12: Ternary and `forall` Expression
*/
// writeln([i in 0..#10] if i%2 == 0 then i+10 else -100);

/*
  Try It Yourself 2: Array Absolute Value
  Write a `proc` using a ternary to take an `int array`, `A`, and return an array where index `i` is the absolute value of `A[i]`

  Call: `arrayAbsVal([-3, 7, 0, -4, 12]);`

  Expected output:
  3 7 0 4 12
 */
// proc arrayAbsVal(A) {
//   // your code here
// }

// writeln(arrayAbsVal([-3, 7, 0, -4, 12]));


/*
  Example 13: Introspection
*/
// proc absoluteVal(a: [?D] ?t): [D] t {
//   return [i in D] if a[i] >= 0 then a[i] else -a[i];
// }
// var r: [0..#5] real = [-3.14, 7:real, 0.0, INFINITY, -INFINITY];
// writeln(absoluteVal(r));

// var i: [0..#5] int = [-3, 7, 0, -4, 12];
// writeln(absoluteVal(i));

/*
  Example 14: Promotion
*/
// proc factorial(n: int) {
//   return * reduce [i in 1..#n] i;
// }
// writeln(factorial(5));
// writeln(factorial([1, 2, 3, 4, 5]));

/*
  Example 15: Filtering
*/
// writeln([i in 0..#10] if i%2 == 0 then -i);

/*
  Try It Yourself 3: Sum Odd Perfect Squares <=25
  Use filtering and reduce to sum all odd perfect squares less than or equal to `25`

  Expected output:
  35
*/

/*
  Example 16: Looping Locales with `coforall`
*/
// use BlockDist;

// // we create a block distributed array and populate with values from 1 to 16
// var A = Block.createArray({1..16}, int);
// A = 1..16;

// // we use a coforall to iterate over the Locales creating one task per
// coforall loc in Locales {
//   on loc {  // Then we use an `on` clause to execute on Locale `loc`
//     // Next we create `localA` by slicing `A` at it's local subdomain
//     const localA = A[A.localSubdomain()];
//     writeln("The chunk of A owned by Locale ", loc.id, " is: ", localA);
//   }
// }

/*
  Example 17: Simple Zippered Iteration
*/
// var A: [1..5] real;
// for (a, i, j) in zip(A, 1.., [3, 0, 1, 2, 4]) {
//   a = i**j;
// }
// writeln(A); 

/*
  Example 18: Zippered Interation in Arkouda
  based on `getLengths` in `SegmentedString`
*/
// const values: [0..#12] string = ['s', 's', 's', '0', '\x00', 's', 's', '1', '\x00', 's', '2', '\x00'],
//       offsets = [0, 5, 9],
//       size = offsets.size;  // size refers to the number of stings in the Segstring, this is always equivalent to offsets.size

// /* Return lengths of all strings, including null terminator. */
// proc getLengths() {
//   // initialize lengths with the same domain as offsets
//   var lengths: [offsets.domain] int;
//   if size == 0 {
//     // if the segstring is empty, the lengths are also empty
//     return lengths;
//   }
//   // save off the last index of offsets
//   const high = offsets.domain.high;
//   forall (i, o, l) in zip(offsets.domain, offsets, lengths) {
//     if i == high {
//       // the last string
//       // len = values.size - start position of this string
//       l = values.size - o;
//     }
//     else {
//       // not the last string
//       // len = start position of next string - start position of this string
//       l = offsets[i+1] - o;
//     }
//   }
//   return lengths;
// }

// writeln(getLengths());

/*
  Example 19: Aggregation in Arkouda
  based on `upper` in `SegmentedString`
  This example uses `getLengths` from Example 15, so be sure to uncomment that example as well.
*/
// use CopyAggregation;

// /*
//   Given a SegString, return a new SegString with all lowercase characters from the original replaced with their uppercase equivalent
//   :returns: Strings – Substrings with lowercase characters replaced with uppercase equivalent
// */
// proc upper() {
//   var upperVals: [values.domain] string;
//   const lengths = getLengths();
//   forall (off, len) in zip(offsets, lengths) with (var valAgg = new DstAggregator(string)) {
//     var i = 0;
//     for char in ''.join(values[off..#len]).toUpper() {
//       valAgg.copy(upperVals[off+i], char);
//       i += 1;
//     }
//   }
//   return (offsets, upperVals);
// }

// writeln("Old Vals:", values);
// var (newOffs, newVals) = upper();
// writeln("New Vals:", newVals);


/*
  Try It Yourself 4: `title`
  Use Aggregation to transform the SegString from [example 15](#ex15) into title case.
  Be sure to use the chapel builtin [`toTitle`](https://chapel-lang.org/docs/language/spec/strings.html?highlight=totitle#String.string.toTitle)
  and `getLengths` from example 15.

  Expected output:
  S s s 0  S s 1  S 2
*/
// proc title() {
//   // your code here!
// }
// writeln(title());

/*
  Example 20: Boolean Compression Indexing
*/
// var X = [1, 2, 5, 5, 1, 5, 2, 5, 3, 1];
// writeln("X = ", X, "\n");

// // we begin by creating a boolean array, `truth`, indicating where the condition is met
// var truth = X == 5;
// writeln("truth = ", truth);

// // we use `truth` to create the indices, `iv`, into the compressed array
// // `+ scan truth - truth` is essentially creating an exclusive scan
// // note: `iv[truth] = [0, 1, 2, 3]`
// var iv = + scan truth - truth;
// writeln("iv = ", iv);
// writeln("iv[truth] = ", [(t, v) in zip(truth, iv)] if t then v, "\n");

// // we then create the return array `Y`
// // it contains all the elements where the condition is met
// // so its size is the number of `True`s i.e. `+ reduce truth`
// var Y: [0..#(+ reduce truth)] int;
// writeln("+ reduce truth = ", + reduce truth);
// writeln("0..#(+ reduce truth) = ", 0..#(+ reduce truth), "\n");

// // now that we have the setup, it's time for the actual indexing
// // we do a may-parallel `forall` to iterate over the indices of `X`
// // we filter on `truth[i]`, so we only act if the condition is met
// // we use the compressed indices `iv[i]` to write into `Y`
// // while using the original indices `i` to get the correct value from `X`
// [i in X.domain] if truth[i] {Y[iv[i]] = X[i];}

// // note we could do the same thing with zippered iteration
// // since `truth`, `X`, and `iv` have the same domain
// // [(t, x, v) in zip(truth, X, iv)] if t {Y[v] = x;}

// writeln("Y = ", Y);

/*
  Example 21: Boolean Expansion Indexing
*/
// var X = [1, 2, 5, 5, 1, 5, 2, 5, 3, 1];
// var Y = [-9, -8, -7, -6];
// writeln("X = ", X);
// writeln("Y = ", Y, "\n");

// // we begin by creating a boolean array, `truth`, indicating where the condition is met
// var truth = X == 5;
// writeln("truth = ", truth);

// // we use `truth` to create the indices, `iv`, into the compressed array
// // `+ scan truth - truth` is essentially creating an exclusive scan
// // note: `iv[truth] = [0, 1, 2, 3]`
// var iv = + scan truth - truth;
// writeln("iv = ", iv);
// writeln("iv[truth] = ", [(t, v) in zip(truth, iv)] if t then v, "\n");

// // now that we have the setup, it's time for the actual indexing
// // this is equivalent to compression indexing with the assignment swapped
// // we do a may-parallel `forall` to iterate over the indices of `X`
// // we filter on `truth[i]`, so we only act if the condition is met
// // we use the original indices `i` to write into `X`
// // while using the compressed indices `iv[i]` to get the correct value from `Y`
// [i in X.domain] if truth[i] {X[i] = Y[iv[i]];}

// // note we could do the same thing with zippered iteration
// // since `truth`, `X`, and `iv` have the same domain
// // [(t, x, v) in zip(truth, X, iv)] if t {x = Y[v];}

// writeln("X = ", X);

/*
  Try It Yourself 5: Array Even Replace 
  Create a `proc` which given two int arrays `A` and `B` with different domains
  will return `A` but with the even values replaced with the values of `B`

  You should aim to use as many of the concepts for the guide as possible:
  - boolean expansion indexing
  - forall
  - filtering
  - scan
  - introspection
  - zippered iteration

  Call:
  - `arrayEvenReplace([8, 9, 7, 2, 4, 3], [17, 19, 21]);`
  - `arrayEvenReplace([4, 4, 7, 4, 4, 4], [9, 9, 9, 9, 9]);`

  Expected output:
  17 9 7 19 21 3
  9 9 7 9 9 9
*/
// proc arrayEvenReplace(A: , B: ) {
//   // your code here!
// }

// writeln(arrayEvenReplace([8, 9, 7, 2, 4, 3], [17, 19, 21]));
// writeln(arrayEvenReplace([4, 4, 7, 4, 4, 4], [9, 9, 9, 9, 9]));

/*
  Example 22: Variable Declarations
*/
// // pretend myBool is determined during runtime
// var myBool = true;

// proc helper(myBool: bool) {
//     return if myBool then 5 else 10;
// }

// // use a var if you expect a value to change
// var myVar = [0, 1, 2];
// // we use a const because we don't know the value at compilation time
// const myConst = helper(myBool);
// // we use a param becasue we know what the value is at compilation time
// param myParam = 17;

// // if we want a copy of myVar we can create a new var based on it
// // this results in more memory usage (because we are creating a new array)
// // but changes to myCopy won't change myVar
// var myCopy = myVar;
// myCopy[1] = 100;
// // we see myVar is unchanged
// writeln("myVar: ", myVar);

// // we use a ref if we do want changes to myRef to update myVar
// // This save us from having to create a whole new array
// ref myRef = myVar;
// myRef[1] = -2000;
// writeln("myVar: ", myVar);

/*
  Example 23: Diagnostics
*/
// use BlockDist, CommDiagnostics, Time;

// var A: [Block.createDomain({0..7})] int = 0..15 by 2;
// var B: [Block.createDomain({0..15})] int = 0..15;
// writeln("A = ", A);
// writeln();
// writeln("B = ", B);
// writeln();

// resetCommDiagnostics();
// startCommDiagnostics();
// var t1 = Time.timeSinceEpoch().totalSeconds();

// forall (a, i) in zip(A, A.domain) {
//   B[B.size - (2*i + 1)] = a;
// }

// var t2 = Time.timeSinceEpoch().totalSeconds() - t1;
// stopCommDiagnostics();
// writeln("Copy without aggregation time = ", t2);
// writeln();
// printCommDiagnosticsTable();
// writeln("B = ", B);

/*
  Example 24: Aggregation Reducing Communication
*/
// use BlockDist, CommDiagnostics, Time, CopyAggregation;

// config param SIZE = 1000000;
// var A: [Block.createDomain({0..#(SIZE / 2)})] int = 0..#SIZE by 2;
// var B: [Block.createDomain({0..#SIZE})] int = 0..#SIZE;

// resetCommDiagnostics();
// startCommDiagnostics();
// var t1 = Time.timeSinceEpoch().totalSeconds();

// forall (a, i) in zip(A, A.domain) {
//   B[B.size - (2*i + 1)] = a;
// }

// var t2 = Time.timeSinceEpoch().totalSeconds() - t1;
// stopCommDiagnostics();
// writeln("Copy without aggregation time = ", t2);
// writeln();
// printCommDiagnosticsTable();

// resetCommDiagnostics();
// startCommDiagnostics();
// t1 = Time.timeSinceEpoch().totalSeconds();

// forall (a, i) in zip(A, A.domain) with (var agg = new DstAggregator(int)) {
//   agg.copy(B[B.size - (2*i + 1)], a);
// }

// t2 = Time.timeSinceEpoch().totalSeconds() - t1;
// stopCommDiagnostics();
// writeln("Copy with aggregation time = ", t2);
// writeln();
// printCommDiagnosticsTable();

/*
  Example 25: Common Pitfalls
*/
// use BlockDist, CommDiagnostics, Time;

// // simplified symenty class
// class SymEntry {
//     type etype;
//     var a;

//     proc init(len: int, type etype) {
//         this.etype = etype;
//         this.a = Block.createArray({0..#len}, etype);
//     }

//     proc init(in a: [?D] ?etype) {
//         this.etype = etype;
//         this.a = a;
//     }
// }

// config param SIZE = 1000000;
// const distDom = Block.createDomain({0..#SIZE});

// // create a array containing tuples of uints
// var hashes: [distDom] (uint, uint) = (1, 1):(uint, uint);

// var upperEntry = new SymEntry(SIZE, uint);
// var lowerEntry = new SymEntry(SIZE, uint);

// resetCommDiagnostics();
// startCommDiagnostics();
// var t1 = Time.timeSinceEpoch().totalSeconds();

// // the leading iterator is a range, so all the computation happens on locale 0
// forall (i, (up, low)) in zip(0..#SIZE, hashes) {
//   upperEntry.a[i] = up;
//   lowerEntry.a[i] = low;
// }

// var t2 = Time.timeSinceEpoch().totalSeconds() - t1;
// stopCommDiagnostics();
// writeln("leading iterator not distributed time = ", t2);
// writeln();
// printCommDiagnosticsTable();

// resetCommDiagnostics();
// startCommDiagnostics();
// t1 = Time.timeSinceEpoch().totalSeconds();

// // leading iterator is distributed
// // but every iteration access the `.a` component
// forall (i, (up, low)) in zip(hashes.domain, hashes) {
//   upperEntry.a[i] = up;
//   lowerEntry.a[i] = low;
// }

// t2 = Time.timeSinceEpoch().totalSeconds() - t1;
// stopCommDiagnostics();
// writeln("leading iterator is distributed, but .a accesses time = ", t2);
// writeln();
// printCommDiagnosticsTable();

// resetCommDiagnostics();
// startCommDiagnostics();
// t1 = Time.timeSinceEpoch().totalSeconds();

// // use refs to avoid repeated accesses
// ref ua = upperEntry.a;
// ref la = lowerEntry.a;
// forall (i, (up, low)) in zip(hashes.domain, hashes) {
//   ua[i] = up;
//   la[i] = low;
// }

// t2 = Time.timeSinceEpoch().totalSeconds() - t1;
// stopCommDiagnostics();
// writeln("using refs time = ", t2);
// writeln();
// printCommDiagnosticsTable();


// var upper: [distDom] uint;
// var lower: [distDom] uint;
// resetCommDiagnostics();
// startCommDiagnostics();
// t1 = Time.timeSinceEpoch().totalSeconds();

// // iterate over arrays directly:
// // since they are distributed the same way,
// // the looping variables will always be local to each other
// forall (up, low, h) in zip(upper, lower, hashes) {
//   (up, low) = h;
// }

// t2 = Time.timeSinceEpoch().totalSeconds() - t1;
// stopCommDiagnostics();
// writeln("looping over arrays directly time = ", t2);
// writeln();
// printCommDiagnosticsTable();

// upperEntry = new SymEntry(upper);
// lowerEntry = new SymEntry(lower);
