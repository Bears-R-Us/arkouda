/*
  This file contains the code for the examples in the associated guide, `CHAPEL_TUTORIAL.md`.
  To start, uncomment the relevant example and navigate to the directory containng this file.

  The compilation command is:
    chpl tutorial.chpl
  The execution command is:
    ./tutorial

  Some later examples will provide additional instructions to add flags to these commands.
*/

/**********************
 * Ranges and Domains *
 **********************/

// for i in 5..10 {
//   writeln(i);
// }

// for i in 5..<10 {
//   writeln(i);
// }

// for i in 5..#10 {
//   writeln(i);
// }

/**************
 * Procedures *
 **************/

/* Serial Factorial */

// proc factorial(n: int) {
//   var fact: int = 1;
//   for i in 1..n {
//     fact *= i;
//   }
//   return fact;
// }
// writeln(factorial(5));

/* Parallel Factorial Attempt */
// just replace `for` with `forall`... what's the worst that can happen 
 
// proc factorial(n: int) {
//   var fact: int = 1;
//   forall i in 1..n {
//     fact *= i;
//   }
//   return fact;
// }
// writeln(factorial(5));

/***********************
 * `reduce` and `scan` *
 ***********************/

// var a: [0..<5] int = [1, 2, 3, 4, 5];
// writeln("a = ", a, "\n");
// writeln("+ scan a = ", + scan a);
// writeln("+ reduce a = ", + reduce a, "\n");
// writeln("* scan a = ", * scan a);
// writeln("* reduce a = ", * reduce a, "\n");
// writeln("min scan a = ", min scan a);
// writeln("min reduce a = ", min reduce a, "\n");
// writeln("max scan a = ", max scan a);
// writeln("max reduce a = ", max reduce a);

/******************
 * 'forall' Loops *
 ******************/

/* Factorial with must-parallel `forall` and Reduction */

// proc factorial(n: int) {
//   var fact: int = 1;
//   forall i in 1..n with (* reduce fact) {
//     fact *= i;
//   }
//   return fact;
// }
// writeln(factorial(5));

/* may-parallel `forall` */

// [i in 0..10] {
//   writeln(i);
// }

/* Forall Expressions */

// // must-parallel forall expression
// var tens = forall i in 1..10 do i*10;
// writeln(tens);
// // may-parallel forall expression
// var negativeTens = [i in tens] -i;
// writeln(negativeTens);

/* Factorial with may-parallel `forall` Expression and Reduction */

// proc factorial(n: int) {
//   return * reduce [i in 1..n] i;
// }
// writeln(factorial(5));

/*
  Try It Yourself: Perfect Squares <=25
  Compute and print out all perfect squares less than or equal to `25`
  Bonus points if you can do it in one line using `forall` expressions and reductions!

  Expected output:
  0 1 4 9 16 25
*/

/**********************
 * Zippered Iteration *
 **********************/

// var A: [1..5] real;
// for (a, i, j) in zip(A, 1.., [3, 0, 1, 2, 4]) {
//   a = i**j;
// }
// writeln(A);

/***********
 * Ternary *
 ***********/

/* Absolute Value Ternary */

// proc absoluteVal(n:int) {
//   return if n >= 0 then n else -n;
// }
// writeln(absoluteVal(-15));
// writeln(absoluteVal(7));

/* Ternary and `forall` Expression */

// writeln([i in 0..<10] if i%2 == 0 then i+10 else -100);

/*
  Try It Yourself: Array Absolute Value
  Write a `proc` using a ternary to take an `int array`, `A`, and return an array where index `i` is the absolute value of `A[i]`

  Call: `arrayAbsVal([-3, 7, 0, -4, 12]);`

  Expected output:
  3 7 0 4 12
 */

// proc arrayAbsVal(A: [] int) {
//   // your code here
// }

// writeln(arrayAbsVal([-3, 7, 0, -4, 12]));

/******************************
 * Generics and Introspection *
 ******************************/

/* Generics */

// proc double(a) {
//   return a * 2;
// }

// writeln(double(-100));
// writeln(double(7.5));
// writeln(double("oh no! we don't want strings!"));

/* Introspection */

// proc double(a: ?t) where t == int || t == uint || t == real {
//   return a * 2;
// }

// writeln(double(-100));
// writeln(double(7.5));

// // Verify this breaks!
// writeln(double("oh no! we don't want strings!"));

/*
  Try It Yourself: Array Absolute Value with Introspection
  Let's build upon our Array Absolute Value Try It Yourself!

  We have a `proc` which takes an `int array`, `A`, and returns the index-wise absolute value.
  Modify it to also accept a `real array`.

  Call:
  arrayAbsVal([-3.14, 7:real, 0.0, inf, -inf]);
  arrayAbsVal([-3, 7, 0, -4, 12]);

  Expected output:
  3.14 7.0 0.0 inf inf
  3 7 0 4 12
 */


// writeln(arrayAbsVal([-3.14, 7:real, 0.0, inf, -inf]));
// writeln(arrayAbsVal([-3, 7, 0, -4, 12]));

/*************
 * Promotion *
 *************/

// proc factorial(n: int) {
//   return * reduce [i in 1..#n] i;
// }
// writeln(factorial(5));
// writeln(factorial([1, 2, 3, 4, 5]));

/*
  Try It Yourself: Absolute Value with Promotion
  
  Write an absolute value `proc` which uses promotion to accept either a single `real` value or a `real` array. 

  Call:
  absoluteVal(-inf);
  arrayAbsVal([-3.14, 7:real, 0.0, inf, -inf]);

  Expected output:
  inf
  3.14 7.0 0.0 inf inf
 */


// writeln(arrayAbsVal(-inf));
// writeln(arrayAbsVal([-3.14, 7:real, 0.0, inf, -inf]));

/*************
 * Filtering *
 *************/

// writeln([i in 0..<10] if i%2 == 0 then -i);

/*
  Try It Yourself: Sum Odd Perfect Squares <=25
  Use filtering and reduce to sum all odd perfect squares less than or equal to `25`

  Expected output:
  35
*/

/**********************************************
 * Boolean Compression and Expansion Indexing *
 **********************************************/


/* Boolean Compression Indexing */

// var X = [1, 2, 5, 5, 1, 5, 2, 5, 3, 1];
// writeln("X = ", X, "\n");

// // we begin by creating a boolean array, `truth`, indicating where the condition is met
// var truth = (X == 5);
// writeln("truth = ", truth);

// // we use `truth` to create the indices, `iv`, into the compressed array
// // `+ scan truth - truth` is essentially creating an exclusive scan
// // note: `iv[truth] = [0, 1, 2, 3]`
// var iv = (+ scan truth) - truth;
// writeln("iv = ", iv);
// writeln("iv[truth] = ", [(t, v) in zip(truth, iv)] if t then v, "\n");

// // we then create the return array `Y`
// // it contains all the elements where the condition is met
// // so its size is the number of `True`s i.e. `+ reduce truth`
// var Y: [0..<(+ reduce truth)] int;
// writeln("+ reduce truth = ", + reduce truth);
// writeln("0..<(+ reduce truth) = ", 0..<(+ reduce truth), "\n");

// // now that we have the setup, it's time for the actual indexing
// // we use a forall to iterate over the indices of `X`
// // we only act if the condition is met i.e. truth[i] is true
// // we then use the compressed indices `iv[i]` to write into `Y`
// // while using the original indices `i` to get the correct value from `X`
// forall i in X.domain {
//   if truth[i] {
//     Y[iv[i]] = X[i];
//   }
// }

// // NOTE:
// // we could also use zippered iteration here since
// // `truth`, `X`, and `iv` have the same domain.
// // Using that and a may-parallel `forall` gives: 
// // [(t, x, v) in zip(truth, X, iv)] if t {Y[v] = x;}

// writeln("Y = ", Y);

/* Boolean Expansion Indexing */

// var X = [1, 2, 5, 5, 1, 5, 2, 5, 3, 1];
// var Y = [-9, -8, -7, -6];
// writeln("X = ", X);
// writeln("Y = ", Y, "\n");

// // we begin by creating a boolean array, `truth`, indicating where the condition is met
// var truth = (X == 5);
// writeln("truth = ", truth);

// // we use `truth` to create the indices, `iv`, into the compressed array
// // `+ scan truth - truth` is essentially creating an exclusive scan
// // note: `iv[truth] = [0, 1, 2, 3]`
// var iv = (+ scan truth) - truth;
// writeln("iv = ", iv);
// writeln("iv[truth] = ", [(t, v) in zip(truth, iv)] if t then v, "\n");

// // now that we have the setup, it's time for the actual indexing
// // notice this is equivalent to compression indexing with the assignment swapped
// // we use a forall to iterate over the indices of `X`
// // we only act if the condition is met i.e. truth[i] is true
// // we use the original indices `i` to write into `X`
// // while using the compressed indices `iv[i]` to get the correct value from `Y`
// forall i in X.domain {
//   if truth[i] {
//     X[i] = Y[iv[i]];
//   }
// }

// // NOTE:
// // we could also use zippered iteration here since
// // `truth`, `X`, and `iv` have the same domain.
// // Using that and a may-parallel `forall` gives: 
// // [(t, x, v) in zip(truth, X, iv)] if t {x = Y[v];}

// writeln("X = ", X);

/*
  Try It Yourself: Array Even Replace 
  Use the following function signature to create a `proc`
  
  Then replace the even values of `A` with the values of `B` and return `A`.
  You can assume the size of `B` will be equal to number of even values in `A`.

  It may be helpful to review boolean expansion indexing

  Note:
  We use an [`in` argument intent](https://chapel-lang.org/docs/primers/procedures.html#argument-intents)
  in the function signature to allow us to modify `A`.

  Call:
  - `arrayEvenReplace([8, 9, 7, 2, 4, 3], [17, 19, 21]);`
  - `arrayEvenReplace([4, 4, 7, 4, 4, 4], [9, 9, 9, 9, 9]);`

  Expected output:
  17 9 7 19 21 3
  9 9 7 9 9 9
*/

// proc arrayEvenReplace(in A: [] int, B: [] int) {}
//   // your code here!
// }

// writeln(arrayEvenReplace([8, 9, 7, 2, 4, 3], [17, 19, 21]));
// writeln(arrayEvenReplace([4, 4, 7, 4, 4, 4], [9, 9, 9, 9, 9]));


/********************************
 * Locales and `coforall` loops *
 ********************************/

/* Simple `coforall` */

// var numTasks = 8;

// coforall tid in 1..numTasks {
//   writeln("Hello from task ", tid, " of ", numTasks);
// }

// writeln("Signing off...");

/* Looping Locales with `coforall` */

// use BlockDist;

// // we create a block distributed array and populate with values from 1 to 16
// var A = blockDist.createArray({1..16}, int);
// A = 1..16;

// // we use a coforall to iterate over the Locales creating one task per
// coforall loc in Locales {
//   on loc {  // Then we use an `on` clause to execute on Locale `loc`
//     // Next we create `localA` by slicing `A` at it's local subdomain
//     const localA = A[A.localSubdomain()];
//     writeln("The chunk of A owned by Locale ", loc.id, " is: ", localA);
//   }
// }

/* Implicit distributed computation with `forall` */

// use BlockDist;

// var MyDistArr = blockDist.createArray({1..16}, int);
// MyDistArr = 1..16;

// forall i in MyDistArr.domain {
//   writeln("element ", i, " (", MyDistArr[i], ") is owned by locale ", here.id);
// }

/***************
 * Aggregation *
 ***************/

// use BlockDist, CopyAggregation;

// config const UseDstAgg = true;

// const dom = blockDist.createDomain({0..<6});

// // named src because this is the source we are copying from
// var src: [dom] int = [0, 1, 2, 3, 4, 5];

// // named dst because this is the destination we are copying to
// var dst: [dom] int;

// writeln("src: ", src);
// writeln("dst: ", dst);

// if UseDstAgg {
//     // when the destination is remote we use a dstAggregator
//     forall (s, i) in zip(src, 0..) with (var agg = new DstAggregator(int)) {
//       // locNum is which locale this loop iteration is executing on
//       var locNum = here.id;

//       // localSubDom is the chunk of the distributed arrays that live on this locale
//       var localSubDom = dom.localSubdomain();
      
//       // we use a single writeln to avoid interleaving output from another locale 
//       writeln("\niteration num: ", i, "\n  on Locale: ", locNum,
//               "\n  on localSubDom: ", localSubDom, "\n  src[", i, "] is local",
//               "\n  dst[", (i + 3) % 6, "] is remote");
    
//       // since dst is remote, we use a dst aggregator
//       // assignment without aggregation would look like:
//       // dst[ (i + 3) % 6 ] = s
//       agg.copy(dst[ (i + 3) % 6 ], s);
//     }
//     writeln();
//     writeln("src: ", src);
//     writeln("dst: ", dst);
// }
// else {
//     // when the source is remote we use a srcAggregator
//     forall (d, i) in zip(dst, 0..) with (var agg = new SrcAggregator(int)) {
//       // locNum is which locale this loop iteration is executing on
//       var locNum = here.id;
//       // localSubDom is the chunk of the distributed arrays that live on this locale
//       var localSubDom = dom.localSubdomain();
      
//       // we use a single writeln to avoid interleaving output from another locale 
//       writeln("\niteration num: ", i, "\n  on Locale: ", locNum,
//               "\n  on localSubDom: ", localSubDom, "\n  src[", (i + 3) % 6, "] is remote",
//               "\n  dst[", i, "] is local");
    
//       // since src is remote, we use a src aggregator
//       // assignment without aggregation would look like:
//       // d = src[ (i + 3) % 6 ]
//       agg.copy(d, src[ (i + 3) % 6 ]);
//     }
//     writeln();
//     writeln("src: ", src);
//     writeln("dst: ", dst);
// }

/*******************************
 * Performance and Diagnostics *
 *******************************/

/* Aggregation Reducing Communication */

// use BlockDist, CommDiagnostics, Time, CopyAggregation;
// // communication comparison betweeen using aggregation and straight writing
// // compile with --no-cache-remote

// config const size = 10**6;
// config const compareBulkTransfer = false;
// const dom = blockDist.createDomain({0..<size});

// // named src because this will be the source we are copying from
// var src: [dom] int = dom;

// // named dst because this will be the destination we are copying to
// var dst: [dom] int;

// resetCommDiagnostics();
// startCommDiagnostics();
// var t1 = Time.timeSinceEpoch().totalSeconds();

// forall (s, i) in zip(src, 0..) {
//   dst[ (i + (size / 2):int ) % size ] = s;
// }

// var t2 = Time.timeSinceEpoch().totalSeconds() - t1;
// stopCommDiagnostics();
// writeln("copy without aggregation time = ", t2);
// writeln("communication without aggregation: ");
// printCommDiagnosticsTable();

// resetCommDiagnostics();
// startCommDiagnostics();
// t1 = Time.timeSinceEpoch().totalSeconds();

// forall (s, i) in zip(src, 0..) with (var agg = new DstAggregator(int)) {
//   agg.copy(dst[ (i + (size / 2):int ) % size ], s);
// }

// t2 = Time.timeSinceEpoch().totalSeconds() - t1;
// stopCommDiagnostics();
// writeln();
// writeln("copy with aggregation time = ", t2);
// writeln("communication using aggregation: ");
// printCommDiagnosticsTable();

// if compareBulkTransfer {
//   resetCommDiagnostics();
//   startCommDiagnostics();
//   var t3 = Time.timeSinceEpoch().totalSeconds();
  
//   // using aggregation is not actually needed
//   // since we are copying a contiguous block 
//   dst[0..<(size / 2)] = src[(size / 2)..<size];
//   dst[(size / 2)..<size] = src[0..<(size / 2)];
  
//   var t4 = Time.timeSinceEpoch().totalSeconds() - t3;
//   stopCommDiagnostics();
//   writeln();
//   writeln("copy with aggregation time = ", t4);
//   writeln("communication using aggregation: ");
//   printCommDiagnosticsTable();
// }
