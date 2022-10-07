# Intro to Chapel
This guide will familiarize new developers with Chapel concepts regularly used in Arkouda.

<a id="toc"></a>
## Table of Contents
1. [Compiling and Running](#compile)
2. [Ranges and Domains](#ranges_and_domains)
   1. [Ranges](#ranges)
      - [Example 1: Range `5..10`](#ex1)
      - [Example 2: Range `5..#10`](#ex2)
   2. [Domains](#domains)
3. [Initial Factorial](#init_fact)
   1. [Example 3: Initial Factorial](#ex3)
   2. [Example 4: Parallel Factorial Attempt](#ex4)
4. [`reduce` and `scan`](#reduce_and_scan)
   1. [Example 5: `reduce` and `scan`](#ex5)
5. [`forall` Loops](#forall)
   1. [Example 6: Factorial with must-parallel `forall` and Reduction](#ex6)
   2. [Example 7: may-parallel `forall`](#ex7)
   3. [Example 8: `forall` Expressions](#ex8)
   4. [Example 9: Factorial with may-parallel `forall` Expression and Reduction](#ex9)
   5. [Try It Yourself 1: Perfect Squares <=25](#TIY1)
6. [Ternary](#ternary)
   1. [Example 10: Absolute Value Ternary](#ex10)
   2. [Example 11: Ternary and `forall` Expression](#ex11)
   3. [Try It Yourself 2: Array Absolute Value](#TIY2)
7. [Introspection](#introspection)
   1. [Example 12: Introspection](#ex12)
8. [Promotion](#promotion)
   1. [Example 13: Promotion](#ex13)
9. [Zippered Iteration and Aggregation](#zip_and_agg)
   1. [Zippered Iteration](#zippered_iteration)
      1. [Example 14: Simple Zippered Iteration](#ex14)
      2. [Example 15: Zippered Iteration in Arkouda](#ex15)
   2. [Aggregation](#aggregation)
      1. [Example 16: Aggregation in Arkouda](#ex16)
      2. [Try It Yourself 3: `title`](#TIY3)
10. [Filtering](#filter)
    1. [Example 17: Filtering](#ex17)
    2. [Try It Yourself 4: Sum Odd Perfect Squares <=25](#TIY4)
11. [Boolean Compression and Expansion Indexing](#bool_expand_and_compress)
    1. [Example 18: Boolean Compression Indexing](#ex18)
    2. [Example 19: Boolean Expansion Indexing](#ex19)
    3. [Try It Yourself 5: Array Even Replace](#TIY5)

<a id="compile"></a>
## Compiling and Running
If you haven't already installed Chapel, be sure to follow the instructions in [INSTALL.md](https://github.com/Bears-R-Us/arkouda/blob/master/INSTALL.md).

For all examples below, the source code is located in `Tutorial.chpl`.
After navigating to the directory containing that file, the terminal command to compile the source code into an executable (named `tutorial`) is:
```console
chpl Tutorial.chpl -o tutorial
```
The command to run the executable is:
```console
./tutorial
```
To follow along, uncomment the relevant example in `Tutorial.chpl`, compile, run, and verify the output matches the guide!

<a id="ranges_and_domains"></a>
## Ranges and Domains
This will only cover enough to provide context for the other examples (the very basics). More information can be found in the Chapel docs for
[ranges](https://chapel-lang.org/docs/primers/ranges.html) and [domains](https://chapel-lang.org/docs/primers/domains.html).

<a id="ranges"></a>
### Ranges

For this guide, the range functionality to highlight is `#`:
- `5..10` starts at `5` and ends at `10` (both inclusive).
<a id="ex1"></a>
#### Example 1: Range `5..10`
```chapel
for i in 5..10 {
  writeln(i);
}
```
```console
5
6
7
8
9
10
```
- `5..#10` starts at `5` and steps forward util we've iterated `10` elements.
<a id="ex2"></a>
#### Example 2: Range `5..#10`
```chapel
for i in 5..#10 {
  writeln(i);
}
```
```console
5
6
7
8
9
10
11
12
13
14
```
- `j..#n` is equivalent to `j..j+(n-1)`.

<a id="domains"></a>
### Domains

A domain is an index set used to specify iteration spaces and define the size and shape of arrays.
A domain’s indices may be distributed across multiple locales.
When iterating through two arrays with the same domain,
you are guaranteed that index `i` of one array will be local to index `i` of the other.

In Arkouda, pdarrays (which stand for parallel, distributed arrays), are Chapel arrays stored in
[block-distributed domains](https://chapel-lang.org/docs/primers/distributions.html#block-and-distribution-basics),
meaning that the elements are split as evenly as possible across all locales.

The syntax for declaring an array with domain `D` and type `t` is:
```Chapel 
var myArray: [D] t;
```
Some examples:
```Chapel
// ranges can be domains, resulting in a local array
var localArray: [0..10] int;

// distributed domains result in distributed arrays
var distArray: [makeDistDom(10)] int;

// two arrays can have the same domain
const D = makeDistDom(10);
var intArray: [D] int;
var realArray: [D] real;

// you can access an existing array's domain using `.domain`
var boolArray: [makeDistDom(10)] bool;
var uintArray: [boolArray.domain] uint;
```
Note: `makeDistDom` is a helper function in Arkouda and not part of Chapel.

<a id="init_fact"></a>
## Initial Factorial
Let's start our Chapel journey by writing a function to calculate the factorial of a given integer `n`.
Where factorial is

$$ n! = \prod_{i=1}^n i = 1 \cdot 2 \cdot\ldots\cdot (n-1) \cdot n$$

This will introduce the syntax for `proc` and `for` loops!

<a id="ex3"></a>
#### Example 3: Initial Factorial
Our initial implementation is similar to other languages:
```Chapel
proc factorial(n: int) {
  var fact: int = 1;
  for i in 1..n {
    fact *= i;
  }
  return fact;
}
writeln(factorial(5));
```
```console
$ chpl Tutorial.chpl -o tutorial
$ ./tutorial
120
```

Excellent! Now let's write this loop in parallel.
Our parallelism comes from the data parallel [`forall`](https://chapel-lang.org/docs/users-guide/datapar/forall.html) loop.
The `forall` loop basically handles the parallelism for you!

<a id="ex4"></a>
#### Example 4: Parallel Factorial Attempt
So what happens if we just replace  the `for` loop with a `forall`?
```Chapel
proc factorial(n: int) {
  var fact: int = 1;
  forall i in 1..n {
    fact *= i;
  }
  return fact;
}
writeln(factorial(5));
```
```console
$ chpl Tutorial.chpl -o tutorial
Tutorial.chpl:41: In function 'factorial':
Tutorial.chpl:44: error: cannot assign to const variable
Tutorial.chpl:43: note: The shadow variable 'fact' is constant due to forall intents in this loop
```
That's not what we want to see! There was an error during compilation.
The issue is different iterations of the parallel loop could be modifying the outer variable `fact` simultaneously.

Chapel handles this by giving each task its own copy of `fact` known as a [shadow variable](https://chapel-lang.org/docs/primers/forallLoops.html#task-intents-and-shadow-variables).
To combine the shadow variables and write the result back into `fact`, we'll need to learn about `reduction` operations.

<a id="reduce_and_scan"></a>
## `reduce` and `scan`
[Reductions and scans](https://chapel-lang.org/docs/language/spec/data-parallelism.html#reductions-and-scans)
cumulatively apply an operation over the elements of an array (or any iterable) in parallel.

- `op scan array`
  - scans over `array` and cumulatively applies `op` to every element
  - returns an array
  - `+ scan a` behaves like [`np.cumsum(a)`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html) in numpy
  - `scan` is an [inclusive scan](https://en.wikipedia.org/wiki/Prefix_sum#Inclusive_and_exclusive_scans)
- `op reduce array`
  - reduces the result of a scan to a single summary value
  - returns a single value
  - `+ reduce a` behaves like [`sum(a)`](https://docs.python.org/3/library/functions.html#sum) in python

<a id="ex5"></a>
#### Example 5: `reduce` and `scan`
```Chapel
var a: [0..#5] int = [1, 2, 3, 4, 5];
writeln("a = ", a, "\n");
writeln("+ scan a = ", + scan a);
writeln("+ reduce a = ", + reduce a, "\n");
writeln("* scan a = ", * scan a);
writeln("* reduce a = ", * reduce a, "\n");
writeln("min scan a = ", min scan a);
writeln("min reduce a = ", min reduce a, "\n");
writeln("max scan a = ", max scan a);
writeln("max reduce a = ", max reduce a);
```
```console
a = 1 2 3 4 5

+ scan a = 1 3 6 10 15
+ reduce a = 15

* scan a = 1 2 6 24 120
* reduce a = 120

min scan a = 1 1 1 1 1
min reduce a = 1

max scan a = 1 2 3 4 5
max reduce a = 5
```
Notice `op reduce array` is always the last element of `op scan array`.

Reductions and scans are defined for [many operations](https://chapel-lang.org/docs/language/spec/data-parallelism.html#reduction-expressions),
and you can even [write your own](https://chapel-lang.org/docs/technotes/reduceIntents.html#user-defined-reduction-example)!

<a id="forall"></a>
## `forall` Loops
<a id="ex6"></a>
#### Example 6: Factorial with must-parallel `forall` and Reduction
Back to parallel factorial, let's use `reduce` to combine the results of each task.
This is a [task intent](https://chapel-lang.org/docs/primers/forallLoops.html#task-intents-and-shadow-variables) as signified by the `with` keyword.
```Chapel
proc factorial(n: int) {
  var fact: int = 1;
  forall i in 1..n with (* reduce fact) {
    fact *= i;
  }
  return fact;
}
writeln(factorial(5));
```
```console
120
```
That's more like it!
Every task multiplies its shadow variable of `fact` by `i`, so a `* reduce` of the shadow variables gives the product of all `i`.

This is an example of a must-parallel `forall`.

There are [2 types of `forall`](https://chapel-lang.org/docs/primers/forallLoops.html#forall-loops) loops:
- must-parallel `forall`
  - Is written using the `forall` keyword i.e. `forall i in D`
  - Requires a parallel iterator
- may-parallel `forall`
  - Is written using bracket notation i.e. `[i in D]`
  - Will use a parallel iterator if present, otherwise will iterate serially

<a id="ex7"></a>
#### Example 7: may-parallel `forall`
Let's look at an example of a may-parallel `forall`:
```Chapel
[i in 0..10] {
  writeln(i);
}
```
```console
8
10
4
0
9
2
6
3
1
7
5
```
As we can see this loop is not executing serially.
This is because core Chapel types like ranges, domains, and arrays support parallel iterators, so they will be invoked by the may-parallel form.
Your output will likely be in a different order than the above and will differ between runs.

<a id="ex8"></a>
#### Example 8: `forall` Expressions
`forall`s can also be used in expressions, for example:
```Chapel
// must-parallel forall expression
var tens = forall i in 1..10 do i*10;
writeln(tens);
// may-parallel forall expression
var negativeTens = [i in tens] -i;
writeln(negativeTens);
```
```console
10 20 30 40 50 60 70 80 90 100
-10 -20 -30 -40 -50 -60 -70 -80 -90 -100
```
<a id="ex9"></a>
#### Example 9: Factorial with may-parallel `forall` Expression and Reduction
Applying a may-parallel `forall` expression and a reduction, our factorial function becomes:
```Chapel
proc factorial(n: int) {
  return * reduce [i in 1..n] i;
}
writeln(factorial(5));
```
```console
120
```

For a specified `n`, we  can do this in one line!
```Chapel
writeln(* reduce [i in 1..#5] i);
```

<a id="TIY1"></a>
#### Try It Yourself 1: Perfect Squares <=25
Problem:
Compute and print out all perfect squares less than or equal to `25`

Bonus points if you can do it in one line using a `forall` expression!
<details>
  <summary>Potential Solution</summary>

```Chapel
writeln([i in 0..5] i**2);
```
</details>
Expected Output:

```console
0 1 4 9 16 25
```

<a id="ternary"></a>
## Ternary
A ternary statement is where a variable can have one of two possible values depending on whether a condition is True or False.

The syntax for a ternary statement is:
```chapel
var x = if cond then val1 else val2;
```
This is equivalent to an `if/else`:
```chapel
var x: t;
if cond {
  x = val1;
}
else {
  x = val2;
}
```
<a id="ex10"></a>
#### Example 10: Absolute Value Ternary
Let's use a ternary to create an absolute value function:
```chapel
proc absoluteVal(n:int) {
  return if n >= 0 then n else -n;
}
writeln(absoluteVal(-15));
writeln(absoluteVal(7));
```
```console
15
7
```
<a id="ex11"></a>
#### Example 11: Ternary and `forall` Expression
We can combine ternary with other tools such as forall expressions.
```chapel
writeln([i in 0..#10] if i%2 == 0 then i+10 else -100);
```
```console
10 -100 12 -100 14 -100 16 -100 18 -100
```

<a id="TIY2"></a>
#### Try It Yourself 2: Array Absolute Value
Problem:
Write a `proc` using a ternary to take an `int array`, `A`, 
and return an array where index `i` is the absolute value of `A[i]`

Call: `arrayAbsVal([-3, 7, 0, -4, 12]);`

<details>
  <summary>Potential Solution</summary>

```Chapel
proc arrayAbsVal(A) {
  return [a in A] if a >= 0 then a else -a;
}
```
</details>
Expected Output:

```console
3 7 0 4 12
```

<a id="introspection"></a>
## Introspection
Introspection is determining properties of a function argument at runtime.
This is often used to determine the type and/or domain of a function argument.
Introspection can avoid duplicating a `proc` for multiple types when none of the logic has changed.
Using introspection will result in a [generic](https://chapel-lang.org/docs/language/spec/generics.html#generics) function.

The syntax for this is:
```Chapel
proc foo(arr1: [?D] ?t, value: t, arr2: [?D2] ?t2) {
  // using `D` and `t` in this proc will refer to `arr1`'s domain and type respectively
  // since value is declared to have type `t`, it must be passed a value that is compatible with `arr1`'s element type
  // `D2` and `t2` refer to the domain and type of `arr2`
}
```
<a id="ex12"></a>
#### Example 12: Introspection
If we adapt our absolute value example to use introspection,
we can use the same `proc` for `real array`s and `int array`s.
```Chapel
proc absoluteVal(a: [?D] ?t): [D] t {
  return [i in D] if a[i] >= 0 then a[i] else -a[i];
}

var r: [0..#5] real = [-3.14, 7:real, 0.0, INFINITY, -INFINITY];
writeln(absoluteVal(r));

var i: [0..#5] int = [-3, 7, 0, -4, 12];
writeln(absoluteVal(i));
```
```console
3.14 7.0 0.0 inf inf
3 7 0 4 12
```

<a id="promotion"></a>
## Promotion
[Promotion](https://chapel-lang.org/docs/users-guide/datapar/promotion.html) is a way to obtain data parallelism implicitly.

A function or operation that operates on a type can automatically work on an array of that type
by essentially applying the function to every element of the array.

<a id="ex13"></a>
#### Example 13: Promotion
Returning to the factorial example, our `proc` can operate on an `int array` even though it's only defined to accept an `int`.
```chapel
proc factorial(n: int) {
  return * reduce [i in 1..#n] i;
}
writeln(factorial(5));
writeln(factorial([1, 2, 3, 4, 5]));
```
```console
120
1 2 6 24 120
```

Promotion is equivalent to a `forall` loop. For this example the equivalent loop would be:
```chapel
[x in [1, 2, 3, 4, 5]] factorial(x);
```

<a id="zip_and_agg"></a>
## Zippered Iteration and Aggregation

<a id="zippered_iteration"></a>
### Zippered Iteration
[Zippered Iteration](https://chapel-lang.org/docs/users-guide/base/zip.html#zippered-iteration) is simultaneously
iterating over multiple iterables (usually arrays and ranges) with compatible shape and size.
For arrays, this means that their domains have the same number of elements in each dimension

The syntax:
```chapel
var A1, A2, A3, ..., An: [D] int;
forall (v1, v2, v3, ..., vn) in zip(A1, A2, A3, ..., An) {
  // for loop iteration `j`, `vi` will refer to the `j`th element of `Ai`
}
```
Since all `Ai` have compatible domains, we know all `vi` will be local to one another on any given loop iteration.

<a id="ex14"></a>
#### Example 14: Simple Zippered Iteration
Let's start with a simple example involving zippered iteration:
```Chapel
var A: [1..5] real;
for (a, i, j) in zip(A, 1.., [3, 0, 1, 2, 4]) {
  a = i**j;
}
writeln(A);
```
```console
1.0 1.0 3.0 16.0 625.0
```
Notice we have an [unbounded range](https://chapel-lang.org/docs/primers/ranges.html#variations-on-basic-ranges), `1..`,
so the end bound is determined by the size of the other iterables.
Since in this case the other iterables are length 5, `1..` is equivalent to `1..5`.

<a id="ex15"></a>
#### Example 15: Zippered Iteration in Arkouda
Now let's look at an example based on [`getLengths`](https://github.com/Bears-R-Us/arkouda/blob/b86adeb843275b0b86553534ede632acef4d15e2/src/SegmentedString.chpl#L388-L406) in `SegmentedString.chpl`.

First we need some context. An Arkouda `Strings` (aka SegString) is an array of variable length strings.
```python
>>> s = ak.array([f"{'s'*(3-i)}{i}" for i in range(3)])
>>> s
array(['sss0', 'ss1', 's2'])
>>> type(s)
arkouda.strings.Strings
```

On the server, a SegString is made up of two distributed array components:
- values: `uint(8)`
  - the flattened array of bytes in all the strings (null byte delimited)
- offsets: `int`
  - the start index of each individual string in the `values`

For the sake of simplicity, we'll treat values as a string array.
So for our `s` above, these components look something like:
```
values = ['s', 's', 's', '0', '\x00', 's', 's', '1', '\x00', 's', '2', '\x00']
offsets = [0, 5, 9]
```
For `getLengths`, we want to calculate the length of each individual string including the null terminator.

```chapel
const values: [0..#12] string = ['s', 's', 's', '0', '\x00', 's', 's', '1', '\x00', 's', '2', '\x00'],
      offsets = [0, 5, 9],
      size = offsets.size;  // size refers to the number of stings in the Segstring, this is always equivalent to offsets.size

/* Return lengths of all strings, including null terminator. */
proc getLengths() {
  // initialize lengths with the same domain as offsets
  var lengths: [offsets.domain] int;
  if size == 0 {
    // if the segstring is empty, the lengths are also empty
    return lengths;
  }
  // save off the last index of offsets
  const high = offsets.domain.high;
  forall (i, o, l) in zip(offsets.domain, offsets, lengths) {
    if i == high {
      // the last string
      // len = values.size - start position of this string
      l = values.size - o;
    }
    else {
      // not the last string
      // len = start position of next string - start position of this string
      l = offsets[i+1] - o;
    }
  }
  return lengths;
}

writeln(getLengths());
```
```console
5 4 3
```
Nice! We'll use `getLengths`, `values`, and `offsets` in the next section as well.

<a id="aggregation"></a>
### Aggregation

[Aggregation](https://chapel-lang.org/docs/modules/packages/CopyAggregation.html#copyaggregation) is used to copy
local values into a remote array (`DstAggregator`) or copy values from a remote array into local variables (`SrcAggregator`).
This provides a significant speed up when doing batch assignments.

Syntax:
```chapel
// without aggregation
forall (i, v) in zip(inds, vals) {
  remoteArr[i] = v;
}

// with aggregation
use CopyAggregation;
forall (i, v) in zip(inds, vals) with (var agg = new DstAggregator(int)) {
  agg.copy(remoteArr[i], v);
}
```
The `with` keyword here declares `agg` as a [Task-Private variable](https://chapel-lang.org/docs/language/spec/data-parallelism.html#task-private-variables),
meaning each task will get its own shadow variable.

It's important to note aggregation will only work if at least one side is local.
Both sides being remote (remote-to-remote aggregation) is not yet supported. 

<a id="ex16"></a>
#### Example 16: Aggregation in Arkouda
Let's do an example based on [`upper`](https://github.com/Bears-R-Us/arkouda/blob/b86adeb843275b0b86553534ede632acef4d15e2/src/SegmentedString.chpl#L427-L444) in `SegmentedString.chpl`.

We will use `getLengths` and the `SegString s` from [example 15](#ex15):
```chapel
use CopyAggregation;

/*
  Given a SegString, return a new SegString with all lowercase characters from the original replaced with their uppercase equivalent
  :returns: Strings – Substrings with lowercase characters replaced with uppercase equivalent
*/
proc upper() {
  var upperVals: [values.domain] string;
  const lengths = getLengths();
  forall (off, len) in zip(offsets, lengths) with (var valAgg = new DstAggregator(string)) {
    var i = 0;
    for char in ''.join(values[off..#len]).toUpper() {
      valAgg.copy(upperVals[off+i], char);
      i += 1;
    }
  }
  return (offsets, upperVals);
}

writeln("Old Vals:", values);
var (newOffs, newVals) = upper();
writeln("New Vals:", newVals);
```
```console
Old Vals:s s s 0  s s 1  s 2
New Vals:S S S 0  S S 1  S 2
```
This function uses our previous `getLengths` function as well as the Chapel builtins
[`toUpper`](https://chapel-lang.org/docs/language/spec/strings.html#String.string.toUpper)
and [`join`](https://chapel-lang.org/docs/language/spec/strings.html#String.string.join)!


<a id="TIY3"></a>
<a id="TIY3"></a>
#### Try It Yourself 3: `title`
Problem:
Use Aggregation to transform the SegString from [example 15](#ex15) into title case.
Be sure to use the Chapel builtin [`toTitle`](https://chapel-lang.org/docs/language/spec/strings.html#String.string.toTitle)
and `getLengths` from example 15.
<details>
  <summary>Potential Solution</summary>

```Chapel
use CopyAggregation;

proc title() {
  var titleVals: [values.domain] string;
  const lengths = getLengths();
  forall (off, len) in zip(offsets, lengths) with (var valAgg = new DstAggregator(string)) {
    var i = 0;
    for char in ''.join(values[off..#len]).toTitle() {
      valAgg.copy(titleVals[off+i], char);
      i += 1;
    }
  }
  return titleVals;
}
writeln(title());
```
</details>
Expected Output:

```console
S s s 0  S s 1  S 2
```

<a id="filter"></a>
## Filtering
<a id="ex17"></a>
#### Example 17: Filtering
We can filter an iterator to only operate on values matching a certain condition:
```chapel
writeln([i in 0..#10] if i%2 == 0 then -i);
```
```console
0 -2 -4 -6 -8
```

<a id="TIY4"></a>
#### Try It Yourself 4: Sum Odd Perfect Squares <=25
Problem:
Use filtering and reduce to sum all odd perfect squares less than or equal to `25`
<details>
  <summary>Potential Solution</summary>

```Chapel
writeln(+ reduce [i in 0..5] if i%2 != 0 then i**2);
```
</details>
Expected Output:

```console
35
```

<a id="bool_expand_and_compress"></a>
## Boolean Compression and Expansion Indexing
[Boolean compression and expansion indexing](https://github.com/Bears-R-Us/ArkoudaWeeklyCall/blob/main/Indexing/ArkoudaBooleanIndexing.pdf)
is an application of `reduce scan op`, `forall`, and filtering. It was covered in an [Arkouda Weekly Call](https://github.com/Bears-R-Us/ArkoudaWeeklyCall).

<a id="ex18"></a>
#### Example 18: Boolean Compression Indexing
Compression indexing is reducing an array to only the values meeting a certain condition.
This is a common operation in numpy.

Compression indexing in `numpy`:
```python
>>> X = np.array([1, 2, 5, 5, 1, 5, 2, 5, 3, 1])
>>> Y = X[X==5]
>>> Y
array([5, 5, 5, 5])
```
We can accomplish the same result in Chapel using tools from this guide! Specifically `reduce scan op`, `forall`, and filtering.
```chapel
var X = [1, 2, 5, 5, 1, 5, 2, 5, 3, 1];
writeln("X = ", X, "\n");

// we begin by creating a boolean array, `truth`, indicating where the condition is met
var truth = X == 5;
writeln("truth = ", truth);

// we use `truth` to create the indices, `iv`, into the compressed array
// `+ scan truth - truth` is essentially creating an exclusive scan
// note: `iv[truth] = [0, 1, 2, 3]`
var iv = + scan truth - truth;
writeln("iv = ", iv);
writeln("iv[truth] = ", [(t, v) in zip(truth, iv)] if t then v, "\n");

// we then create the return array `Y`
// it contains all the elements where the condition is met
// so its size is the number of `True`s i.e. `+ reduce truth`
var Y: [0..#(+ reduce truth)] int;
writeln("+ reduce truth = ", + reduce truth);
writeln("0..#(+ reduce truth) = ", 0..#(+ reduce truth), "\n");

// now that we have the setup, it's time for the actual indexing
// we do a may-parallel `forall` to iterate over the indices of `X`
// we filter on `truth[i]`, so we only act if the condition is met
// we use the compressed indices `iv[i]` to write into `Y`
// while using the original indices `i` to get the correct value from `X`
[i in X.domain] if truth[i] {Y[iv[i]] = X[i];}

// note we could do the same thing with zippered iteration
// since `truth`, `X`, and `iv` have the same domain
// [(t, x, v) in zip(truth, X, iv)] if t {Y[v] = x;}

writeln("Y = ", Y);
```
```console
X = 1 2 5 5 1 5 2 5 3 1

truth = false false true true false true false true false false
iv = 0 0 0 1 2 2 3 3 4 4
iv[truth] = 0 1 2 3

+ reduce truth = 4
0..#(+ reduce truth) = 0..3

Y = 5 5 5 5
```
Awesome! We used most of the information from this guide to reproduce useful numpy functionality!
With the added benefit that our implementation is parallel and works on distributed data!
Now let's tackle the very similar challenge of expansion indexing.

<a id="ex19"></a>
#### Example 19: Boolean Expansion Indexing
Expansion indexing is writing a smaller array into a larger array at only the values meeting a certain condition.
This is a common operation in numpy.

Expansion indexing in `numpy`:
```python
>>> X = np.array([1, 2, 5, 5, 1, 5, 2, 5, 3, 1])
>>> Y = np.array([-9, -8, -7, -6])
>>> X[X==5] = Y
>>> X
array([ 1,  2, -9, -8,  1, -7,  2, -6,  3,  1])
```
We can accomplish the same result in Chapel using tools from this guide! Specifically `reduce scan op`, `forall`, and filtering.
```chapel
var X = [1, 2, 5, 5, 1, 5, 2, 5, 3, 1];
var Y = [-9, -8, -7, -6];
writeln("X = ", X);
writeln("Y = ", Y, "\n");

// we begin by creating a boolean array, `truth`, indicating where the condition is met
var truth = X == 5;
writeln("truth = ", truth);

// we use `truth` to create the indices, `iv`, into the compressed array
// `+ scan truth - truth` is essentially creating an exclusive scan
// note: `iv[truth] = [0, 1, 2, 3]`
var iv = + scan truth - truth;
writeln("iv = ", iv);
writeln("iv[truth] = ", [(t, v) in zip(truth, iv)] if t then v, "\n");

// now that we have the setup, it's time for the actual indexing
// this is equivalent to compression indexing with the assignment swapped
// we do a may-parallel `forall` to iterate over the indices of `X`
// we filter on `truth[i]`, so we only act if the condition is met
// we use the original indices `i` to write into `X`
// while using the compressed indices `iv[i]` to get the correct value from `Y`
[i in X.domain] if truth[i] {X[i] = Y[iv[i]];}

// note we could do the same thing with zippered iteration
// since `truth`, `X`, and `iv` have the same domain
// [(t, x, v) in zip(truth, X, iv)] if t {x = Y[v];}

writeln("X = ", X);
```
```console
X = 1 2 5 5 1 5 2 5 3 1
Y = -9 -8 -7 -6

truth = false false true true false true false true false false
iv = 0 0 0 1 2 2 3 3 4 4
iv[truth] = 0 1 2 3

X = 1 2 -9 -8 1 -7 2 -6 3 1
```
Great! Now let's use this to tackle our final Try It Yourself!

<a id="TIY5"></a>
#### Try It Yourself 5: Array Even Replace
Problem:
Create a `proc` which given two int arrays `A` and `B` with different domains
will return `A` but with the even values replaced with the values of `B`

You should aim to use as many of the concepts from the guide as possible:
  - boolean expansion indexing
  - may-parallel `forall`
  - filtering
  - scan
  - introspection
  - zippered iteration

Call:
  - `arrayEvenReplace([8, 9, 7, 2, 4, 3], [17, 19, 21]);`
  - `arrayEvenReplace([4, 4, 7, 4, 4, 4], [9, 9, 9, 9, 9]);`

<details>
  <summary>Potential Solution</summary>

```Chapel
proc arrayEvenReplace(A: [?D] int, B: [?D2] int) {
  const truth = A % 2 == 0;
  const iv = + scan truth - truth;
  [(t, a, v) in zip(truth, A, iv)] if t {a = B[v];}
  return A;
}
```
</details>
Expected Output:

```console
17 9 7 19 21 3
9 9 7 9 9 9
```
Congrats!! You've completed the Chapel tutorial!

## Want to learn more?
Most of this functionality (and much more) is covered in depth in the Chapel documentation! Check out their [User's Guide](https://chapel-lang.org/docs/users-guide/index.html)
and [primers](https://chapel-lang.org/docs/primers/index.html).
