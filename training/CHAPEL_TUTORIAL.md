# Intro to Chapel
This guide aims to:
* Introduce new Arkouda developers to some Chapel concepts commonly used in Arkouda
* Serve as a reference when encountering unfamiliar Chapel code
  * Hopefully turn "I have no clue what this is doing" into "oh I think I saw this in the intro! Let me go reread that"
* Provide links to the relevant [Chapel docs](https://chapel-lang.org/docs/) for further reading

Ideally this material will be accessible to developers with no parallel or distributed programming experience.
If you see something that should be updated for clarity or correctness,
please [add an issue](https://github.com/Bears-R-Us/arkouda/issues) to let us know! 

There three main sections of the Chapel docs linked in this document:
the [primers](https://chapel-lang.org/docs/primers/index.html), the
[user guide](https://chapel-lang.org/docs/users-guide/index.html), and the
[language specification](https://chapel-lang.org/docs/language/spec/index.html).
I link to the primers or user guide when possible since they are more beginner-friendly.
If you prefer a more precise and exhaustive treatment of the material,
I recommend looking into the language spec!

<a id="toc"></a>
## Table of Contents
* [Compiling and Running](#compile)
* [Ranges and Domains](#ranges_and_domains)
  * [Ranges](#ranges)
  * [Domains](#domains)
* [Procedures](#procs)
  * [Serial Factorial](#serial_factorial)
  * [Parallel Factorial Attempt](#parallel_factorial_attempt)
* [`scan` and `reduce` operations](#reduce_and_scan)
* [`forall` Loops](#forall)
  * [Factorial with must-parallel `forall` and Reduction](#must_parallel)
  * [May-parallel `forall`](#may_parallel)
  * [`forall` Expressions](#forall_expr)
  * [Factorial with may-parallel `forall` Expression and Reduction](#may_parallel_expr_reduce)
  * [Try It Yourself: Perfect Squares <=25](#TIY_perf_squares)
* [Zippered Iteration](#zippered_iteration)
* [Ternary](#ternary)
  * [Absolute Value Ternary](#abs_val_ter)
  * [Ternary and `forall` Expression](#ter_forall_expr)
  * [Try It Yourself: Array Absolute Value](#TIY_arr_abs_val)
* [Generics and Introspection](#generics_introspection)
  * [Generics](#generics)
  * [Introspection](#introspection)
  * [Try It Yourself: Array Absolute Value with Introspection](#TIY_abs_val_introspec)
* [Promotion](#promotion)
  * [Try It Yourself: Absolute Value with Promotion](#TIY_abs_val_promotion)
* [Filtering](#filter)
  * [Try It Yourself: Sum Odd Perfect Squares <=25](#TIY_sum_odd_sqaures)
* [Boolean Compression and Expansion Indexing](#bool_expand_and_compress)
  * [Boolean Compression Indexing](#comp_ind)
  * [Boolean Expansion Indexing](#expan_ind)
  * [Try It Yourself: Array Even Replace](#TIY_arr_even_repl)
* [Locales and `coforall` loops](#loc_and_coforall)
  * [Locales](#locale)
  * [`coforall` Loops](#coforall)
  * [Enabling multiple locales](#compile-multiloc)
  * [Looping Locales with a `coforall`](#locale_looping)
  * [Implicit distributed computation with `forall`](#forall_distribution)
* [Aggregation](#aggregation)
* [Performance and Diagnostics](#perf)
  * [Variable Declarations](#var_dec)
  * [Diagnostics](#diagnostics)
  * [Aggregation reducing Communication](#agg_comm)
  * [Common Pitfalls](#pitfalls)

<a id="compile"></a>
## Compiling and Running
If you haven't already installed Chapel and Arkouda, be sure to follow the
[installation instructions](https://bears-r-us.github.io/arkouda/setup/install_menu.html).

For all examples below, the source code is located in `tutorial.chpl`.
After navigating to the directory containing that file, the terminal command to compile
the source code into an executable (named `tutorial`) is:
```console
chpl tutorial.chpl
```

The command to run the executable is:
```console
./tutorial
```

Some later examples will provide additional instructions to add flags to these commands.

To follow along, uncomment the relevant code in `tutorial.chpl`. Then compile, run, and verify the output matches the guide!

<a id="ranges_and_domains"></a>
## Ranges and Domains
This will only cover enough to provide context for the other examples (the very basics).
More information can be found in the Chapel docs for
[ranges](https://chapel-lang.org/docs/primers/ranges.html) and
[domains](https://chapel-lang.org/docs/primers/domains.html).

<a id="ranges"></a>
### Ranges

For this guide, the range functionality to highlight is `<` and `#`:
#### Range `5..10`
`5..10` starts at `5` and ends at `10` (both inclusive).

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

#### Range `5..<10`
`5..<10` starts at `5` (inclusive) and ends at `10` (exclusive).

```chapel
for i in 5..<10 {
  writeln(i);
}
```
```console
5
6
7
8
9
```
- `j..<n` is equivalent to `j..(n-1)`.

#### Range `5..#10`
`5..#10` starts at `5` and counts forward `10` elements.

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

A domain represents a set of indices.
Domains can be used to specify the indices of an array, or the indices for a loop.
For arrays, our most common use case, the domain specifies:
- the size and shape of the array
- if and how the elements of the array are distributed across the locales

We cover locales in more detail in a later section, but you can think of a locale as a single computer 
in a cluster of computers all working together to solve a problem.

A distributed domain contains information about which locale an array element is stored on.
Knowing where the data lives is necessary to take full advantage of distributed resources.
A natural way to split up work is to have each locale operate on the elements of the distributed array that reside on it.
This often ends up being the most efficient approach as well.

When iterating over two arrays with the same domain, you are guaranteed that index `i` of one
array will be co-located with index `i` of the other (i.e. they will be stored on the same locale).

In Arkouda, pdarrays (which stand for parallel, distributed arrays), are Chapel arrays with
[block-distributed domains](https://chapel-lang.org/docs/primers/distributions.html#the-block-distribution).
This means the elements are split as evenly as possible across all locales.
We'll see this in action in a later section!

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

<a id="procs"></a>
## Procedures
Let's start our Chapel journey by writing a function to calculate the factorial of a given integer `n`.
Where factorial is

$$ n! = \prod_{i=1}^n i = 1 \cdot 2 \cdot\ldots\cdot (n-1) \cdot n$$

Functions in Chapel are called [procedures](https://chapel-lang.org/docs/primers/procedures.html),
and they use the `proc` keyword.

<a id="serial_factorial"></a>
#### Serial Factorial
Our initial implementation is similar to other languages. We iterate the values from 1 to n and multiply
them all together. We use a [`for` loop](https://chapel-lang.org/docs/users-guide/base/forloops.html)
which behaves like `for` loops in other languages. A `for` loop is a serial
(non-parallel) loop, so it executes its loop iterations in order and a loop iteration doesn't start until the iteration
before it finishes. 

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
120
```

Excellent! We just wrote our first bit of Chapel code! Now can we make it parallel?

In a parallel loop, a loop iteration doesn't wait for the iterations before it to complete before it starts. So 
if any of our logic relies on a previous loop iteration to finish first, using a parallel loop would sometimes
give incorrect results.

For our example, the order that the iterations complete doesn't affect the final answer (i.e. multiplying
by `2` before `1` doesn't change the resulting product).

Now that we've identified `factorial` as a good candidate, let's try to use a parallel loop!

<a id="parallel_factorial_attempt"></a>
#### Parallel Factorial Attempt
The parallel loop we're going to use is a
[`forall`](https://chapel-lang.org/docs/users-guide/datapar/forall.html) loop. We'll cover `forall` in more detail
later, but for now just take it to mean that multiple iterations can execute at once. 

So what happens if we just replace  the `for` loop with a `forall`? Let's try it.
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
$ chpl tutorial.chpl
tutorial.chpl:45: In function 'factorial':
tutorial.chpl:48: error: cannot assign to const variable
tutorial.chpl:47: note: The shadow variable 'fact' is constant due to task intents in this loop
```
That's not what we want to see! There was an error during compilation.

To parallelize this loop, Chapel breaks it up into a number of
[`tasks`](https://chapel-lang.org/docs/users-guide/taskpar/taskParallelismOverview.html)
that can be performed simultaneously. For a `forall` loop, Chapel determines the number of tasks for us
automagically using info like what we're iterating over and what system resources are available.
It's likely that a single task will be responsible for multiple loop iterations.  

If that's a lot to take in the key takeaway is there are several tasks that are executing at the same time.

So there are multiple tasks executing at once and only one logical `fact` variable.
This could lead to different tasks trying to modify `fact` simultaneously. 
This could cause problems because it's possible for one task to prevent another from
updating `fact` correctly, resulting in incorrect answers and inconsistent behavior between runs.

Okay our tasks are like kids that don't share well, so how do we avoid these problems?
One idea is to give each task its own copy of `fact` and at the end figure out how to
combine them into a final answer.

As turns out, Chapel does the first half of that for us!
It gives each task its own private copy of `fact` called a
[task private variable](https://chapel-lang.org/docs/primers/forallLoops.html#task-private-variables).
These are a special kind of task private variables since they have the same name as the `fact` from
the outer scope (aka [shadowing `fact`](https://en.wikipedia.org/wiki/Variable_shadowing)).
Due to this, they're called 
[shadow variables](https://chapel-lang.org/docs/primers/forallLoops.html#task-intents-and-shadow-variables).

So right now each task has its own private copy of `fact`, but we still need to combine all these
shadow variables to get our final answer. To do that, we'll need to learn about `reduction` operations.

<a id="reduce_and_scan"></a>
## `reduce` and `scan`
[Reductions and scans](https://chapel-lang.org/docs/language/spec/data-parallelism.html#reductions-and-scans)
apply an operation over the elements of an array (or any iterable) in parallel.
- scan operations
  - have form `op scan array`
  - scans over `array` and cumulatively applies `op` to every element
  - returns an array
  - for those familiar with numpy, `+ scan a` behaves like [`np.cumsum(a)`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
  - `scan` is an [inclusive scan](https://en.wikipedia.org/wiki/Prefix_sum#Inclusive_and_exclusive_scans)
- reduce operations
  - have form `op reduce array`
  - reduces the result of a scan to a single summary value
  - returns a single value
  - for those familiar with python, `+ reduce a` behaves like [`sum(a)`](https://docs.python.org/3/library/functions.html#sum)

#### `reduce` and `scan` Example
```Chapel
var a: [0..<5] int = [1, 2, 3, 4, 5];
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

Reductions and scans are defined for
[many operations](https://chapel-lang.org/docs/language/spec/data-parallelism.html#reduction-expressions).

<a id="forall"></a>
## `forall` Loops
<a id="must_parallel"></a>
#### Factorial with must-parallel `forall` and Reduction
Back to parallel factorial! We want to combine all our shadow variables. Since our goal is the product of all
the values, it sounds like we might be able to use a `* reduce` to combine the results of each task.

To do this we use a [task intent](https://chapel-lang.org/docs/primers/forallLoops.html#task-intents-and-shadow-variables)
to tell the tasks how we want them to handle their task private variables (in this case to reduce them to a single answer).
We use the `with` keyword to signal we are adding a task intent for `fact`.

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
Yay! That's more like it!
Every task multiplies its shadow variable of `fact` by all the values of `i` it owns,
so a `* reduce` of
all the shadow variables gives the product of all `i`. This reduction is then combined
into `fact` from the outer scope. 

Awesome, we've successfully used a `forall` loop to calculate factorial in parallel!

<a id="may_parallel"></a>
#### may-parallel `forall`
There's another type of `forall` loop that uses a different syntax. The primary difference between the two is
whether they are required to execute in parallel or not. For something to be executed in parallel, it needs a
[parallel iterator](https://chapel-lang.org/docs/primers/parIters.html#primers-pariters).
Core Chapel types like ranges, domains, and arrays all support parallel iterators.

- must-parallel `forall`
  - is written using the `forall` keyword i.e. `forall i in D`
  - requires a parallel iterator
- may-parallel `forall`
  - is written using bracket notation i.e. `[i in D]`
  - will use a parallel iterator if present, otherwise will iterate serially

Up until now, we've only used the `must-parallel` form, let's look at an example of a may-parallel `forall`:
```Chapel
[i in 0..<10] {
  writeln(i);
}
```
```console
8
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
As we can see this loop is not executing serially because a `range` has a parallel iterator, which
will be invoked by the may-parallel form.
Your output will likely be in a different order than the above and will differ between runs.

<a id="forall_expr"></a>
#### `forall` Expressions
`forall`s can also be used in expressions, for example:
```Chapel
// must-parallel forall expression
var tens = forall i in 1..10 do i*10;
writeln(tens);
// may-parallel forall expression
var negativeTens = [t in tens] -t;
writeln(negativeTens);
```
```console
10 20 30 40 50 60 70 80 90 100
-10 -20 -30 -40 -50 -60 -70 -80 -90 -100
```
Note: by doing `[t in tens] -t;`, we also just showed we can iterate over the values of an array directly using a `forall`.
This is equivalent to looping over the domain and indexing into the array `[i in tens.domain] -tens[i];`

<a id="may_parallel_expr_reduce"></a>
#### Factorial with may-parallel `forall` Expression and Reduction
Now let's add some of these new features to our factorial function! 
Applying a may-parallel `forall` expression and a reduction, our function becomes:
```Chapel
proc factorial(n: int) {
  return * reduce [i in 1..n] i;
}
writeln(factorial(5));
```
```console
120
```

For a specified `n`, we can even do this in one line!
```Chapel
writeln(* reduce [i in 1..5] i);
```

<a id="TIY_perf_squares"></a>
#### Try It Yourself: Perfect Squares <=25
Problem:
Compute and print out all perfect squares less than or equal to `25`

Bonus points if you can do it in one line using a `forall` expression!

Expected Output:

```console
0 1 4 9 16 25
```

<details>
  <summary>Potential Solutions</summary>

One way is
```Chapel
var arr: [0..5] int;

forall i in 0..5 {
  arr[i] = i**2;
}

writeln(arr);
```

or the one-liner!
```Chapel
writeln([i in 0..5] i**2);
```
</details>

<a id="zippered_iteration"></a>
## Zippered Iteration
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

Let's look at an example of zippered iteration:
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
Notice we have an [unbounded range](https://chapel-lang.org/docs/primers/ranges.html#variations-on-basic-ranges),
`1..`, so the end bound is determined by the size of the other iterables.
Since in this case the other iterables are length 5, `1..` is equivalent to `1..5`.

<a id="ternary"></a>
## Ternary
A ternary statement can have one of two possible values depending on whether a condition is true or false.
Because Chapel is a strongly typed language, both values must be the same type.

The syntax for a ternary statement in Chapel is:
```chapel
var x = if cond then val1 else val2;
```
This is equivalent to an `if/else`:
```chapel
var x;
if cond {
  x = val1;
}
else {
  x = val2;
}
```
<a id="abs_val_ter"></a>
#### Absolute Value Ternary
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
<a id="ter_forall_expr"></a>
#### Ternary and `forall` Expression
Now let's combine a ternary with a `forall` expression!

We're going to loop over the positive integers less than 10.
If the value is even, we'll write out the `value + 10`, and if it's odd, we'll write out `-100`.
```chapel
writeln([i in 0..<10] if i%2 == 0 then i+10 else -100);
```
```console
10 -100 12 -100 14 -100 16 -100 18 -100
```
Awesome! Now try to combine some of the topics we've covered.

<a id="TIY_arr_abs_val"></a>
#### Try It Yourself: Array Absolute Value
Problem:

Use the following function signature to write
a `proc` using a ternary which takes an `int array` 
and returns an array whose elements are the absolute values
of the corresponding input array values:
```Chapel
proc arrayAbsVal(A: [] int)
```

Call: `arrayAbsVal([-3, 7, 0, -4, 12]);`

Expected Output:

```console
3 7 0 4 12
```

<details>
  <summary>Potential Solutions</summary>

using a forall loop
```Chapel
proc arrayAbsVal(A: [] int) {
  var absArr: [A.domain] int;
  forall (v, a) in zip(absArr, A) {
    v = if a >= 0 then a else -a;
  } 
  return absArr;
}
```
or using a forall expression
```Chapel
proc arrayAbsVal(A: [] int) {
  return [a in A] if a >= 0 then a else -a;
}
```
</details>

<a id="generics_introspection"></a>
## Generics and Introspection
Let's say you want a function that takes some integer input and doubles it.
We can do this pretty easily with what we've learned so far

```Chapel
proc double(a: int) {
  return a * 2;
}
```

Great! Now say we want to be able to do the same thing for `uint` and `real` input.
We could [overload](https://chapel-lang.org/docs/primers/procedures.html#overloading-functions)
our procedure and create multiple with the same name that accept different types. 

```Chapel
proc double(a:int) {
  return a * 2;
}

proc double(a:uint) {
  return a * 2;
}

proc double(a:real) {
  return a * 2;
}

writeln(double(-100));
writeln(double(7.5));
```

```console
-200
15.0
```
This works... but it's pretty inefficient considering the logic itself hasn't actually changed.
Plus if we ever want to update this, we would need to remember to update all these spots.

It would be nice if we could write one proc that accepts all these types.
Luckily we can! This is called a
[generic](https://chapel-lang.org/docs/language/spec/generics.html#generics) procedure.

<a id="generics"></a>
### Generics
One way to make our `double` proc generic is to leave off the type annotation. 
If we do this Chapel will allow it take input of any type.
This is super nice because we don't have to do very much!

```Chapel
proc double(a) {
  return a * 2;
}

writeln(double(-100));
writeln(double(7.5));
```

```console
-200
15.0
```

There is a problem here though... Chapel will allow it take input of ANY type. 
What if someone tries to pass in a `string`?

```Chapel
proc double(a) {
  return a * 2;
}

writeln(double(-100));
writeln(double(7.5));
writeln(double("oh no! we don't want strings!"));
```

```console
-200
15.0
oh no! we don't want strings!oh no! we don't want strings!
```

Since we're no longer adding type annotations, it's possible for unintended types to slip through.
For a small program that only you modify, this might not be an issue. But for a bigger project like Arkouda,
the chances that your proc will be used in a way you didn't intend increases.

Okay so we don't want to duplicate our function, but we'd like to do some type enforcement.
To solve this, we'll use type introspection!

<a id="introspection"></a>
### Introspection

Introspection is the process of determining properties of an object.
In Chapel, this is often used to determine the type and/or domain of a function argument.
When you see `?` preceding an identifier, it is acting as a
[query expression](https://chapel-lang.org/docs/language/spec/expressions.html#the-query-expression)
and is querying the type or value.

The syntax for this is:
```Chapel
proc foo(arr1: [?D] ?t, val: t, arr2: [?D2] ?t2) {
  // `D` and `t` are now equal to `arr1`'s domain and element type respectively
  // since val is declared to have type `t`, it must be passed a value that is compatible with `arr1`'s element type
  // `D2` and `t2` refer to the domain and element type of `arr2`
}
```

To use `?` for type enforcement of our `double` proc, we need to briefly touch on 
[`where` clauses](https://chapel-lang.org/docs/language/spec/procedures.html#where-clauses).
A `where` clause has a condition that must be satisfied for a proc to be used.

So we want to define a proc where the input's type is `int` or `uint` or `real`.
In code this is:

```Chapel
proc double(a: ?t) where t == int || t == uint || t == real {
  return a * 2;
}

writeln(double(-100));
writeln(double(7.5));
```

```console
-200
15.0
```

It works! Okay moment of truth, what happens if we try to pass in a `string`?

```Chapel
proc double(a: ?t) where t == int || t == uint || t == real {
  return a * 2;
}

writeln(double("oh no! we don't want strings!"));
```

```console
error: unresolved call 'double("oh no! we don't want strings!")'
note: this candidate did not match: double(a: ?t)
note: because where clause evaluated to false
```

Awesome! We used type querying and a where clause to create a generic
proc which only accepts the types we want.

<a id="TIY_abs_val_introspec"></a>
#### Try It Yourself: Array Absolute Value with Introspection
Problem:
Let's build upon our Array Absolute Value Try It Yourself!

We have a `proc` which takes an `int array`, `A`, and returns the index-wise absolute value.
Modify it to also accept a `real array`.

Call:
```
arrayAbsVal([-3.14, 7:real, 0.0, inf, -inf]);
arrayAbsVal([-3, 7, 0, -4, 12]);
```
Expected Output:

```console
3.14 7.0 0.0 inf inf
3 7 0 4 12
```

<details>
  <summary>Potential Solutions</summary>

using a forall loop
```Chapel
proc arrayAbsVal(A: [] ?t) where t == int || t == real {
  var absArr: [A.domain] t;
  
  forall (v, a) in zip(absArr, A) {
    v = if a >= 0 then a else -a;
  }
  return absArr;
}
```

using a forall expression
```Chapel
proc arrayAbsVal(A: [] ?t) where t == int || t == real {
  return [a in A] if a >= 0 then a else -a;
}
```
</details>

<a id="promotion"></a>
## Promotion
A function or operation that works on a single value will also work on an array of values of the same type automatically. 
 This is called
[promotion](https://chapel-lang.org/docs/users-guide/datapar/promotion.html).
It's essentially applying the function to every element of the array.

Returning to the factorial example, our `proc` is only defined to accept an `int`. But if an `int array` is passed in,
it will be automatically promoted to handle the array.
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

<a id="TIY_abs_val_promotion"></a>
#### Try It Yourself: Absolute Value with Promotion
Problem:
Write an absolute value `proc` which uses promotion to accept either a single `real` value or a `real` array. 

Call:
```
absoluteVal(-inf);
arrayAbsVal([-3.14, 7:real, 0.0, inf, -inf]);
```

<details>
  <summary>Potential Solution</summary>

```Chapel
proc absoluteVal(a: real) {
  return if a >= 0 then a else -a;
}
```
</details>
Expected Output:

```console
inf
3.14 7.0 0.0 inf inf
```

<a id="filter"></a>
## Filtering
Let's say we want to negate the even values less than 10 and drop the others on the floor.
We can iterate `0..<10` and filter out values that don't match our condition.
```chapel
writeln([i in 0..<10] if i%2 == 0 then -i);
```
```console
0 -2 -4 -6 -8
```
Notice this is essentially a ternary but without an `else`.

<a id="TIY_sum_odd_sqaures"></a>
#### Try It Yourself: Sum Odd Perfect Squares <=25
Problem:
Use filtering and reduce to sum all odd perfect squares less than or equal to `25`

Expected Output:

```console
35
```
<details>
  <summary>Potential Solutions</summary>

Using a forall loop
```Chapel
var arr: [0..5] int;

forall i in 0..5 {
  if i%2 != 0 {
    arr[i] = i**2;
  }
}

writeln(+ reduce arr);
```

using a forall expression
```Chapel
writeln(+ reduce [i in 0..5] if i%2 != 0 then i**2);
```

</details>

<a id="bool_expand_and_compress"></a>
## Boolean Compression and Expansion Indexing
A bit of a warning, this section is possibly the most difficult in the tutorial,
so don't worry if you don't understand everything! No new material is introduced
in this part, it's just an application of functionality already covered. So
if you find yourself getting overwhelmed or intimidated, feel free to skip to the next section
([Locales and `coforall` loops](#loc_and_coforall)). 

Let's dig into boolean indexing!
Applications of this and similar logic pop up in various places in Arkouda.

<a id="comp_ind"></a>
#### Boolean Compression Indexing
Compression indexing is reducing an array to only the values meeting a certain condition.
This is a common operation in numpy.

Compression indexing in `numpy`:
```python
>>> X = np.array([1, 2, 5, 5, 1, 5, 2, 5, 3, 1])
>>> Y = X[X==5]
>>> Y
array([5, 5, 5, 5])
```
We can accomplish the same result in Chapel using tools from this guide!
```chapel
var X = [1, 2, 5, 5, 1, 5, 2, 5, 3, 1];
writeln("X = ", X, "\n");

// we begin by creating a boolean array, `truth`, indicating where the condition is met
var truth = (X == 5);
writeln("truth = ", truth);

// we use `truth` to create the indices, `iv`, into the compressed array
// `+ scan truth - truth` is essentially creating an exclusive scan
// note: `iv[truth] = [0, 1, 2, 3]`
var iv = (+ scan truth) - truth;
writeln("iv = ", iv);
writeln("iv[truth] = ", [(t, v) in zip(truth, iv)] if t then v, "\n");

// we then create the return array `Y`
// it contains all the elements where the condition is met
// so its size is the number of `True`s i.e. `+ reduce truth`
var Y: [0..<(+ reduce truth)] int;
writeln("+ reduce truth = ", + reduce truth);
writeln("0..<(+ reduce truth) = ", 0..<(+ reduce truth), "\n");

// now that we have the setup, it's time for the actual indexing
// we use a forall to iterate over the indices of `X`
// we only act if the condition is met i.e. truth[i] is true
// we then use the compressed indices `iv[i]` to write into `Y`
// while using the original indices `i` to get the correct value from `X`
forall i in X.domain {
  if truth[i] {
    Y[iv[i]] = X[i];
  }
}

// NOTE:
// we could also use zippered iteration here since
// `truth`, `X`, and `iv` have the same domain.
// Using that and a may-parallel `forall` gives: 
// [(t, x, v) in zip(truth, X, iv)] if t {Y[v] = x;}

writeln("Y = ", Y);
```
```console
X = 1 2 5 5 1 5 2 5 3 1

truth = false false true true false true false true false false
iv = 0 0 0 1 2 2 3 3 4 4
iv[truth] = 0 1 2 3

+ reduce truth = 4
0..<(+ reduce truth) = 0..3

Y = 5 5 5 5
```
Awesome! We reproduced useful numpy functionality using only information from this guide!
With the added benefit that our implementation is parallel and works on distributed data!
Now let's tackle the very similar challenge of expansion indexing.

<a id="expan_ind"></a>
#### Boolean Expansion Indexing
Expansion indexing is writing a smaller array into a larger array at only the values meeting a certain condition.
This is another common operation in numpy.

Expansion indexing in `numpy`:
```python
>>> X = np.array([1, 2, 5, 5, 1, 5, 2, 5, 3, 1])
>>> Y = np.array([-9, -8, -7, -6])
>>> X[X==5] = Y
>>> X
array([ 1,  2, -9, -8,  1, -7,  2, -6,  3,  1])
```
We can accomplish the same result in Chapel using tools from this guide!
```chapel
var X = [1, 2, 5, 5, 1, 5, 2, 5, 3, 1];
var Y = [-9, -8, -7, -6];
writeln("X = ", X);
writeln("Y = ", Y, "\n");

// we begin by creating a boolean array, `truth`, indicating where the condition is met
var truth = (X == 5);
writeln("truth = ", truth);

// we use `truth` to create the indices, `iv`, into the compressed array
// `+ scan truth - truth` is essentially creating an exclusive scan
// note: `iv[truth] = [0, 1, 2, 3]`
var iv = (+ scan truth) - truth;
writeln("iv = ", iv);
writeln("iv[truth] = ", [(t, v) in zip(truth, iv)] if t then v, "\n");

// now that we have the setup, it's time for the actual indexing
// notice this is equivalent to compression indexing with the assignment swapped
// we use a forall to iterate over the indices of `X`
// we only act if the condition is met i.e. truth[i] is true
// we use the original indices `i` to write into `X`
// while using the compressed indices `iv[i]` to get the correct value from `Y`
forall i in X.domain {
  if truth[i] {
    X[i] = Y[iv[i]];
  }
}

// NOTE:
// we could also use zippered iteration here since
// `truth`, `X`, and `iv` have the same domain.
// Using that and a may-parallel `forall` gives: 
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
Great! Now let's use this for a Try It Yourself!

<a id="TIY_arr_even_repl"></a>
#### Try It Yourself: Array Even Replace
Problem:

Use the following function signature to create a `proc`:
```Chapel
proc arrayEvenReplace(in A: [] int, B: [] int)
```
Then replace the even values of `A` with the values of `B` and return `A`.
You can assume the size of `B` will be equal to number of even values in `A`.

It may be helpful to review boolean expansion indexing

Note:
We use an [`in` argument intent](https://chapel-lang.org/docs/primers/procedures.html#argument-intents)
in the function signature to allow us to modify `A`.

Call:
  - `arrayEvenReplace([8, 9, 7, 2, 4, 3], [17, 19, 21]);`
  - `arrayEvenReplace([4, 4, 7, 4, 4, 4], [9, 9, 9, 9, 9]);`

Expected Output:

```console
17 9 7 19 21 3
9 9 7 9 9 9
```

<details>
  <summary>Potential Solutions</summary>

Using a `forall` loop
```Chapel
proc arrayEvenReplace(in A: [] int, B: [] int) {
  var isEven = (A % 2 == 0);
  var expandedIdx = (+ scan isEven) - isEven;
  forall (even, a, i) in zip(isEven, A, expandedIdx) {
     if even {
       a = B[i];
     }
   }
  return A;
}
```

using a `forall` expression
```Chapel
proc arrayEvenReplace(in A: [] int, B: [] int) {
  var isEven = (A % 2 == 0);
  var expandedIdx = (+ scan isEven) - isEven;
  [(even, a, i) in zip(isEven, A, expandedIdx)] if even {a = B[i];}
  return A;
}
```
</details>

<a id="loc_and_coforall"></a>
## Locales and `coforall` loops

<a id="locale"></a>
### Locales
We've mentioned _locales_ briefly in earlier sections, but let's dive into them a bit deeper.
I like to think of a locale as a single computer in a cluster of computers all working together to solve a problem.
This isn't always the case, but it is a useful model. More generally a
[locale](https://chapel-lang.org/docs/users-guide/locality/localesInChapel.html) is
"a piece of a target architecture that has processing and storage capabilities".


For any given computer in our cluster, accessing the data it stores locally will be faster than having it fetch data
from a different computer in the cluster. This is because the data will need to be transferred from the computer
that has it to the computer that needs it. 

Data stored on a computer is "local" to that computer and data stored on
a different computer is "remote". This terminology reflects the time penalty we have to pay when accessing "remote" data when compared
to "local" data. This same intuition holds for locales in general.

The big takeaway is it's in our best interest to minimize how often locales need to operate
on remote data.

To reiterate a bit more exactly:

> Say `x` and `y` are both stored on `locale_i`. From the perspective of `locale_i`, we say `x` and `y` are
both local. There is no significant difference in access times.
> 
> But if `x` is on `locale_i` and `y` is on `locale_j`. From the perspective of `locale_i`, we say `x` is local
and `y` is remote. In this case, accessing `y` would take longer than `x`.

At runtime, Chapel assigns each locale a number from `0..<numLocales`. And if you were to create a program with a
single simple statement like `var x = 5+3;`, it will be executed on locale 0. So how would we perform the
computation on a different locale?  

The most direct way is to use an [`on` clause](https://chapel-lang.org/docs/users-guide/locality/onClauses.html).
```Chapel
// happens on locale 0
var x = 5+3;

on Locales[2] {
  // happens on locale 2
  var y = 5+3;
}
```
But sometimes you don't need an explict `on` clause to operate on other locales.
One example of this is a `forall` loop over a distributed array or domain.

<a id="coforall"></a>
### `coforall` Loops
The second most common parallel loop in Arkouda is the [`coforall`](https://chapel-lang.org/docs/users-guide/taskpar/coforall.html) loop.
Let's see how this compares to the [`forall`](https://chapel-lang.org/docs/users-guide/datapar/forall.html)
loops that we've been using up until now.

The biggest differences between a `coforall` and `forall` is how many tasks are created and which locales
execute these tasks. 
I think of `forall` loops as a reasonable default where Chapel determines this for you.
A `coforall` offers more control, but the tradeoff is you might need to manage some stuff yourself

A `coforall` loop:
* creates one distinct task per loop iteration, each of which executes a copy of the loop body.
* [task based parallelism](https://chapel-lang.org/docs/users-guide/index.html#task-parallelism):
I want exactly this many tasks operating in parallel

A `forall` loop:
* creates a variable number of tasks determined by the data it's iterating and the number of available processors.
* [data based parallelism](https://chapel-lang.org/docs/users-guide/index.html#data-parallelism): operate on this
data in parallel

Here's a simple example of a `coforall`
```Chapel
var numTasks = 8;

coforall tid in 1..numTasks {
  writeln("Hello from task ", tid, " of ", numTasks);
}

writeln("Signing off...");
```
```console
Hello from task 4 of 8
Hello from task 1 of 8
Hello from task 2 of 8
Hello from task 5 of 8
Hello from task 3 of 8
Hello from task 6 of 8
Hello from task 7 of 8
Hello from task 8 of 8
Signing off...
```

<a id="compile-multiloc"></a>
### Enabling multiple locales
For the rest of this tutorial, we will be using Chapel with
[multiple locales](https://chapel-lang.org/docs/users-guide/locality/compilingAndExecutingMultiLocalePrograms.html).

If you're not running on a distributed system, Chapel will simulate running in a distributed way as if you had
multiple distinct machines. This is useful for running diagnostics and finding bugs that only show up
in a distributed setting.

It's worth noting the performance of simulating multi-locale on a non-distributed machine isn't a good indicator
of how it will perform on an actual distributed system, but the amount of communication
taking place should be the same. Minimizing communication is key to writing efficient chapel code
and will be covered in later sections.

To change your Chapel build to enable this, follow the instructions [here](https://bears-r-us.github.io/arkouda/developer/GASNET.html).
If this is your first time configuring your environment variables to enable multi-locale, you'll need to rebuild `chpl`.
After that you can switch into multi-locale mode by setting the same environment variables. To switch back, reset
your environment variables to their original state. 

The command to compile multi-locale programs doesn't change, but running the executable now requires a `-nl` flag to specify
the number of locales.

```console
# to run with 2 locales
$ ./tutorial -nl 2

# to run with 5 locales
$ ./tutorial -nl 5
```

All the previous sections should work with multiple locales.

<a id="locale_looping"></a>
### Looping Locales with `coforall`
The most common use of `coforall` in Arkouda is to create a single task for every locale.
We can then use `on` blocks to have a single task run on each locale.

The most common distribution used in Arkouda is the
[block distribution](https://chapel-lang.org/docs/primers/distributions.html#the-block-distribution).
In this distribution the elements are split as evenly as possible across all locales.

Let's look at an example to visualize how the data is distributed for block distributed arrays.


To do this we'll use an [on clause](https://chapel-lang.org/docs/users-guide/locality/onClauses.html) to control
which locale the computation is occurring on.

We'll also need 
[local subdomain](https://chapel-lang.org/docs/primers/distributions.html#block-and-distribution-basics),
which is the subsection of the domain that is stored on the locale where the computation is happening,
```Chapel
use BlockDist;

// we create a block distributed array and fill it with values from 1 to 16
var A = blockDist.createArray({1..16}, int);
A = 1..16;

// we use a coforall to create one task per Locales 
coforall loc in Locales {
  // Then we use an `on` clause to execute on Locale number `loc`
  // so now there is exactly one task executing on each locale
  on loc {
    // Next we get the local part of `A` by slicing at it's local subdomain
    var localA = A[A.localSubdomain()];
    writeln("The chunk of A owned by Locale ", loc.id, " is: ", localA);
  }
}
```
When running with `CHPL_COMM=none`, we see there's only one locale that owns all the data.
```console
$ ./tutorial
The chunk of A owned by Locale 0 is: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```
After enabling multi-locale and recompiling, we can try this out with
different numbers of locales to see how the distribution changes!
```console
$ ./tutorial -nl 2
The chunk of A owned by Locale 0 is: 1 2 3 4 5 6 7 8
The chunk of A owned by Locale 1 is: 9 10 11 12 13 14 15 16

$ ./tutorial -nl 4
The chunk of A owned by Locale 0 is: 1 2 3 4
The chunk of A owned by Locale 1 is: 5 6 7 8
The chunk of A owned by Locale 3 is: 13 14 15 16
The chunk of A owned by Locale 2 is: 9 10 11 12

$ ./tutorial -nl 8
The chunk of A owned by Locale 0 is: 1 2
The chunk of A owned by Locale 6 is: 13 14
The chunk of A owned by Locale 2 is: 5 6
The chunk of A owned by Locale 7 is: 15 16
The chunk of A owned by Locale 3 is: 7 8
The chunk of A owned by Locale 5 is: 11 12
The chunk of A owned by Locale 4 is: 9 10
The chunk of A owned by Locale 1 is: 3 4
```

<a id="forall_distribution"></a>
### Implicit distributed computation with `forall`

We mentioned earlier one way to do distributed computation without an explicit `on` statement
is using a `forall` on a distributed domain or array. To demonstrate this we're going
to use the `here` keyword which refers to the locale where the computation is taking place.

```Chapel
use BlockDist;

var MyDistArr = blockDist.createArray({1..16}, int);
MyDistArr = 1..16;

forall i in MyDistArr.domain {
  writeln("element ", i, " (", MyDistArr[i], ") is owned by locale ", here.id);
}
```

```console
$ ./tutorial -nl 4
element 1 (1) is owned by locale 0
element 2 (2) is owned by locale 0
element 4 (4) is owned by locale 0
element 3 (3) is owned by locale 0
element 5 (5) is owned by locale 1
element 9 (9) is owned by locale 2
element 14 (14) is owned by locale 3
element 11 (11) is owned by locale 2
element 15 (15) is owned by locale 3
element 8 (8) is owned by locale 1
element 10 (10) is owned by locale 2
element 12 (12) is owned by locale 2
element 16 (16) is owned by locale 3
element 13 (13) is owned by locale 3
element 7 (7) is owned by locale 1
element 6 (6) is owned by locale 1
```


<a id="aggregation"></a>
## Aggregation

For the best performance we want to minimize how often we move data between locales. But there are situations
where you need to move lots of data between locales. For example assigning local values into a distributed array
at random indices. In this case, one locale might be copying multiple elements to every other locale.
Sending each of those elements one at a time results in a lot of communication. Aggregation is used to mitigate
this.

I like to think of this in terms of transporting people, where our goal is minimizing the total number of trips.
If a bunch of people are going to the same place, they could all travel there individually. But we'd have
fewer total trips if we wait until we have enough people to fill a bus and send a whole group at once.

[Copy aggregation](https://chapel-lang.org/docs/modules/packages/CopyAggregation.html#copyaggregation)
is the same idea. Instead of copying each value individually, we wait until we have a bunch
that are all going to the same locale and copy them over all at once.

There are two types of copy aggregation:
* copying local values into remote variables (`DstAggregator`)
* copying remote values into local variables (`SrcAggregator`)

The question of which aggregator do I need to use can sometimes trip people up.
This is because the same operation could use either a source or destination aggregator
depending on which locale is performing the operation.

For example:
> Let's say `x` is stored on `locale_i` and we want to copy `x` into a position on `locale_j`.
Which aggregator do we use? It depends
> 
> If `locale_i` is the one doing the computation, then `x` is a local value.
Since we're viewing this from perspective of `locale_i`, positions on `locale_j` are remote.
So we need a `DstAggregator` because we are putting a local value into a remote position.
>
> If instead `locale_j` is the one doing this computation, the position
we want to write into is local.
And from `locale_j`'s perspective `x` is remote. 
Since we're getting a remote value and writing it into a local position, we need a `SrcAggregator`.

It's important to note copy aggregation will only work if at least one side is local.
Both sides being remote (remote-to-remote aggregation) is not currently supported.

Syntax:
```chapel
// without aggregation; every element is sent immediately 
forall (i, v) in zip(inds, vals) {
  remoteArr[i] = v;
}

// with aggregation; wait until we have a few and send them together
use CopyAggregation;
forall (i, v) in zip(inds, vals) with (var agg = new DstAggregator(int)) {
  agg.copy(remoteArr[i], v);
}
```
The `with` keyword here declares `agg` as a [Task-Private variable](https://chapel-lang.org/docs/language/spec/data-parallelism.html#task-private-variables),
meaning each task will get its own private variable with the same name.

Let's look at an example, where we copy values from one distributed array into another but shifted over by 3.
We'll call the array we are copying from `src` and the array we are copying to `dst`.
This example is designed so when ran with 2 locales, every value will be copied to or from a remote location.

When you do a `forall` loop over a distributed array, each locale will handle the values that are local to it.
So we can go about this 2 different ways:
* We could loop over the `src` array, so the source values will always be local. We then copy our local values
into a remote position of the `dst` array using a `DstAggregator`
* Or we loop over the `dst` array, so destination values are local. We then want to copy remote values from
the `src` array into our local `dst` array positions using a `SrcAggregator`

When doing a `forall` loop over a distributed array, I like to think of it as doing the computation from the
perspective of that array. For this example, we can accomplish our goal just as easily using source or destination
aggregation. But in some cases one may be easier than the other.

We'll use a [config variable](https://chapel-lang.org/docs/users-guide/base/configs.html)
to switch between source and destination aggregation. When `UseDstAgg` is set to `true` (the default), it will use
destination aggregation. To switch to source aggregation we need to add the flag `--UseDstAgg=false` to our execution
command.

```chapel
use BlockDist, CopyAggregation;

config const UseDstAgg = true;

const dom = blockDist.createDomain({0..<6});

// named src because this is the source we are copying from
var src: [dom] int = [0, 1, 2, 3, 4, 5];

// named dst because this is the destination we are copying to
var dst: [dom] int;

writeln("src: ", src);
writeln("dst: ", dst);

if UseDstAgg {
    // when the destination is remote we use a dstAggregator
    forall (s, i) in zip(src, 0..) with (var agg = new DstAggregator(int)) {
      // locNum is which locale this loop iteration is executing on
      var locNum = here.id;

      // localSubDom is the chunk of the distributed arrays that live on this locale
      var localSubDom = dom.localSubdomain();
      
      // we use a single writeln to avoid interleaving output from another locale 
      writeln("\niteration num: ", i, "\n  on Locale: ", locNum,
              "\n  on localSubDom: ", localSubDom, "\n  src[", i, "] is local",
              "\n  dst[", (i + 3) % 6, "] is remote");
    
      // since dst is remote, we use a dst aggregator
      // assignment without aggregation would look like:
      // dst[ (i + 3) % 6 ] = s
      agg.copy(dst[ (i + 3) % 6 ], s);
    }
    writeln();
    writeln("src: ", src);
    writeln("dst: ", dst);
}
else {
    // when the source is remote we use a srcAggregator
    forall (d, i) in zip(dst, 0..) with (var agg = new SrcAggregator(int)) {
      // locNum is which locale this loop iteration is executing on
      var locNum = here.id;
      // localSubDom is the chunk of the distributed arrays that live on this locale
      var localSubDom = dom.localSubdomain();
      
      // we use a single writeln to avoid interleaving output from another locale 
      writeln("\niteration num: ", i, "\n  on Locale: ", locNum,
              "\n  on localSubDom: ", localSubDom, "\n  src[", (i + 3) % 6, "] is remote",
              "\n  dst[", i, "] is local");
    
      // since src is remote, we use a src aggregator
      // assignment without aggregation would look like:
      // d = src[ (i + 3) % 6 ]
      agg.copy(d, src[ (i + 3) % 6 ]);
    }
    writeln();
    writeln("src: ", src);
    writeln("dst: ", dst);
}
```

using a `DstAggregator`:
```console
$ chpl tutorial.chpl --no-cache-remote
$ ./tutorial -nl 2

src: 0 1 2 3 4 5
dst: 0 0 0 0 0 0

iteration num: 0
  on Locale: 0
  on localSubDom: {0..2}
  src[0] is local
  dst[3] is remote

iteration num: 2
  on Locale: 0
  on localSubDom: {0..2}
  src[2] is local
  dst[5] is remote

iteration num: 1
  on Locale: 0
  on localSubDom: {0..2}
  src[1] is local
  dst[4] is remote

iteration num: 3
  on Locale: 1
  on localSubDom: {3..5}
  src[3] is local
  dst[0] is remote

iteration num: 5
  on Locale: 1
  on localSubDom: {3..5}
  src[5] is local
  dst[2] is remote

iteration num: 4
  on Locale: 1
  on localSubDom: {3..5}
  src[4] is local
  dst[1] is remote

src: 0 1 2 3 4 5
dst: 3 4 5 0 1 2
```

using a `SrcAggregator`:
```console
$ ./tutorial -nl 2 --UseDstAgg=false
src: 0 1 2 3 4 5
dst: 0 0 0 0 0 0

iteration num: 0
  on Locale: 0
  on localSubDom: {0..2}
  src[3] is remote
  dst[0] is local

iteration num: 1
  on Locale: 0
  on localSubDom: {0..2}
  src[4] is remote
  dst[1] is local

iteration num: 2
  on Locale: 0
  on localSubDom: {0..2}
  src[5] is remote
  dst[2] is local

iteration num: 3
  on Locale: 1
  on localSubDom: {3..5}
  src[0] is remote
  dst[3] is local

iteration num: 5
  on Locale: 1
  on localSubDom: {3..5}
  src[2] is remote
  dst[5] is local

iteration num: 4
  on Locale: 1
  on localSubDom: {3..5}
  src[1] is remote
  dst[4] is local

src: 0 1 2 3 4 5
dst: 3 4 5 0 1 2
```

<a id="perf"></a>
## Performance and Diagnostics
Now that you're a pro at writing chapel code, let's talk about how to make the code you write more efficient!

<a id="var_dec"></a>
### Variable Declarations
Throughout this tutorial we've mainly used `var` to declare variables, but this is not the only option.
The driving factor of using other declarations is to improve performance or reduce memory usage.

[`const`](https://chapel-lang.org/docs/users-guide/base/constParam.html):
* can be used when we know a variable should never be changed
* a program execution time constant. This means the compiler doesn't need to know its value, and it can be 
set at runtime (i.e. passed in by a user)
* knowing this value won't change can help the compiler to make optimizations
* acts as a guardrail to prevent you from modifying the variable accidentally
* programmers should always reach for const when they have variables that they know won't be changing
for both the performance and code safety benefits

[`param`](https://chapel-lang.org/docs/users-guide/base/constParam.html#declaring-params):
* similar to `const`, except it must be known at compile time.

[`ref`](https://chapel-lang.org/docs/language/spec/variables.html?highlight=ref#ref-variables):
* creates a reference to an existing object
* avoids creating a copy (less memory use)
* any updates to this will update the object it is referencing 
* useful to avoid repeated accesses of an attribute (especially within a loop)


[`const ref`](https://chapel-lang.org/docs/language/spec/variables.html?highlight=ref#ref-variables)
* same as a `ref` but cannot be changed

<a id="diagnostics"></a>
### Diagnostics

There are several chapel modules available to aid in optimizing your code. The most common ones
used by the Arkouda team are [CommDiagnostics](https://chapel-lang.org/docs/modules/standard/CommDiagnostics.html)
and [Time](https://chapel-lang.org/docs/modules/standard/Time.html).
There is also
[memory diagnostics](https://chapel-lang.org/docs/modules/standard/Memory/Diagnostics.html),
which is used less frequently.

For this section we will add the flag `--no-cache-remote` to our compilation command to make
the output of comm diagnostics a bit simpler.

<a id="agg_comm"></a>
### Aggregation reducing Communication
Let's craft an example where we are intentionally copying remote values to show the
communication that takes place. We'll be comparing aggregation, regular assignment, and a secret third way. 

In this example, we use a
[`config const`](https://chapel-lang.org/docs/users-guide/base/configs.html#config-var-and-config-const)
to set the size. We change the size of our arrays by adding `--size=` to our execution command.

```Chapel
use BlockDist, CommDiagnostics, Time, CopyAggregation;
// communication comparison betweeen using aggregation and straight writing
// compile with --no-cache-remote

config const size = 10**6;
config const compareBulkTransfer = false;
const dom = blockDist.createDomain({0..<size});

// named src because this will be the source we are copying from
var src: [dom] int = dom;

// named dst because this will be the destination we are copying to
var dst: [dom] int;

resetCommDiagnostics();
startCommDiagnostics();
var t1 = Time.timeSinceEpoch().totalSeconds();

forall (s, i) in zip(src, 0..) {
  dst[ (i + (size / 2):int ) % size ] = s;
}

var t2 = Time.timeSinceEpoch().totalSeconds() - t1;
stopCommDiagnostics();
writeln("copy without aggregation time = ", t2);
writeln("communication without aggregation: ");
printCommDiagnosticsTable();

resetCommDiagnostics();
startCommDiagnostics();
t1 = Time.timeSinceEpoch().totalSeconds();

forall (s, i) in zip(src, 0..) with (var agg = new DstAggregator(int)) {
  agg.copy(dst[ (i + (size / 2):int ) % size ], s);
}

t2 = Time.timeSinceEpoch().totalSeconds() - t1;
stopCommDiagnostics();
writeln();
writeln("copy with aggregation time = ", t2);
writeln("communication using aggregation: ");
printCommDiagnosticsTable();

if compareBulkTransfer {
  resetCommDiagnostics();
  startCommDiagnostics();
  var t3 = Time.timeSinceEpoch().totalSeconds();
  
  // using aggregation is not actually needed
  // since we are copying a contiguous block 
  dst[0..<(size / 2)] = src[(size / 2)..<size];
  dst[(size / 2)..<size] = src[0..<(size / 2)];
  
  var t4 = Time.timeSinceEpoch().totalSeconds() - t3;
  stopCommDiagnostics();
  writeln();
  writeln("copy with aggregation time = ", t4);
  writeln("communication using aggregation: ");
  printCommDiagnosticsTable();
}
```
The output of `printCommDiagnosticsTable()` has several categories

At a high level these categories mean:
* "get": reading a value from a remote locale's memory
* "put": storing a value into a remote locale's memory
* "execute_on": creating a task on a remote locale, for example via an on-clause
* "execute_on_nb": same as above but non-blocking (doesn't prevent the original task from continuing)
```console
$ chpl tutorial.chpl --no-cache-remote
$ ./tutorial -nl 4
copy without aggregation time = 13.2172
communication without aggregation:
| locale | get |    put | execute_on_nb |
| -----: | --: | -----: | ------------: |
|      0 |   7 | 250000 |             3 |
|      1 |   7 | 250000 |             0 |
|      2 |   7 | 250000 |             0 |
|      3 |   7 | 250000 |             0 |

copy with aggregation time = 0.159691
communication using aggregation:
| locale | get | put | execute_on | execute_on_nb |
| -----: | --: | --: | ---------: | ------------: |
|      0 |  80 |  40 |         40 |             3 |
|      1 |  80 |  40 |         40 |             0 |
|      2 |  80 |  40 |         40 |             0 |
|      3 |  80 |  40 |         40 |             0 |
```

We see the number of `put`s has decreased drastically, and it's way faster! Aggregation is great!
But what's this secret third way?

Until now, we've been comparing aggregation to copying values individually.
And that makes sense if we're copying to/from random indices. But there's a lot of structure 
in this case that we're not using; we just want to swap the two halves. So why can't we just do two large copies?

We can! We call it bulk transfer. In general if you want to copy a contiguous chunk of values into another 
contiguous chunk of values, it's probably most efficient to do it in one go. 

The code to do this looks something like this:

```chapel
// copy second half of src into first half of dst
dst[0..<(size / 2)] = src[(size / 2)..<size];

// copy first half of src into second half of dst
dst[(size / 2)..<size] = src[0..<(size / 2)];
```
This is already present in our example and can be enabled by setting the config const
`compareBulkTransfer` to true.

```console
$ ./tutorial -nl 4 --compareBulkTransfer=true
copy without aggregation time = 12.974
communication without aggregation:
| locale | get |    put | execute_on_nb |
| -----: | --: | -----: | ------------: |
|      0 |   7 | 250000 |             3 |
|      1 |   7 | 250000 |             0 |
|      2 |   7 | 250000 |             0 |
|      3 |   7 | 250000 |             0 |

copy with aggregation time = 0.233566
communication using aggregation:
| locale | get | put | execute_on | execute_on_nb |
| -----: | --: | --: | ---------: | ------------: |
|      0 |  80 |  40 |         40 |             3 |
|      1 |  80 |  40 |         40 |             0 |
|      2 |  80 |  40 |         40 |             0 |
|      3 |  80 |  40 |         40 |             0 |

copy with aggregation time = 0.0141501
communication using aggregation:
| locale | get | put | execute_on_nb |
| -----: | --: | --: | ------------: |
|      0 |   6 |   0 |            75 |
|      1 |  54 |   4 |             0 |
|      2 |  54 |   4 |             0 |
|      3 |  54 |   4 |             0 |
```

<a id="pitfalls"></a>
### Common Pitfalls
In this section we cover some common pitfalls that can hurt performance.
  - Using ranges as leading iterators of parallel loops with distributed arrays 
    - i.e. `forall (i, a) in zip(0..<A.size, A)`
    - Ranges aren't distributed, so this actually turns into a parallel loop that only executes on locale 0.
    So a task per core on locale 0 will be executing, but the rest of the machine(s) will be unused.
  - Not using aggregation (with random indices) or bulk transfer (with contiguous indices) for copying.
    - We saw in the previous section what a big difference these can make when copying random elements
    out of or into a distributed array.
  - Not using refs for component accesses inside loops.
    - If you find yourself accessing a component of a class during every iteration of a loop, it's
    probably worth saving a reference to that component to avoid fetching it every time.

Check out this [actual discussion](https://github.com/Bears-R-Us/arkouda/pull/2159#discussion_r1113699483)
in Arkouda that highlights a lot of these points.

Congrats!! You've completed the Chapel tutorial!

## Want to learn more?

Most of this functionality (and much more) is covered in depth in the Chapel documentation!
Check out their [User's Guide](https://chapel-lang.org/docs/users-guide/index.html)
and [primers](https://chapel-lang.org/docs/primers/index.html).
