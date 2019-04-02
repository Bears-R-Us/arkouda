use BlockDist;

config param debug = false;

// how many unique values will we see in our data array?
const numUniqueEntries = 4;

// lame way to bootstrap a distributed array of data values from 1..4
var Aloc = [1, 4, 3, 2, 1, 4, 3, 2, 2, 2, 4, 3, 1, 4, 1, 3, 4, 1, 3, 3, 2, 1, 4, 2, 4, 2, 3, 2, 1, 4, 3, 2, 1, 3, 4, 2, 3, 1, 4, 4, 2, 3, 1, 4, 1, 2, 3, 4, 2, 4, 3, 4, 4, 2, 2];
var D = newBlockDom(1..Aloc.size);
var A: [D] int = Aloc;


// Compute the correct answer trivially / serially
var counts: [1..numUniqueEntries] int;

for a in Aloc do
  counts[a] += 1;

writeln("counts are: ", counts);

const answer = + scan counts;

writeln("answer is: ", answer);

// Prove to myself that the answer is correct
var Asorted: [D] int;

var answerClone = answer;
for a in Aloc {
  Asorted[answerClone[a]] = a;
  answerClone[a] -= 1;
}

writeln("sorted array is: ", Asorted);


// It's annoying that I have to define these in order to run a +
// reduction over an array of atomic ints, but that seems to be
// the case at present...  I should file an issue for this.
proc +(x: atomic int, y: atomic int) {
  return x.read() + y.read();
}

proc +=(X: [?D] int, Y: [D] atomic int) {
  forall i in D do 
    X[i] += Y[i].read();
}


// Take 1: Use reduce intents on a forall loop to (a) give each task
// its local copy of the counters variable and (b) to + reduce them as
// the tasks complete.  Note that this does not leave us with per-task
// / per-locale copies of the counters arrays when the forall loop has
// completed, so just global information.

{
  var counters: [1..numUniqueEntries] int;

  forall a in A with (+ reduce counters) do
    counters[a] += 1;

  var myAnswer = + scan counters;

  if (|| reduce (answer != myAnswer)) {
    writeln("reduce intents didn't get the right answer:");
    writeln("counters was: ", counters);
    writeln("answer was: ", myAnswer);
  } else {
    writeln("reduce intents version passed!");
  }
}


// Take 2: Use the `PrivateSpace` domain to give each locale its own
// counter array.  The `PrivateSpace` domain has an element per
// locale, indexed using the locale's ID.  Note that I'm using an
// atomic int here since each locale will likely have multiple tasks
// iterating over its chunk of A at a time.

{
  use PrivateDist;
  
  var counters: [PrivateSpace] [1..numUniqueEntries] atomic int;

  forall a in A do
    counters[here.id][a].add(1); // index into counters using our locale's ID

  if debug {
    writeln("private counters:");
    for loc in LocaleSpace do
      writeln(counters[loc]);
  }
  
  var globCounters = + reduce [i in PrivateSpace] counters[i];

  var myAnswer = + scan globCounters;

  if (|| reduce (answer != myAnswer)) {
    writeln("private distribution didn't get the right answer:");
    writeln("final counters were: ", globCounters);
    writeln("answer was: ", myAnswer);
  } else {
    writeln("private distribution version passed!");
  }
}

// Take 3: Use the `Replicated` distribution to accomplish a similar
// per-locale counter array.  Here, each locale automatically gets its
// own copy of the array and automatically refers to its local copy.
{
  use ReplicatedDist;

  const counterDom = {1..numUniqueEntries} dmapped Replicated();
  var counters: [counterDom] atomic int;

  forall a in A do
    counters[a].add(1);

  if debug {
    writeln("replicated counters:");
    for loc in Locales do
      writeln(counters.replicand(loc));
  }
  
  var globCounters = + reduce [loc in Locales] counters.replicand(loc);

  var myAnswer = + scan globCounters;

  if (|| reduce (answer != myAnswer)) {
    writeln("replicated distribution didn't get the right answer:");
    writeln("final counters were: ", globCounters);
    writeln("answer was: ", myAnswer);
  } else {
    writeln("replicated distribution version passed!");
  }
}

// Take 4: Use a Block distribution to more manually accomplish the
// same as the previous two versions.
{
  use BlockDist;

  const onePerLoc = newBlockDom({0..#numLocales});
  var counters: [onePerLoc] [1..numUniqueEntries] atomic int;

  forall a in A do
    counters[here.id][a].add(1);

  if debug {
    writeln("block counters:");
    for loc in Locales do
      writeln(counters[loc.id]);
  }
  
  var globCounters = + reduce [loc in Locales] counters[loc.id];

  var myAnswer = + scan globCounters;

  if (|| reduce (answer != myAnswer)) {
    writeln("block distribution didn't get the right answer:");
    writeln("final counters were: ", globCounters);
    writeln("answer was: ", myAnswer);
  } else {
    writeln("block distribution version passed!");
  }
}

// Take 5: Use a Block distribution, but create an index per task
// rather than per locale to avoid having to use atomic ints.

// Left for another day...
