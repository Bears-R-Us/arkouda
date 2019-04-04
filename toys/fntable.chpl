use MultiTypeSymbolTable;

//
// create a range that specifies the types that we support using the
// seType enum defined in MultiTypeSymbolTable:
//
const types = seType.Int..seType.Real;

//
// create an array over {types x types} where each element is a
// function that takes in two (borrowed) GenSymEntries and returns a
// shared GenSymEntry.
//
var FnTable: [types, types] func(GenSymEntry, GenSymEntry, shared GenSymEntry);

//
// populate the table
//
FnTable[seType.Int,  seType.Int]  = addII;
FnTable[seType.Real, seType.Int]  = addRI;
FnTable[seType.Int,  seType.Real] = addIR;
FnTable[seType.Real, seType.Real] = addRR;

// now let's exercise this stuff
proc main() {
  // create a SymEntry for each of the int and real types:
  var seI: GenSymEntry = new owned SymEntry(3, int);
  var seR: GenSymEntry = new owned SymEntry(3, real);

  // initialize them:
  seI = 42;
  seR = 3.14;

  // Check that all the add*() routines work as expected:
  writeln(addIR(seI, seR));
  writeln(addII(seI, seI));
  writeln(addRR(seR, seR));
  writeln(addRI(seR, seI));
  writeln();

  // For the sake of testing, store seI and seR into an array (this is
  // why I made the static types of seI and seR a 'GenSymEntry'... so
  // that this array would have a consistent element type).
  var ses = [seI, seR];

  // loop over the elements in the array in a nested loop showing that
  // we can use the enums in GenSymEntry to look up the function.
  for se1 in ses do
    for se2 in ses do
      writeln("adding ", se1, " and ", se2, " results in: ",
              FnTable[se1.rtEtype, se2.rtEtype](se1, se2));
  writeln();

  // OK, so now let's try to clean this up by creating an object
  // around the array that cleans up its use:
  var AddOps = new TypeTypeFnTable(addII, addIR, addRI, addRR);

  // here's that same loop again, but the add is cleaner now:
  for se1 in ses do
    for se2 in ses do
      writeln("adding ", se1, " and ", se2, " results in: ", AddOps(se1, se2));
  writeln();
}

record TypeTypeFnTable {
  //
  // the array of functions
  //
  var fns: [types, types] borrowed func(GenSymEntry, GenSymEntry, shared GenSymEntry);

  //
  // for the initializer, I just pass them in in a well-defined order;
  // this is not particularly clever... we could do more to clean it
  // up, or initialize it via accesses to the table, or ...
  //
  // Note that I need to assure the lifetime checker that this table
  // will not outlive the borrows of the four functions...  The
  // compiler should be smarter about this for function types I
  // believe...
  proc init(II, IR, RI, RR) lifetime this < II, this < IR, this < RI, this < RR {
    use seType;  // 'use' the enum for convenience

    this.complete();      // ensure the array is allocated...

    fns[Int,  Int]  = II; // ...then fill it
    fns[Int,  Real] = IR;
    fns[Real, Int]  = RI;
    fns[Real, Real] = RR;
  }

  //
  // create an accessor for the table that both looks up and calls the
  // correct function.
  //
  proc this(x: GenSymEntry, y: GenSymEntry): shared GenSymEntry {
    return fns[x.rtEtype, y.rtEtype](x, y);
  }
}


//
// My four 'add' functions follow: [real|int] x [real|int]
//
proc addRR(x: GenSymEntry, y: GenSymEntry): shared GenSymEntry {
  var concreteX = x: SymEntry(real);
  var concreteY = y: SymEntry(real);

  assert(concreteX != nil || concreteY != nil, "wrong type in addRR()");
  
  if concreteX.a.size != concreteY.a.size then
    halt("Size mismatch in addRR(): ", concreteX.a.size, " != ", concreteY.a.size);
  
  var res = concreteX.a + concreteY.a;

  return new shared SymEntry(res);
}

proc addIR(x: GenSymEntry, y: GenSymEntry): shared GenSymEntry {
  var concreteX = x: SymEntry(int);
  var concreteY = y: SymEntry(real);

  assert(concreteX != nil || concreteY != nil, "wrong type in addIR()");

  if concreteX.a.size != concreteY.a.size then
    halt("Size mismatch in addIR(): ", concreteX.a.size, " != ", concreteY.a.size);
  
  var res = concreteX.a + concreteY.a;

  return new shared SymEntry(res);
}

proc addRI(x: GenSymEntry, y: GenSymEntry): shared GenSymEntry {
  var concreteX = x: SymEntry(real);
  var concreteY = y: SymEntry(int);

  assert(concreteX != nil || concreteY != nil, "wrong type in addRI()");

  if concreteX.a.size != concreteY.a.size then
    halt("Size mismatch in addRI(): ", concreteX.a.size, " != ", concreteY.a.size);
  
  var res = concreteX.a + concreteY.a;

  return new shared SymEntry(res);
}

proc addII(x: GenSymEntry, y: GenSymEntry): shared GenSymEntry {
  var concreteX = x: SymEntry(int);
  var concreteY = y: SymEntry(int);

  assert(concreteX != nil || concreteY != nil, "wrong type in addRI()");

  if concreteX.a.size != concreteY.a.size then
    halt("Size mismatch in addRI(): ", concreteX.a.size, " != ", concreteY.a.size);
  
  var res = concreteX.a + concreteY.a;

  return new shared SymEntry(res);
}

