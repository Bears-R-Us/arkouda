public use Time;
public use CommDiagnostics;
public use IO;

public use ServerConfig;
public use ServerErrorStrings;

public use MultiTypeSymEntry;
public use MultiTypeSymbolTable;

public use SymArrayDmap;
public use SegmentedArray;
public use RandArray;
public use AryUtil;


// Message helpers

proc parseName(s: string): string {
  const low = [1..1].domain.low;
  var fields = s.split();
  return fields[low+1];
}

proc parseTwoNames(s: string): (string, string) {
  var (nameOne, nameTwo) = s.splitMsgToTuple('+', 2);
  return (parseName(nameOne), parseName(nameTwo));
}
