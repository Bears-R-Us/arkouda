use BlockDist;

var D = {0..#10};
var DD: domain(1) dmapped Block(boundingBox=D) = D;
var A: [DD] int = [5, 4, 3, 2, 1, 0, 1, 2, 3, 4];
writeln(min scan A);
