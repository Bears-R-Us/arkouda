

//Tests supported pow operations, vv, vs, sv, **=, etc

var a: [0..#3] int;
var b: [0..#3] real;
for (i,j) in zip(a.domain,b.domain){
    a[i]=i+1;
    b[j]=j+4.456;
}
var c: int =2;
writeln("a is",a);
writeln("b is",b);
writeln(c**a);

b**=c;
writeln(b);