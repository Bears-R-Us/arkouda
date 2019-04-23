# coding: utf-8
a = ak.arange(0, 10, 1)
b1 = a < 5
b2 = (a % 2) == 0
a + (b1 ^ b2)
a + True
a * True
a * False
a - True
a.min()
a.max()
f = ak.linspace(0, 8, 10)
f
f + True
f - True
f * True
b1 + 3
b1 + f
b2 + 2
a % (b2 + 2)
aref = ak.array(a)
aref is a
True * f
True + f
