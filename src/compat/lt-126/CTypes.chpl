// In Chapel 1.26, various C types from 'SysCTypes', 'SysBasic' and
// 'CPtr' are being brought together into a single module, 'CTypes'.
// This compatibility module brings the pieces that Arkouda relies
// upon together so that it can use the new organization yet still
// be compiled with older compilers.

public use SysCTypes;
public use CPtr;

type c_size_t = size_t;
