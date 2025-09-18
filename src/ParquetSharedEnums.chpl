/* WARNING: this file is included in C++ code as if it is a regular C header. So
 * even though it has chpl extension, the syntax has to be legal both in Chapel
 * and C++ and the contents should be limited to enum definitions shared between
 * different layers to ensure consistent signalling.
 */

enum NullMode { noNulls=0,
                onlyFloats=1,
                all=2
};
