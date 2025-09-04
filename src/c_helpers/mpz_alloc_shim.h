#ifndef MPZ_ALLOC_SHIM_H
#define MPZ_ALLOC_SHIM_H

#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return capacity (in limbs)
int ark_mpz_alloc_from_struct_p(const __mpz_struct* s);

// (optional) Return used size (in limbs, signed; sign encodes mpz sign)
int ark_mpz_size_from_struct_p(const __mpz_struct* s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MPZ_ALLOC_SHIM_H
