#include "mpz_alloc_shim.h"

int ark_mpz_alloc_from_struct_p(const __mpz_struct* s) {
    return s ? s->_mp_alloc : 0;
}

int ark_mpz_size_from_struct_p(const __mpz_struct* s) {
    return s ? s->_mp_size : 0;
}
