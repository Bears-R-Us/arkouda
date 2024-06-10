// Generated with c2chapel version 0.1.0

// Header given to c2chapel:
require "idn2.h";

// Note: Generated with fake std headers

use CTypes;
extern proc idn2_lookup_u8(ref src : uint(8), ref lookupname : c_ptr(uint(8)), flags : c_int) : c_int;

extern proc idn2_lookup_u8(src : c_ptr(uint(8)), lookupname : c_ptr(c_ptr(uint(8))), flags : c_int) : c_int;

extern proc idn2_register_u8(ref ulabel : uint(8), ref alabel : uint(8), ref insertname : c_ptr(uint(8)), flags : c_int) : c_int;

extern proc idn2_register_u8(ulabel : c_ptr(uint(8)), alabel : c_ptr(uint(8)), insertname : c_ptr(c_ptr(uint(8))), flags : c_int) : c_int;

extern proc idn2_lookup_ul(src : c_ptrConst(c_char), ref lookupname : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_lookup_ul(src : c_ptrConst(c_char), lookupname : c_ptr(c_ptrConst(c_char)), flags : c_int) : c_int;

extern proc idn2_register_ul(ulabel : c_ptrConst(c_char), alabel : c_ptrConst(c_char), ref insertname : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_register_ul(ulabel : c_ptrConst(c_char), alabel : c_ptrConst(c_char), insertname : c_ptr(c_ptrConst(c_char)), flags : c_int) : c_int;

extern proc idn2_to_ascii_4i(ref input : uint(32), inlen : c_size_t, output : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_to_ascii_4i(input : c_ptr(uint(32)), inlen : c_size_t, output : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_to_ascii_4i2(ref input : uint(32), inlen : c_size_t, ref output : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_to_ascii_4i2(input : c_ptr(uint(32)), inlen : c_size_t, output : c_ptr(c_ptrConst(c_char)), flags : c_int) : c_int;

extern proc idn2_to_ascii_4z(ref input : uint(32), ref output : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_to_ascii_4z(input : c_ptr(uint(32)), output : c_ptr(c_ptrConst(c_char)), flags : c_int) : c_int;

extern proc idn2_to_ascii_8z(input : c_ptrConst(c_char), ref output : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_to_ascii_8z(input : c_ptrConst(c_char), output : c_ptr(c_ptrConst(c_char)), flags : c_int) : c_int;

extern proc idn2_to_ascii_lz(input : c_ptrConst(c_char), ref output : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_to_ascii_lz(input : c_ptrConst(c_char), output : c_ptr(c_ptrConst(c_char)), flags : c_int) : c_int;

extern proc idn2_to_unicode_8z4z(input : c_ptrConst(c_char), ref output : c_ptr(uint(32)), flags : c_int) : c_int;

extern proc idn2_to_unicode_8z4z(input : c_ptrConst(c_char), output : c_ptr(c_ptr(uint(32))), flags : c_int) : c_int;

extern proc idn2_to_unicode_4z4z(ref input : uint(32), ref output : c_ptr(uint(32)), flags : c_int) : c_int;

extern proc idn2_to_unicode_4z4z(input : c_ptr(uint(32)), output : c_ptr(c_ptr(uint(32))), flags : c_int) : c_int;

extern proc idn2_to_unicode_44i(ref in_arg : uint(32), inlen : c_size_t, ref out_arg : uint(32), ref outlen : c_size_t, flags : c_int) : c_int;

extern proc idn2_to_unicode_44i(in_arg : c_ptr(uint(32)), inlen : c_size_t, out_arg : c_ptr(uint(32)), outlen : c_ptr(c_size_t), flags : c_int) : c_int;

extern proc idn2_to_unicode_8z8z(input : c_ptrConst(c_char), ref output : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_to_unicode_8z8z(input : c_ptrConst(c_char), output : c_ptr(c_ptrConst(c_char)), flags : c_int) : c_int;

extern proc idn2_to_unicode_8zlz(input : c_ptrConst(c_char), ref output : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_to_unicode_8zlz(input : c_ptrConst(c_char), output : c_ptr(c_ptrConst(c_char)), flags : c_int) : c_int;

extern proc idn2_to_unicode_lzlz(input : c_ptrConst(c_char), ref output : c_ptrConst(c_char), flags : c_int) : c_int;

extern proc idn2_to_unicode_lzlz(input : c_ptrConst(c_char), output : c_ptr(c_ptrConst(c_char)), flags : c_int) : c_int;

extern proc idn2_strerror(rc : c_int) : c_ptrConst(c_char);

extern proc idn2_strerror_name(rc : c_int) : c_ptrConst(c_char);

extern proc idn2_check_version(req_version : c_ptrConst(c_char)) : c_ptrConst(c_char);

extern proc idn2_free(ptr : c_ptr(void)) : void;

// ==== c2chapel typedefs ====

// Idna_flags enum
extern type Idna_flags = c_int;
extern const IDNA_ALLOW_UNASSIGNED :Idna_flags;
extern const IDNA_USE_STD3_ASCII_RULES :Idna_flags;


// Idna_rc enum
extern type Idna_rc = c_int;
extern const IDNA_SUCCESS :Idna_rc;
extern const IDNA_STRINGPREP_ERROR :Idna_rc;
extern const IDNA_PUNYCODE_ERROR :Idna_rc;
extern const IDNA_CONTAINS_NON_LDH :Idna_rc;
extern const IDNA_CONTAINS_LDH :Idna_rc;
extern const IDNA_CONTAINS_MINUS :Idna_rc;
extern const IDNA_INVALID_LENGTH :Idna_rc;
extern const IDNA_NO_ACE_PREFIX :Idna_rc;
extern const IDNA_ROUNDTRIP_VERIFY_ERROR :Idna_rc;
extern const IDNA_CONTAINS_ACE_PREFIX :Idna_rc;
extern const IDNA_ICONV_ERROR :Idna_rc;
extern const IDNA_MALLOC_ERROR :Idna_rc;
extern const IDNA_DLOPEN_ERROR :Idna_rc;


// idn2_flags enum
extern type idn2_flags = c_int;
extern const IDN2_NFC_INPUT :idn2_flags;
extern const IDN2_ALABEL_ROUNDTRIP :idn2_flags;
extern const IDN2_TRANSITIONAL :idn2_flags;
extern const IDN2_NONTRANSITIONAL :idn2_flags;
extern const IDN2_ALLOW_UNASSIGNED :idn2_flags;
extern const IDN2_USE_STD3_ASCII_RULES :idn2_flags;
extern const IDN2_NO_TR46 :idn2_flags;
extern const IDN2_NO_ALABEL_ROUNDTRIP :idn2_flags;


// idn2_rc enum
extern type idn2_rc = c_int;
extern const IDN2_OK :idn2_rc;
extern const IDN2_MALLOC :idn2_rc;
extern const IDN2_NO_CODESET :idn2_rc;
extern const IDN2_ICONV_FAIL :idn2_rc;
extern const IDN2_ENCODING_ERROR :idn2_rc;
extern const IDN2_NFC :idn2_rc;
extern const IDN2_PUNYCODE_BAD_INPUT :idn2_rc;
extern const IDN2_PUNYCODE_BIG_OUTPUT :idn2_rc;
extern const IDN2_PUNYCODE_OVERFLOW :idn2_rc;
extern const IDN2_TOO_BIG_DOMAIN :idn2_rc;
extern const IDN2_TOO_BIG_LABEL :idn2_rc;
extern const IDN2_INVALID_ALABEL :idn2_rc;
extern const IDN2_UALABEL_MISMATCH :idn2_rc;
extern const IDN2_INVALID_FLAGS :idn2_rc;
extern const IDN2_NOT_NFC :idn2_rc;
extern const IDN2_2HYPHEN :idn2_rc;
extern const IDN2_HYPHEN_STARTEND :idn2_rc;
extern const IDN2_LEADING_COMBINING :idn2_rc;
extern const IDN2_DISALLOWED :idn2_rc;
extern const IDN2_CONTEXTJ :idn2_rc;
extern const IDN2_CONTEXTJ_NO_RULE :idn2_rc;
extern const IDN2_CONTEXTO :idn2_rc;
extern const IDN2_CONTEXTO_NO_RULE :idn2_rc;
extern const IDN2_UNASSIGNED :idn2_rc;
extern const IDN2_BIDI :idn2_rc;
extern const IDN2_DOT_IN_LABEL :idn2_rc;
extern const IDN2_INVALID_TRANSITIONAL :idn2_rc;
extern const IDN2_INVALID_NONTRANSITIONAL :idn2_rc;
extern const IDN2_ALABEL_ROUNDTRIP_FAILED :idn2_rc;
