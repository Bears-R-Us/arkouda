// Generated with c2chapel version 0.1.0

// Header given to c2chapel:
require "idn2.h";

// Note: Generated with fake std headers

use CTypes;
extern proc idn2_lookup_u8(ref src : uint(8), ref lookupname : c_ptr(uint(8)), flags : c_int) : c_int;

extern proc idn2_lookup_u8(src : c_ptr(uint(8)), lookupname : c_ptr(c_ptr(uint(8))), flags : c_int) : c_int;

extern proc idn2_register_u8(ref ulabel : uint(8), ref alabel : uint(8), ref insertname : c_ptr(uint(8)), flags : c_int) : c_int;

extern proc idn2_register_u8(ulabel : c_ptr(uint(8)), alabel : c_ptr(uint(8)), insertname : c_ptr(c_ptr(uint(8))), flags : c_int) : c_int;

extern proc idn2_lookup_ul(src : c_string, ref lookupname : c_string, flags : c_int) : c_int;

extern proc idn2_lookup_ul(src : c_string, lookupname : c_ptr(c_string), flags : c_int) : c_int;

extern proc idn2_register_ul(ulabel : c_string, alabel : c_string, ref insertname : c_string, flags : c_int) : c_int;

extern proc idn2_register_ul(ulabel : c_string, alabel : c_string, insertname : c_ptr(c_string), flags : c_int) : c_int;

extern proc idn2_to_ascii_4i(ref input : uint(32), inlen : c_size_t, output : c_string, flags : c_int) : c_int;

extern proc idn2_to_ascii_4i(input : c_ptr(uint(32)), inlen : c_size_t, output : c_string, flags : c_int) : c_int;

extern proc idn2_to_ascii_4i2(ref input : uint(32), inlen : c_size_t, ref output : c_string, flags : c_int) : c_int;

extern proc idn2_to_ascii_4i2(input : c_ptr(uint(32)), inlen : c_size_t, output : c_ptr(c_string), flags : c_int) : c_int;

extern proc idn2_to_ascii_4z(ref input : uint(32), ref output : c_string, flags : c_int) : c_int;

extern proc idn2_to_ascii_4z(input : c_ptr(uint(32)), output : c_ptr(c_string), flags : c_int) : c_int;

extern proc idn2_to_ascii_8z(input : c_string, ref output : c_string, flags : c_int) : c_int;

extern proc idn2_to_ascii_8z(input : c_string, output : c_ptr(c_string), flags : c_int) : c_int;

extern proc idn2_to_ascii_lz(input : c_string, ref output : c_string, flags : c_int) : c_int;

extern proc idn2_to_ascii_lz(input : c_string, output : c_ptr(c_string), flags : c_int) : c_int;

extern proc idn2_to_unicode_8z4z(input : c_string, ref output : c_ptr(uint(32)), flags : c_int) : c_int;

extern proc idn2_to_unicode_8z4z(input : c_string, output : c_ptr(c_ptr(uint(32))), flags : c_int) : c_int;

extern proc idn2_to_unicode_4z4z(ref input : uint(32), ref output : c_ptr(uint(32)), flags : c_int) : c_int;

extern proc idn2_to_unicode_4z4z(input : c_ptr(uint(32)), output : c_ptr(c_ptr(uint(32))), flags : c_int) : c_int;

extern proc idn2_to_unicode_44i(ref in_arg : uint(32), inlen : c_size_t, ref out_arg : uint(32), ref outlen : c_size_t, flags : c_int) : c_int;

extern proc idn2_to_unicode_44i(in_arg : c_ptr(uint(32)), inlen : c_size_t, out_arg : c_ptr(uint(32)), outlen : c_ptr(c_size_t), flags : c_int) : c_int;

extern proc idn2_to_unicode_8z8z(input : c_string, ref output : c_string, flags : c_int) : c_int;

extern proc idn2_to_unicode_8z8z(input : c_string, output : c_ptr(c_string), flags : c_int) : c_int;

extern proc idn2_to_unicode_8zlz(input : c_string, ref output : c_string, flags : c_int) : c_int;

extern proc idn2_to_unicode_8zlz(input : c_string, output : c_ptr(c_string), flags : c_int) : c_int;

extern proc idn2_to_unicode_lzlz(input : c_string, ref output : c_string, flags : c_int) : c_int;

extern proc idn2_to_unicode_lzlz(input : c_string, output : c_ptr(c_string), flags : c_int) : c_int;

extern proc idn2_strerror(rc : c_int) : c_string;

extern proc idn2_strerror_name(rc : c_int) : c_string;

extern proc idn2_check_version(req_version : c_string) : c_string;

extern proc idn2_free(ptr : c_void_ptr) : void;

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


// c2chapel thinks these typedefs are from the fake headers:
/*
extern type FILE = c_int;

// Opaque struct?
extern record MirBlob {};

// Opaque struct?
extern record MirBufferStream {};

// Opaque struct?
extern record MirConnection {};

// Opaque struct?
extern record MirDisplayConfig {};

extern type MirEGLNativeDisplayType = c_void_ptr;

extern type MirEGLNativeWindowType = c_void_ptr;

// Opaque struct?
extern record MirPersistentId {};

// Opaque struct?
extern record MirPromptSession {};

// Opaque struct?
extern record MirScreencast {};

// Opaque struct?
extern record MirSurface {};

// Opaque struct?
extern record MirSurfaceSpec {};

extern type _LOCK_RECURSIVE_T = c_int;

extern type _LOCK_T = c_int;

extern type __FILE = c_int;

extern type __ULong = c_int;

extern type __builtin_va_list = c_int;

extern type __dev_t = c_int;

extern type __gid_t = c_int;

extern type __gnuc_va_list = c_int;

extern type __int16_t = c_int;

extern type __int32_t = c_int;

extern type __int64_t = c_int;

extern type __int8_t = c_int;

extern type __int_least16_t = c_int;

extern type __int_least32_t = c_int;

extern type __loff_t = c_int;

extern type __off_t = c_int;

extern type __pid_t = c_int;

extern type __s16 = c_int;

extern type __s32 = c_int;

extern type __s64 = c_int;

extern type __s8 = c_int;

extern type __sigset_t = c_int;

extern type __tzinfo_type = c_int;

extern type __tzrule_type = c_int;

extern type __u16 = c_int;

extern type __u32 = c_int;

extern type __u64 = c_int;

extern type __u8 = c_int;

extern type __uid_t = c_int;

extern type __uint16_t = c_int;

extern type __uint32_t = c_int;

extern type __uint64_t = c_int;

extern type __uint8_t = c_int;

extern type __uint_least16_t = c_int;

extern type __uint_least32_t = c_int;

extern type _flock_t = c_int;

extern type _fpos_t = c_int;

extern type _iconv_t = c_int;

extern type _mbstate_t = c_int;

extern type _off64_t = c_int;

extern type _off_t = c_int;

extern type _sig_func_ptr = c_int;

extern type _ssize_t = c_int;

extern type _types_fd_set = c_int;

extern type bool = _Bool;

extern type caddr_t = c_int;

extern type clock_t = c_int;

extern type clockid_t = c_int;

extern type cookie_close_function_t = c_int;

extern type cookie_io_functions_t = c_int;

extern type cookie_read_function_t = c_int;

extern type cookie_seek_function_t = c_int;

extern type cookie_write_function_t = c_int;

extern type daddr_t = c_int;

extern type dev_t = c_int;

extern type div_t = c_int;

extern type fd_mask = c_int;

extern type fpos_t = c_int;

extern type gid_t = c_int;

extern type ino_t = c_int;

extern type int16_t = c_int;

extern type int32_t = c_int;

extern type int64_t = c_int;

extern type int8_t = c_int;

extern type int_fast16_t = c_int;

extern type int_fast32_t = c_int;

extern type int_fast64_t = c_int;

extern type int_fast8_t = c_int;

extern type int_least16_t = c_int;

extern type int_least32_t = c_int;

extern type int_least64_t = c_int;

extern type int_least8_t = c_int;

extern type intmax_t = c_int;

extern type intptr_t = c_int;

extern type jmp_buf = c_int;

extern type key_t = c_int;

extern type ldiv_t = c_int;

extern type lldiv_t = c_int;

extern type mbstate_t = c_int;

extern type mode_t = c_int;

extern type nlink_t = c_int;

extern type off_t = c_int;

extern type pid_t = c_int;

extern type pthread_attr_t = c_int;

extern type pthread_barrier_t = c_int;

extern type pthread_barrierattr_t = c_int;

extern type pthread_cond_t = c_int;

extern type pthread_condattr_t = c_int;

extern type pthread_key_t = c_int;

extern type pthread_mutex_t = c_int;

extern type pthread_mutexattr_t = c_int;

extern type pthread_once_t = c_int;

extern type pthread_rwlock_t = c_int;

extern type pthread_rwlockattr_t = c_int;

extern type pthread_spinlock_t = c_int;

extern type pthread_t = c_int;

extern type ptrdiff_t = c_int;

extern type rlim_t = c_int;

extern type sa_family_t = c_int;

extern type sem_t = c_int;

extern type sig_atomic_t = c_int;

extern type siginfo_t = c_int;

extern type sigjmp_buf = c_int;

extern type sigset_t = c_int;

extern type size_t = c_int;

extern type ssize_t = c_int;

extern type stack_t = c_int;

extern type suseconds_t = c_int;

extern type time_t = c_int;

extern type timer_t = c_int;

extern type u_char = c_int;

extern type u_int = c_int;

extern type u_long = c_int;

extern type u_short = c_int;

extern type uid_t = c_int;

extern type uint = c_int;

extern type uint16_t = c_int;

extern type uint32_t = c_int;

extern type uint64_t = c_int;

extern type uint8_t = c_int;

extern type uint_fast16_t = c_int;

extern type uint_fast32_t = c_int;

extern type uint_fast64_t = c_int;

extern type uint_fast8_t = c_int;

extern type uint_least16_t = c_int;

extern type uint_least32_t = c_int;

extern type uint_least64_t = c_int;

extern type uint_least8_t = c_int;

extern type uintmax_t = c_int;

extern type uintptr_t = c_int;

extern type useconds_t = c_int;

extern type ushort = c_int;

extern type va_list = c_int;

extern type wchar_t = c_int;

extern type wint_t = c_int;

// Opaque struct?
extern record xcb_connection_t {};

extern type xcb_visualid_t = uint(32);

extern type xcb_window_t = uint(32);

extern type z_stream = c_int;

*/
