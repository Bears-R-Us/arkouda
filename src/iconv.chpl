// Generated with c2chapel version 0.1.0

// Header given to c2chapel:
require "iconv.h";

// Note: Generated with fake std headers

use CTypes;
extern var _libiconv_version : c_int;

extern proc libiconv_open(tocode : c_string, fromcode : c_string) : libiconv_t;

extern proc libiconv(cd : libiconv_t, ref inbuf : c_string, ref inbytesleft : c_size_t, ref outbuf : c_string, ref outbytesleft : c_size_t) : c_size_t;

extern proc libiconv(cd : libiconv_t, inbuf : c_ptr(c_string), inbytesleft : c_ptr(c_size_t), outbuf : c_ptr(c_string), outbytesleft : c_ptr(c_size_t)) : c_size_t;

extern proc libiconv_close(cd : libiconv_t) : c_int;

extern proc libiconv_open_into(tocode : c_string, fromcode : c_string, ref resultp : iconv_allocation_t) : c_int;

extern proc libiconv_open_into(tocode : c_string, fromcode : c_string, resultp : c_ptr(iconv_allocation_t)) : c_int;

extern proc libiconvctl(cd : libiconv_t, request : c_int, argument : c_void_ptr) : c_int;

extern "struct iconv_hooks" record iconv_hooks {
  var uc_hook : iconv_unicode_char_hook;
  var wc_hook : iconv_wide_char_hook;
  var data : c_void_ptr;
}

extern "struct iconv_fallbacks" record iconv_fallbacks {
  var mb_to_uc_fallback : iconv_unicode_mb_to_uc_fallback;
  var uc_to_mb_fallback : iconv_unicode_uc_to_mb_fallback;
  var mb_to_wc_fallback : iconv_wchar_mb_to_wc_fallback;
  var wc_to_mb_fallback : iconv_wchar_wc_to_mb_fallback;
  var data : c_void_ptr;
}

extern proc libiconvlist(do_one : c_fn_ptr, data : c_void_ptr) : void;

extern proc iconv_canonicalize(name : c_string) : c_string;

extern proc libiconv_set_relocation_prefix(orig_prefix : c_string, curr_prefix : c_string) : void;

// ==== c2chapel typedefs ====

extern record iconv_allocation_t {
  var dummy1 : c_ptr(c_void_ptr);
  var dummy2 : mbstate_t;
}

extern type iconv_unicode_char_hook = c_fn_ptr;

extern type iconv_unicode_mb_to_uc_fallback = c_fn_ptr;

extern type iconv_unicode_uc_to_mb_fallback = c_fn_ptr;

extern type iconv_wchar_mb_to_wc_fallback = c_fn_ptr;

extern type iconv_wchar_wc_to_mb_fallback = c_fn_ptr;

extern type iconv_wide_char_hook = c_fn_ptr;

extern type libiconv_t = c_void_ptr;

extern type mbstate_t = c_int;

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
