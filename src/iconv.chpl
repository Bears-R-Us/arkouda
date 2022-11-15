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
