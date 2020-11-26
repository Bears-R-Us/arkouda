module SACA{
// In this module, different algorithms to construct suffix array are provided
//Nov.15, 2020

// The first algorithm divsufsort is the fastest C codes on suffix array
require "../thirdparty/SA/libdivsufsort/include/config.h";
require "../thirdparty/SA/libdivsufsort/include/divsufsort.h";
require "../thirdparty/SA/libdivsufsort/include/divsufsort_private.h";
require "../thirdparty/SA/libdivsufsort/include/lfs.h";

require "../thirdparty/SA/libdivsufsort/lib/divsufsort.c";
require "../thirdparty/SA/libdivsufsort/lib/sssort.c";
require "../thirdparty/SA/libdivsufsort/lib/trsort.c";
require "../thirdparty/SA/libdivsufsort/lib/utils.c";
/*
require "/home/z/zd4/SA/nong/saca-k-tois-20130413/saca-k/saca-k.cc";
*/
extern proc divsufsort(inputstr:[] uint(8),suffixarray:[] int(32),totallen:int(32));
}
