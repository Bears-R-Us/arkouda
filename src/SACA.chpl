module SACA{
// In this module, different algorithms to construct suffix array are provided
//Nov.15, 2020

// The first algorithm is divsufsort which is the fastest sequential and OpenMP c codes on suffix array
require "../../../SA/libdivsufsort/include/config.h";
require "../../../SA/libdivsufsort/include/divsufsort.h";
require "../../../SA/libdivsufsort/include/divsufsort_private.h";
require "../../../SA/libdivsufsort/include/lfs.h";

require "../../../SA/libdivsufsort/lib/divsufsort.c";
require "../../../SA/libdivsufsort/lib/sssort.c";
require "../../../SA/libdivsufsort/lib/trsort.c";
require "../../../SA/libdivsufsort/lib/utils.c";
/*
require "/home/z/zd4/SA/nong/saca-k-tois-20130413/saca-k/saca-k.cc";
*/
extern proc divsufsort(inputstr:[] uint(8),suffixarray:[] int(32),totallen:int(32));
}
