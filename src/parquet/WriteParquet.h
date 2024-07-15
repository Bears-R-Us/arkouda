#include <stdint.h>
#include <stdbool.h>

// Wrap functions in C extern if compiling C++ object file
#ifdef __cplusplus
#include <iostream>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/column_reader.h>
#include <parquet/api/writer.h>
#include <parquet/schema.h>
#include <cmath>
#include <queue>
extern "C" {
#endif

  int cpp_writeColumnToParquet(const char* filename, void* chpl_arr,
                               int64_t colnum, const char* dsetname, int64_t numelems,
                               int64_t rowGroupSize, int64_t dtype, int64_t compression,
                               char** errMsg);
  int c_writeColumnToParquet(const char* filename, void* chpl_arr,
                             int64_t colnum, const char* dsetname, int64_t numelems,
                             int64_t rowGroupSize, int64_t dtype, int64_t compression, char** errMsg);

  int c_writeStrColumnToParquet(const char* filename, void* chpl_arr, void* chpl_offsets,
                                const char* dsetname, int64_t numelems,
                                int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                char** errMsg);
  int cpp_writeStrColumnToParquet(const char* filename, void* chpl_arr, void* chpl_offsets,
                                  const char* dsetname, int64_t numelems,
                                  int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                  char** errMsg);
  
  int c_writeMultiColToParquet(const char* filename, void* column_names, 
                                void** ptr_arr, void** offset_arr, void* objTypes, void* datatypes,
                                void* segArr_sizes, int64_t colnum, int64_t numelems, int64_t rowGroupSize,
                                int64_t compression, char** errMsg);

  int cpp_writeMultiColToParquet(const char* filename, void* column_names, 
                                  void** ptr_arr, void** offset_arr, void* objTypes, void* datatypes,
                                  void* segArr_sizes, int64_t colnum, int64_t numelems, int64_t rowGroupSize,
                                  int64_t compression, char** errMsg);

  int c_writeStrListColumnToParquet(const char* filename, void* chpl_segs, void* chpl_offsets, 
                                    void* chpl_arr, const char* dsetname, int64_t numelems,
                                    int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                    char** errMsg);
  int cpp_writeStrListColumnToParquet(const char* filename, void* chpl_segs, void* chpl_offsets, 
                                      void* chpl_arr, const char* dsetname, int64_t numelems,
                                      int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                      char** errMsg);
  
#ifdef __cplusplus
}
#endif
