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

std::shared_ptr<parquet::schema::GroupNode> SetupSchema(void* column_names, void * objTypes, void* datatypes, int64_t colnum);

extern "C" {
#endif

#define ARROWINT64 0
#define ARROWINT32 1
#define ARROWUINT64 2
#define ARROWUINT32 3
#define ARROWBOOLEAN 4
#define ARROWFLOAT 5
#define ARROWDOUBLE 7
#define ARROWTIMESTAMP ARROWINT64
#define ARROWSTRING 6
#define ARROWLIST 8
#define ARROWDECIMAL 9
#define ARROWERROR -1

#define ARRAYVIEW 0 // not currently used, but included for continuity with Chapel
#define PDARRAY 1
#define STRINGS 2
#define SEGARRAY 3

// compression mappings
#define SNAPPY_COMP 1
#define GZIP_COMP 2
#define BROTLI_COMP 3
#define ZSTD_COMP 4
#define LZ4_COMP 5

  typedef struct {
    uint32_t len;
    const uint8_t* ptr;
  } MyByteArray;

  void c_openFile(const char* filename, int64_t idx);
  void cpp_openFile(const char* filename, int64_t idx);

  void c_createRowGroupReader(int64_t rowGroup, int64_t readerIdx);
  void cpp_createRowGroupReader(int64_t rowGroup, int64_t readerIdx);

  void c_createColumnReader(const char* colname, int64_t readerIdx);
  void cpp_createColumnReader(const char* colname, int64_t readerIdx);

  int c_getNumRowGroups(int64_t readerIdx);
  int cpp_getNumRowGroups(int64_t readerIdx);
  
  // Each C++ function contains the actual implementation of the
  // functionality, and there is a corresponding C function that
  // Chapel can call into through C interoperability, since there
  // is no C++ interoperability supported in Chapel today.
  int64_t c_getNumRows(const char*, char** errMsg);
  int64_t cpp_getNumRows(const char*, char** errMsg);
  
  int64_t c_getStringColumnNullIndices(const char* filename, const char* colname, void* chpl_nulls, char** errMsg);
  int64_t cpp_getStringColumnNullIndices(const char* filename, const char* colname, void* chpl_nulls, char** errMsg);

  int c_getType(const char* filename, const char* colname, char** errMsg);
  int cpp_getType(const char* filename, const char* colname, char** errMsg);

  int c_getListType(const char* filename, const char* colname, char** errMsg);
  int cpp_getListType(const char* filename, const char* colname, char** errMsg);

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
  
  int c_createEmptyListParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                               int64_t compression, char** errMsg);
  int cpp_createEmptyListParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                               int64_t compression, char** errMsg);

  int c_writeListColumnToParquet(const char* filename, void* chpl_offsets, void* chpl_arr,
                                  const char* dsetname, int64_t numelems,
                                  int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                  char** errMsg);
  int cpp_writeListColumnToParquet(const char* filename, void* chpl_offsets, void* chpl_arr,
                                  const char* dsetname, int64_t numelems,
                                  int64_t rowGroupSize, int64_t dtype, int64_t compression,
                                  char** errMsg);
  
  int c_createEmptyParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                               int64_t compression, char** errMsg);
  int cpp_createEmptyParquetFile(const char* filename, const char* dsetname, int64_t dtype,
                                 int64_t compression, char** errMsg);
  
  int c_appendColumnToParquet(const char* filename, void* chpl_arr,
                              const char* dsetname, int64_t numelems,
                              int64_t dtype, int64_t compression,
                              char** errMsg);
  int cpp_appendColumnToParquet(const char* filename, void* chpl_arr,
                                const char* dsetname, int64_t numelems,
                                int64_t dtype, int64_t compression,
                                char** errMsg);

  int c_getPrecision(const char* filename, const char* colname, char** errMsg);
  int cpp_getPrecision(const char* filename, const char* colname, char** errMsg);
    
  const char* c_getVersionInfo(void);
  const char* cpp_getVersionInfo(void);

  int c_getDatasetNames(const char* filename, char** dsetResult, bool readNested, char** errMsg);
  int cpp_getDatasetNames(const char* filename, char** dsetResult, bool readNested, char** errMsg);

  void c_free_string(void* ptr);
  void cpp_free_string(void* ptr);
  
  void c_freeMapValues(void* row);
  void cpp_freeMapValues(void* row);

  int c_readParquetColumnChunks(const char* filename, int64_t batchSize, int64_t numElems,
                                int64_t readerIdx, int64_t* numRead,
                                void** outData, bool* containsNulls, char** errMsg);
  int cpp_readParquetColumnChunks(const char* filename, int64_t batchSize, int64_t numElems,
                                  int64_t readerIdx, int64_t* numRead,
                                  void** outData, bool* containsNulls, char** errMsg);
  
#ifdef __cplusplus
  bool check_status_ok(arrow::Status status, char** errMsg);
}
#endif
