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
#include <parquet/types.h>
#include <cmath>
#include <queue>
extern "C" {
#endif

  typedef struct {
    uint32_t len;
    const uint8_t* ptr;
  } MyByteArray;

  void* c_getByteArray(void);
  void* cpp_getByteArray(void);
  void c_freeByteArray(void* input);

#ifdef __cplusplus
}
#endif
