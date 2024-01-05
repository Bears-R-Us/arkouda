#include "test.h"

void* c_getByteArray() {
  return cpp_getByteArray();
}

void* cpp_getByteArray() {
  parquet::ByteArray* string_values =
    (parquet::ByteArray*)malloc(10*sizeof(parquet::ByteArray));
  string_values[0] = parquet::ByteArray("asd0");
  string_values[1] = parquet::ByteArray("asd1");
  string_values[2] = parquet::ByteArray("asd2");
  string_values[3] = parquet::ByteArray("asd3");
  string_values[4] = parquet::ByteArray("asd4");
  string_values[5] = parquet::ByteArray("asd5");
  string_values[6] = parquet::ByteArray("asd6");
  string_values[7] = parquet::ByteArray("asd7");
  string_values[8] = parquet::ByteArray("asd8");
  string_values[9] = parquet::ByteArray("asd9");

  return (void*)string_values;
}

void c_freeByteArray(void* input) {
  free(input);
}
