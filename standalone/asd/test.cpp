#include "test.h"

void cpp_getParquetReader(void* reader, const char* filename, const char* colname) {
  std::unique_ptr<parquet::ParquetFileReader> parquet_reader =
    parquet::ParquetFileReader::OpenFile(filename, false);

  std::shared_ptr<parquet::FileMetaData> file_metadata = parquet_reader->metadata();

  int64_t i = 0;
  std::shared_ptr<parquet::RowGroupReader> row_group_reader =
    parquet_reader->RowGroup(0);

  std::shared_ptr<parquet::ColumnReader> column_reader;

  auto idx = file_metadata -> schema() -> ColumnIndex(colname);

  column_reader = row_group_reader->Column(idx);
  
  reader = static_cast<void*>(column_reader.get());
}

void* cpp_readParquetColumn(void* reader, void* chpl_arr, int64_t batchSize, int64_t numElems) {
  std::cout << "top" << std::endl;
  auto ba_reader = static_cast<parquet::ByteArrayReader*>(reader);
  std::cout << "cast" << std::endl;
  parquet::ByteArray* string_values =
    (parquet::ByteArray*)malloc(numElems*sizeof(parquet::ByteArray));
  std::cout << "malloc" << std::endl;
  std::vector<int16_t> definition_level(numElems);
  int64_t values_read = 0;
  (void)ba_reader_cpp->ReadBatch(1, definition_level.data(), nullptr, string_values, &values_read);
  std::cout << "read" << std::endl;
  std::cout << values_read << std::endl;
  return (void*)string_values;
}

void c_getParquetReader(void* reader, const char* filename, const char* colname) {
  return cpp_getParquetReader(reader, filename, colname);
}

void* c_readParquetColumn(void* reader, void* chpl_arr, int64_t batchSize, int64_t numElems) {
  return cpp_readParquetColumn(reader, chpl_arr, batchSize, numElems);
}

void c_freeByteArray(void* input) {
  free(input);
}
