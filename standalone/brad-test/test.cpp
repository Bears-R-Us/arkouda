#include "test.h"

/*
int main() {
  const char* filename = "test-file_LOCALE0000";
  const char* colname = "strings_array";
  int64_t batchSize = 116510;
  int64_t numElems = 100000000;

  int idx = 0;
  
  cpp_openFile(filename, idx);
  cpp_createRowGroupReader(0, idx);
  cpp_createColumnReader(colname, idx);
  
  void* ret = cpp_readParquetColumnChunks(filename, batchSize, numElems, idx);
  parquet::ByteArray* asd = (parquet::ByteArray*)ret;
  for(int i = 0; i < 10; i++) {
    printf("%.*s\n", asd[i*100].len, asd[i*100].ptr);
  }
  
  return 0;
  }*/

void cpp_openFile(const char* filename, int64_t idx) {
  std::shared_ptr<parquet::ParquetFileReader> parquet_reader =
    parquet::ParquetFileReader::OpenFile(filename, false);
  globalFiles[idx] = parquet_reader;
}

void cpp_createRowGroupReader(int64_t rowGroup, int64_t readerIdx) {
  std::shared_ptr<parquet::RowGroupReader> row_group_reader =
    globalFiles[readerIdx]->RowGroup(rowGroup);
  globalRowGroupReaders[readerIdx] = row_group_reader;
}

void cpp_createColumnReader(const char* colname, int64_t readerIdx) {
  std::shared_ptr<parquet::FileMetaData> file_metadata = globalFiles[readerIdx]->metadata();
  auto idx = file_metadata -> schema() -> ColumnIndex(colname);
  
  std::shared_ptr<parquet::ColumnReader> column_reader;
  column_reader = globalRowGroupReaders[readerIdx]->Column(idx);
  globalColumnReaders[readerIdx] = column_reader;
}

void* cpp_readParquetColumnChunks(const char* filename, int64_t batchSize, int64_t numElems, int64_t readerIdx, int64_t* numRead) { 
  auto reader = static_cast<parquet::ByteArrayReader*>(globalColumnReaders[readerIdx].get());
  parquet::ByteArray* string_values =
    (parquet::ByteArray*)malloc(numElems*sizeof(parquet::ByteArray));
  std::vector<int16_t> definition_level(numElems);
  int64_t values_read = 0;
  int64_t total_read = 0;
  while(reader->HasNext() && total_read < numElems) {
    if((numElems - total_read) < batchSize)
      batchSize = numElems - total_read;
    (void)reader->ReadBatch(batchSize, definition_level.data(), nullptr, string_values + total_read, &values_read);
    total_read += values_read;
  }
  std::cout << total_read << std::endl;
  *numRead = total_read;
  return (void*)string_values;
}

int cpp_getNumRowGroups(int64_t readerIdx) {
  std::shared_ptr<parquet::FileMetaData> file_metadata = globalFiles[readerIdx]->metadata();
  return file_metadata->num_row_groups();
}

void c_openFile(const char* filename, int64_t idx) {
  cpp_openFile(filename, idx);
}

void c_createRowGroupReader(int64_t rowGroup, int64_t readerIdx) {
  return cpp_createRowGroupReader(rowGroup, readerIdx);
}

void c_createColumnReader(const char* colname, int64_t readerIdx) {
  cpp_createColumnReader(colname, readerIdx);
}

void* c_readParquetColumnChunks(const char* filename, int64_t batchSize, int64_t numElems, int64_t readerIdx, int64_t* numRead) {
  return cpp_readParquetColumnChunks(filename, batchSize, numElems, readerIdx, numRead);
}

int c_getNumRowGroups(int64_t readerIdx) {
  return cpp_getNumRowGroups(readerIdx);
}
