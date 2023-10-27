#include "read-parquet.h"

int readColumnByName(std::string filename, int col_num, int64_t* arr, int64_t numElems, int64_t batchSize) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile,
                          arrow::io::ReadableFile::Open(filename,
                                                        arrow::default_memory_pool()));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  std::shared_ptr<arrow::ChunkedArray> array;
  PARQUET_THROW_NOT_OK(reader->ReadColumn(col_num, &array));
  return 0;
}

void readColumns(std::string filename, std::string colname, int num_cols, int64_t* arr, int64_t numElems, int64_t batchSize) {
  for(int i = 0; i < num_cols; i++) {
    readColumnByName(filename, i, arr, numElems, batchSize);
  }
}

int main(int argc, char** argv) {
  std::string filename = argv[1];
  std::string colname = argv[2];
  int batchSize = atoi(argv[3]);
  int num_cols = atoi(argv[4]);

  int64_t numElems = 1000000000;
  int64_t* arr = (int64_t*)malloc(numElems*sizeof(int64_t));

  std::cout << "Reading " << num_cols << " columns using standard API: ";
  auto start = std::chrono::high_resolution_clock::now();
  readColumns(filename, "col", num_cols, arr, numElems, batchSize);
  auto finish = std::chrono::high_resolution_clock::now();
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);
  std::cout << milliseconds.count()/1000.0 << "s\n";
  
  return 0;
}

