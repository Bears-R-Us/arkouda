#include "read-parquet.h"

int getColIdx(std::string filename, std::string colname) {
  auto infile = arrow::io::ReadableFile::Open(filename,arrow::default_memory_pool()).ValueOrDie();

  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto st = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);

  std::shared_ptr<arrow::Schema> sc;
  std::shared_ptr<arrow::Schema>* out = &sc;
  st = reader->GetSchema(out);

  return sc -> GetFieldIndex(colname);
}

int readColumnByName(std::string filename, int col_num) {
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

void readColumns(std::string filename, std::string colname, int num_cols) {
  for(int i = 1; i <= num_cols; i++) {
    std::string col = colname;
    col.append(std::to_string(i));
    readColumnByName(filename, getColIdx(filename, col));
  }
}

int main(int argc, char** argv) {
  std::string filename = argv[1];
  std::string colname = argv[2];
  int num_cols = atoi(argv[3]);

  std::cout << "Reading " << num_cols << " columns using standard API: ";
  auto start = std::chrono::high_resolution_clock::now();
  readColumns(filename, colname, num_cols);
  auto finish = std::chrono::high_resolution_clock::now();
  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);
  std::cout << milliseconds.count()/1000.0 << "s\n";
  
  return 0;
}

