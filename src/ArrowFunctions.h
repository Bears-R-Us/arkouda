// Wrap functions in C extern if compiling C++ object file
#ifdef __cplusplus
extern "C" {
#endif

  #define ARROWINT64 0
  #define ARROWINT32 1
  #define ARROWUNDEFINED -1
  #define ARROWERROR -1

  // Each C++ function contains the actual implementation of the
  // functionality, and there is a corresponding C function that
  // Chapel can call into through C interoperability, since there
  // is no C++ interoperability supported in Chapel today.
  int c_getNumRows(const char*, char** errMsg);
  int cpp_getNumRows(const char*, char** errMsg);

  int c_readColumnByName(const char* filename, void* chpl_arr,
                         const char* colname, int numElems,
                         char** errMsg);
  int cpp_readColumnByName(const char* filename, void* chpl_arr,
                           const char* colname, int numElems,
                           char** errMsg);

  int c_getType(const char* filename, const char* colname, char** errMsg);
  int cpp_getType(const char* filename, const char* colname, char** errMsg);

  int cpp_writeColumnToParquet(const char* filename, void* chpl_arr,
                               int colnum, const char* dsetname, int numelems,
                               int rowGroupSize, char** errMsg);
  int c_writeColumnToParquet(const char* filename, void* chpl_arr,
                             int colnum, const char* dsetname, int numelems,
                             int rowGroupSize, char** errMsg);
    
  const char* c_getVersionInfo(void);
  const char* cpp_getVersionInfo(void);

  void c_free_string(void* ptr);
  void cpp_free_string(void* ptr);
  
#ifdef __cplusplus
}
#endif
