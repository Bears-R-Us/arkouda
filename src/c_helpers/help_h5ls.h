/**
 * Function prototypes for HDF5 helper functions to simulate `h5ls`
 * These are helper functions to process opaque data objects passed
 * via void* / c_void_ptr.
 * See GenSymIO.simulate_h5ls
 */

#ifndef _AK_H5LS_HELPER_H_
#define _AK_H5LS_HELPER_H_

#include "hdf5.h"
#include <string.h>

/* C function to retrieve the HDF5 object type for a given object name */
herr_t c_get_HDF5_obj_type (hid_t loc_id, const char *name, H5O_type_t *obj_type);

/* C helper function to increment a counter passed via `void*` */
void c_incrementCounter(void *data);


/* C helper function to wrap `strlen` */
size_t c_strlen(char* s);

/* C helper function to append HDF5 fieldnames to a char* passed as `void*` */
void c_append_HDF5_fieldname(void *data, const char *name);

#endif