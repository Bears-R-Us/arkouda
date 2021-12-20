/**
 * External C functions for simulating h5ls and processing HDF5 API objects/data.
 * HDF5 API passes void* data objects in between calls which can't be processed
 * directly in chapel, so you need C functions to handle opaque data objects.
 */
#include "c_helpers/help_h5ls.h"

/**
 * C function to retrieve the HDF5 object type for a given object name
 */
herr_t c_get_HDF5_obj_type(hid_t loc_id, const char *name, H5O_type_t *obj_type)
{
    herr_t status;
    H5O_info_t info_t;
    H5O_info_t* info_t_ptr = &info_t;
    status = H5Oget_info_by_name(loc_id, name, info_t_ptr, H5P_DEFAULT);
    *obj_type = info_t.type;
    return status;
}

/**
 * C helper function to increment a counter passed via void*
 */
void c_incrementCounter(void *data)
{
    int i = *(int *)data;
    i = i + 1;
    *(int *)data = i;
}

/**
 * C helper function to wrap `strlen`
 */
size_t c_strlen(char *s)
{
    return strlen(s);
}

/**
 * C helper function to append HDF5 fieldnames to a char* passed as void*
 */
void c_append_HDF5_fieldname(void *data, const char *name)
{
    char *d = (char *)data; // Turn void* data into char*
    if (strlen(d) > 0)
    {
        strcat(d, ",");
    }
    strcat(d, name);
}
