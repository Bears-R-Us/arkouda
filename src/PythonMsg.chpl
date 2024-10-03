
module PythonMsg
{
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use NumPyDType;
    use ServerErrorStrings;

    use Map;
    //use ArkoudaIOCompat;
    //use ArkoudaAryUtilCompat;

    private config const logLevel = ServerConfig.logLevel;
    private config const logChannel = ServerConfig.logChannel;
    const sLogger = new Logger(logLevel, logChannel);

    extern {
        #if defined(linux)

        #include <stdio.h>

        #ifdef _POSIX_C_SOURCE
        #undef _POSIX_C_SOURCE
        #endif

        #ifdef _FILE_OFFSET_BITS
        #undef _FILE_OFFSET_BITS
        #endif

        #ifdef _XOPEN_SOURCE
        #undef _XOPEN_SOURCE
        #endif

        #ifdef _DARWIN_C_SOURCE
        #undef _DARWIN_C_SOURCE
        #endif

        #ifdef _NETBSD_SOURCE
        #undef _NETBSD_SOURCE
        #endif

        #endif // linux

        #define PY_SSIZE_T_CLEAN
        #include "Python.h"

        static PyObject* arkjit_compile = NULL;
        void initializePythonMsg(void);

        typedef void (*gen_disp_t)(void);
        typedef double (*disp_1d_t)(double);

        int dispIsValid(gen_disp_t);
        gen_disp_t dispatchPythonMsg(const char*, const char*, Py_ssize_t);

        double callDisp1D(disp_1d_t, double);

        void initializePythonMsg(void) {
            PyObject* dispmod = NULL;

            if (!Py_IsInitialized()) {
        #if PY_VERSION_HEX < 0x03020000
                PyEval_InitThreads();
        #endif
        #if PY_VERSION_HEX < 0x03080000
                Py_Initialize();
        #else
                PyConfig config;
                PyConfig_InitPythonConfig(&config);
                PyConfig_SetString(&config, &config.program_name, L"arkouda");
                Py_InitializeFromConfig(&config);
        #endif
        #if PY_VERSION_HEX >= 0x03020000
        #if PY_VERSION_HEX < 0x03090000
                PyEval_InitThreads();
        #endif
        #endif
            }

            dispmod = PyImport_ImportModule("arkouda.dispatch_utils");
            if (dispmod) {
                arkjit_compile = PyObject_GetAttrString(dispmod, "compile");
                Py_DECREF(dispmod);
            } else {
                PyErr_Print();
            }
        }

        gen_disp_t dispatchPythonMsg(const char* dt, const char* pkl, Py_ssize_t lpkl) {
            PyObject* result = NULL;

            if (arkjit_compile) {
                PyObject* pyfunc = PyBytes_FromStringAndSize(pkl, lpkl);
                result = PyObject_CallFunction(arkjit_compile, "Os", pyfunc, dt, NULL);
                Py_DECREF(pyfunc);

                if (!result)
                    return NULL;

                gen_disp_t fdisp = (gen_disp_t)PyLong_AsLongLong(result);

                Py_DECREF(result);
                return fdisp;
            }
            return NULL;
        }

        int dispIsValid(gen_disp_t fdisp) {
            return (int)(fdisp != NULL);
        }

        double callDisp1D(disp_1d_t fdisp, double d) {
            return fdisp(d);
        }

    }

    initializePythonMsg();

    @arkouda.registerND
    proc pythonMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab, param nd: int): MsgTuple throws {
        const aname = msgArgs.getValueOf("a");
        const pkl = msgArgs.getValueOf("pickle");
        const inplace: bool = msgArgs.get("inplace").getBoolValue();

        var arr: borrowed GenSymEntry = getGenericTypedArrayEntry(aname, st);
        var dt: string = dtype2str(arr.dtype);

        var pydisp = dispatchPythonMsg(dt.c_str(), pkl.c_str(), pkl.numBytes);

        if (!dispIsValid(pydisp)) {
            PyErr_Print();
            return new MsgTuple("failed to compile Python function", MsgType.ERROR);
        }

        select (arr.dtype) {
            when (DType.Float64) {
                var l = toSymEntry(arr, real, nd);
                if (inplace) {
                    forall a in l.a do
                        a = callDisp1D(pydisp, a);
                    return new MsgTuple("success", MsgType.NORMAL);
                } else {
                    var rname = st.nextName();
                    var e = st.addEntry(rname, l.tupShape, real);
                    e.a = forall a in l.a do
                              callDisp1D(pydisp, a);
                    var repMsg = "created %s".format(st.attrib(rname));
                    return new MsgTuple(repMsg, MsgType.NORMAL);
                }
            }
        }

        return new MsgTuple("unsupported array type", MsgType.ERROR);
    }

}
