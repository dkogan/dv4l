#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <stdbool.h>
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <assert.h>

#include "dv4l.h"

// Python is silly. There's some nuance about signal handling where it sets a
// SIGINT (ctrl-c) handler to just set a flag, and the python layer then reads
// this flag and does the thing. Here I'm running C code, so SIGINT would set a
// flag, but not quit, so I can't interrupt the solver. Thus I reset the SIGINT
// handler to the default, and put it back to the python-specific version when
// I'm done
#define SET_SIGINT() struct sigaction sigaction_old;                    \
do {                                                                    \
    if( 0 != sigaction(SIGINT,                                          \
                       &(struct sigaction){ .sa_handler = SIG_DFL },    \
                       &sigaction_old) )                                \
    {                                                                   \
        PyErr_SetString(PyExc_RuntimeError, "sigaction() failed");      \
        goto done;                                                      \
    }                                                                   \
} while(0)
#define RESET_SIGINT() do {                                             \
    if( 0 != sigaction(SIGINT,                                          \
                       &sigaction_old, NULL ))                          \
        PyErr_SetString(PyExc_RuntimeError, "sigaction-restore failed"); \
} while(0)

#define PYMETHODDEF_ENTRY(function_prefix, name, args) {#name,          \
                                                        (PyCFunction)function_prefix ## name, \
                                                        args,           \
                                                        function_prefix ## name ## _docstring}

#define BARF(fmt, ...) PyErr_Format(PyExc_RuntimeError, "%s:%d %s(): "fmt, __FILE__, __LINE__, __func__, ## __VA_ARGS__)


typedef struct {
    PyObject_HEAD

    dv4l_t camera;

} camera;


static int
camera_init(camera* self, PyObject* args, PyObject* kwargs)
{
    // Any existing factorization goes away. If this function fails, we lose the
    // existing factorization, which is fine. I'm placing this on top so that
    // __init__() will get rid of the old state
    dv4l_deinit(&self->camera);


    // error by default
    int result = -1;

    char* keywords[] = {"device",
                        "width",
                        "heigth",
                        "fps",
                        "streaming",
                        "pixelformat",
                        NULL};

    const char* device             = NULL;
    int         width              = -1;
    int         height             = -1;
    int         fps                = -1;
    int         streaming          = true;
    const char* pixelformat_fourcc = NULL;

    dv4l_pixelformat_choice_t pixelformat;

    if( !PyArg_ParseTupleAndKeywords(args, kwargs,
                                     "s|iiips", keywords,
                                     &device,
                                     &width,
                                     &height,
                                     &fps,
                                     &streaming,
                                     &pixelformat_fourcc ))
        goto done;


    if(pixelformat_fourcc == NULL)
        pixelformat.choice = BEST_COLOR_PIXELFORMAT;
    else
    {
        if(strlen(pixelformat_fourcc) > 4)
        {
            BARF("Invalid pixelformat_fourcc: '%s'. Must be at most 4 characters", pixelformat_fourcc);
            goto done;
        }

        pixelformat.choice = USE_REQUESTED_PIXELFORMAT;
        pixelformat.pixelformat =
            (((uint32_t)pixelformat_fourcc[0]) << 24) |
            (((uint32_t)pixelformat_fourcc[1]) << 16) |
            (((uint32_t)pixelformat_fourcc[2]) <<  8) |
            (((uint32_t)pixelformat_fourcc[3]) <<  0);
    }

    if(!dv4l_init(&self->camera,
                  device,
                  width,
                  height,
                  fps,
                  streaming,
                  pixelformat,
                  NULL, 0))
    {
        BARF("Couldn't init dv4l camera");
        goto done;
    }

    result = 0;

 done:
    return result;
}

static void camera_dealloc(camera* self)
{
    dv4l_deinit(&self->camera);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
camera_get_frame(camera* self, PyObject* args, PyObject* kwargs)
{
    PyObject* result = NULL;
    PyObject* image =
        PyArray_SimpleNew(3,
                          ((npy_intp[]){self->camera.format.fmt.pix.height,
                                        self->camera.format.fmt.pix.width,
                                        3}),
                          NPY_UINT8);
    uint64_t timestamp_us;

    if(image == NULL)
    {
        BARF("Couldn't allocate image output");
        goto done;
    }

    if(!dv4l_getframe(&self->camera,
                      PyArray_DATA((PyArrayObject*)image),
                      &timestamp_us))
    {
        BARF("Couldn't capture frame");
        goto done;
    }

    result =
        Py_BuildValue("(KO)",
                      timestamp_us, image);

 done:
    Py_XDECREF(image);

    return result;
}

static const char camera_docstring[] =
#include "camera.docstring.h"
    ;
static const char camera_get_frame_docstring[] =
#include "camera_get_frame.docstring.h"
    ;

static PyMethodDef camera_methods[] =
    {
        PYMETHODDEF_ENTRY(camera_, get_frame, METH_VARARGS | METH_KEYWORDS),
        {}
    };


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
// PyObject_HEAD_INIT throws
//   warning: missing braces around initializer []
// This isn't mine to fix, so I'm ignoring it
static PyTypeObject camera_type =
{
     PyObject_HEAD_INIT(NULL)
    .tp_name      = "dv4l.camera",
    .tp_basicsize = sizeof(camera),
    .tp_new       = PyType_GenericNew,
    .tp_init      = (initproc)camera_init,
    .tp_dealloc   = (destructor)camera_dealloc,
    .tp_methods   = camera_methods,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = camera_docstring,
};
#pragma GCC diagnostic pop



#define MODULE_DOCSTRING \
    "Python-wrapper around the dv4l video4linux2 library\n"

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "dv4l",
     MODULE_DOCSTRING,
     -1,
     NULL,
    };

PyMODINIT_FUNC PyInit_dv4l(void)
{
    if (PyType_Ready(&camera_type) < 0)
        return NULL;

    PyObject* module = PyModule_Create(&module_def);

    Py_INCREF(&camera_type);
    PyModule_AddObject(module, "camera", (PyObject *)&camera_type);

    import_array();

    return module;
}
