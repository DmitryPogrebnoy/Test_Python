#ifndef PYSCOPEDPOINTERDELETER_H
#define PYSCOPEDPOINTERDELETER_H

#include <math.h>
#include <Python.h>
// Использую библиотеку python37.lib
/*
 * Система Win10 64-bit
 * С подключением Python.h возник баг в уже существующей библиотеке cmath
 *
 * Из коробки pyconfig.h(который находится там же где и Python.h) в Windows определяет
 * hypot как _hypot, который впоследствии наносит вред объявлению hypot в
 * <math.h> и в конечном итоге нарушает объявление using в <cmath>.
 *
 * Фиксится баг добавлением #include <math.h> или #include <cmath> перед #include <Python.h>
 * Затем, при необходимости, подключаем остальные необходимые файлы.
 *
 * На 32 битной версии этого бага может не быть.
 */

struct ScopedPointerPy_DecodeLocaleDeleter{
    static inline void cleanup(wchar_t* py_argv){
        PyMem_RawFree(py_argv);
    }
};

struct ScopedPointerPyObjectDeleter{
    static inline void cleanup(PyObject* p){
        Py_CLEAR(p);
    }
};

#endif // PYSCOPEDPOINTERDELETER_H
