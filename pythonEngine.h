#pragma once
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

#include "engineinterface.h"

#include "pyConfig.h"

class PythonEngine : public EngineInterface
{

public:
    PythonEngine();
    ~PythonEngine();

    void evaluate();

    void pauseUnpause();

    void setDirectory(const QString & path);

private:
    void printOutputPython();
    void printErrorPython();
    void cleanOutputError();

    bool checkBufferForEmptiness(PyObject * buffer, const QString nameBaffer);
};

