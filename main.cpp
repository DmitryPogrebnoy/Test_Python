//
#include <math.h>
#include <Python.h>
/* Система Win10 64-bit
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

 /*
 * Кроме того, подключать нужно библиотеку pythonX.lib, где X - версия. В данный момент используется библиотека python37.lib.
 */
#include <QCoreApplication>
#include <stdio.h>


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    //printf("%s %s %s %s %s %s",argv[0],"\n",argv[1],"\n",argv[1],"\n");

    /* Устанавливается один раз при запуске main, чтобы определить значение argv[0] - имя программы
     * Оно используется Py_GetPath()и некоторыми другими функциями, чтобы найти run-time библиотеки Python относительно исполняемого файла интерпретатора.
     *
     * В конце программы необходимо вызвать PyMem_RawFree(program); чтобы очистить память.
    */
    wchar_t *program = Py_DecodeLocale(argv[0], nullptr);
    if (program == nullptr) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);  /* optional but recommended */
    //


    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue;
    int i;
    int argument[2] = {2,10};

    Py_Initialize();
    /*
     * Пути до папок с необходимыми скриптами должны лежать в PYTHONPATH или сами скрипты должны лежать в папке с запускаемым приложением.
     * Вывести текущее значение PYTHONPATH можно запустив python -c "import sys; print('\n'.join(sys.path))"
    */
    pName = PyUnicode_DecodeFSDefault("python_script");
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, "multiply");
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(2);
            for (i = 0; i < 2; ++i) {
                pValue = PyLong_FromLong(argument[i]);
                if (!pValue) {
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
                    fprintf(stderr, "Cannot convert argument\n");
                    return 1;
                }
                /* pValue reference stolen here: */
                PyTuple_SetItem(pArgs, i, pValue);
            }
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", "multiply");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", "python_script");
        return 1;
    }
    if (Py_FinalizeEx() < 0) {
        return 120;
    }


    /*
    Py_Initialize();
    PyRun_SimpleString("from time import time,ctime\n"
                           "print('Today is', ctime(time()))\n");
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    */

    //очищаем память
    PyMem_RawFree(program);
    return a.exec();
}
