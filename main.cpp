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
 * Используем библиотеку python3.lib 
 */
#include <QCoreApplication>
#include <stdio.h>
#include <pyhelper.h>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    //printf("%s %s %s %s %s %s",argv[0],"\n",argv[1],"\n",argv[1],"\n");


    /*
     * Я написал "оболочку" для некоторых объектов которые использую. Чтобы не следить за памятью.
     * Она будет автоматически очищаться при выходе из области видимости.
     * Хорошая ли это идея?
     *
    */

    CPyObject pName, pModule, pFunc, pArgs, pValue;

    int i;
    int argument[2] = {2,10};

    wchar_t *program = Py_DecodeLocale(argv[0], nullptr);
       if (program == nullptr) {
           fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
           exit(1);
    }
    Py_SetProgramName(program);  /* optional but recommended */
    //

    /*
     * CPyObject и CPyInstance избавляют от необходимости следить за очисткой(при выходе из зоны видимости очищаются автоматически).
    */
    CPyInstance pyInstance;

    /*
     * Пути до папок с необходимыми скриптами должны лежать в PYTHONPATH или сами скрипты должны лежать в папке с запускаемым приложением.
     * Вывести текущее значение PYTHONPATH можно запустив python -c "import sys; print('\n'.join(sys.path))"
    */


    pName = PyUnicode_DecodeFSDefault("python_script_1");
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, "multiply");
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(2);
            for (i = 0; i < 2; ++i) {
                pValue = PyLong_FromLong(argument[i]);
                if (!pValue) {
                    fprintf(stderr, "Cannot convert argument\n");
                    PyMem_Free(program);
                    return 1;
                }
                /* pValue reference stolen here: */
                PyTuple_SetItem(pArgs, i, pValue);
            }
            pValue = PyObject_CallObject(pFunc, pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyLong_AsLong(pValue));
            }
            else {
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                PyMem_Free(program);
                return 1;
            }
        }
        else {
            if (PyErr_Occurred()) PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", "multiply");
        }
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", "python_script");
        PyMem_Free(program);
        return 1;
    }


    //Тест подключения библиотек в скрипты
    CPyObject pArg1;

    /*
     * Возникла проблема, что некоторые библиотеки(например matplotlib.pyplot) требуют sys.arg[0] путь (какой угодно, хоть "123", в документации ничего его значение не указано), но возможно лучше указывать путь до скрипта который будет исполняться.
     * При работе с обычным python путь исполняемого скрипта устанавливается автоматически.
     * При встраивании python это не работает. Поэтому перед запуском скриптов нужно принудительно его определять.
     *
    */
    wchar_t * pyargvW;
    pyargvW = Py_DecodeLocale("123", nullptr);
    PySys_SetArgv(1, &pyargvW);



    pName = PyUnicode_DecodeFSDefault("python_script_2");
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, "test");
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {

            pValue = PyObject_CallObject(pFunc, NULL);
            if (PyErr_Occurred()) PyErr_Print(); // Вот здесь выводим пойманную ошибку
            if (pValue != NULL) {
                printf("Result of call: %f\n", PyFloat_AsDouble(pValue));
            }
            else {
                fprintf(stderr,"Call failed\n");
                PyMem_Free(program);
                return 1;
            }
        }
        else {
            if (PyErr_Occurred()) PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", "multiply");
        }
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", "python_script");
        PyMem_Free(program);
        return 1;
    }

    PyMem_Free(program);
    return a.exec();
}
