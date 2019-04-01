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

// Использую версию 1.16.2
#include <arrayobject.h>


#include <QCoreApplication>
#include <QScopedArrayPointer>
#include <QScopedPointer>
#include <cstdio>
#include <iostream>
#include "pyexception.hpp"
using namespace std;


struct ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter{
    static inline void cleanup(wchar_t* py_argv[1]){
        PyMem_RawFree(py_argv[0]);
    }
};

//Будем передавать указатель на массив из 3 элементов -- пока так потом поменяю для нужного количества аргументов
struct ScopedArrayPointerPy_DecodeLocaleDeleter{
    static inline void cleanup(wchar_t* py_argv[3]){
        for(int i = 0; i < 3; i++) {
            PyMem_RawFree(py_argv[i]);
        }
    }
};

struct ScopedPointerPyObjectDeleter{
    static inline void cleanup(PyObject* p){
        Py_CLEAR(p);
    }
};


void runPyScript(const char* namePyScript, int py_argc, wchar_t** py_argv){
    PySys_SetArgv(py_argc, py_argv);
    FILE *file = _Py_fopen(namePyScript, "r+");
    if (file != nullptr) {
        PyRun_SimpleFile(file, namePyScript);
    } else {
        cout<<"File not found!"<<endl;
    };
}

void importPyLib() {
    wchar_t* arg[1];
    QScopedArrayPointer<wchar_t*,ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter> py_argv(arg);
    py_argv[0] = Py_DecodeLocale("py_script_import_lib.py", nullptr);
    runPyScript("py_script_import_lib.py", 1, py_argv.data());
}

//Метод принимает массив из двух массивов типа double
int runPyMethod(const char* nameModule, const char* nameMethod, int argument[2][3]) {
    try{
        //Set sys.argv[1] = nameModule
        wchar_t* py_argv_init[1];
        QScopedArrayPointer<wchar_t*,ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter> py_argv(py_argv_init);
        py_argv[0] = Py_DecodeLocale(nameModule, nullptr);
        PySys_SetArgv(1, py_argv.data());

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_NameModule(PyUnicode_DecodeFSDefault(nameModule));
        if (py_NameModule.isNull()) {
            throw DecodeException(nameModule);
        }
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Module(PyImport_Import(py_NameModule.data()));
        if (py_Module.isNull()) {
            throw ModuleNotFoundException(nameModule);
        }
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Method(PyObject_GetAttrString(py_Module.data(), nameMethod));
        if (py_Method.isNull() || !PyCallable_Check(py_Method.data())) {
            throw MethodNotFoundException(nameModule,nameMethod);
        }


        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Args(PyTuple_New(2));
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Value(PyLong_FromLong(0));

        npy_intp dims[1] = {3};
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_List1(PyArray_SimpleNewFromData(1, dims, NPY_INT, argument[0]));
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_List2(PyArray_SimpleNewFromData(1, dims, NPY_INT, argument[1]));

        PyTuple_SetItem(py_Args.data(), 0, py_List1.data());
        PyTuple_SetItem(py_Args.data(), 1, py_List2.data());


        py_Value.reset(PyObject_CallObject(py_Method.data(), py_Args.data()));
        if (py_Value.isNull()){
            throw CallMethodException(nameModule,nameMethod);
        }


        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Array(py_Value.data());
        if (!PyArray_Check(py_Array.data())){
            throw NotArrayException(nameModule, nameMethod);
        }
        if (PyArray_NDIM(py_Array.data()) != 1 || PyArray_DIMS(py_Array.data())[PyArray_NDIM(py_Array.data())-1] != 3){
            throw WrongDimArrayException(nameModule, nameMethod);
        }

        int* result = reinterpret_cast<int*>PyArray_DATA(py_Value.data());
        cout<<"Result of call: ["<<result[0]<<", "<<result[1]<<", "<<result[2]<<"]"<<endl;

    } catch (Exception& e) {
        cout<<e.message()<<endl;
        if (PyErr_Occurred()) PyErr_Print();
    }
    return 0;
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    wchar_t* program[1];
    QScopedArrayPointer<wchar_t*,ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter> programName(program);
    programName[0] = Py_DecodeLocale(argv[0], nullptr);
    if (programName == nullptr) {
        cerr<<"Fatal error: cannot decode argv[0]\n"<<endl;
        exit(1);
    }
    Py_SetProgramName((programName.data())[0]);

    Py_Initialize();

    //Попытка использования Numpy C API
    //Вызывать только после Py_SetProgramName и Py_Initialize
    import_array();

    importPyLib();

    wchar_t* py_argv_script_1[1];
    QScopedArrayPointer<wchar_t*,ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter> py_argv_1(py_argv_script_1);
    py_argv_1[0] = Py_DecodeLocale("python_script_1.py", nullptr);
    runPyScript("python_script_1.py", 1, py_argv_1.data());



    int arguments[2][3] = {{7,8,13},{2,9,5}};
    runPyMethod("python_script_test_method","test", arguments);

    cout<<"Point 3"<<endl;

    /*
    //Начала падать после того как успешно отрабатывает runPyMethod  - падает в самом скрипте при попытке вызвать np.dot() - опытным путем
    // выясненно что падает на любой функции numpy
    //Если runPyMethod закоменитить, то все успешно выполняется и завершается
    wchar_t* py_argv_script_2[3];
    QScopedArrayPointer<wchar_t*,ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter> py_argv_2(py_argv_script_2);
    py_argv_2[0] = Py_DecodeLocale("python_script_2.py", nullptr);
    py_argv_2[1] = Py_DecodeLocale("7,8,13", nullptr);
    py_argv_2[2] = Py_DecodeLocale("2,9,5", nullptr);
    runPyScript("python_script_2.py", 3, py_argv_2.data());
    */


    /*
     * Два способа вызвать метод:
     * 1) Полностью выполнить скрипт
     *    - Аргументы передаются через sys.argv
     *    - Все переменные в глобальной видимости, т.е. можем не переопределять переменные, значения которых не изменились
     *    - Ничего возвращать не можем
     * Из-за последнего пункта этот вариант не подходит -- слишком верхнеуровневый
     *
     * 2) Загрузить скрипт с пмощью API и выполнить конкретный метод
     *    - Аргументы передаются методу c помощью API
     *    - Можем возвращать значения
     *    - В методе все происходит в локальном окружении
     *
    */

    //Не использовать с умными указателями, иначе сегфолт
    //Py_Finalize();

    return a.exec();
}


