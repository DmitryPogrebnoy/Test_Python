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

#include "pyMainAlgWorker.hpp"

// Использую версию 1.16.2
#include <arrayobject.h>

//Почему-то все работает и без этих двух include хотя не должно
//#include <QScopedArrayPointer>
//#include <QScopedPointer>

#include <iostream>
#include <fstream>

#include "pyExcept.hpp"
#include "pyScopedPointerDeleter.hpp"

using namespace std;


PyMainAlgWorker::PyMainAlgWorker(){
    cout << "pyMainAlg worker start" << endl;
}

PyMainAlgWorker::~PyMainAlgWorker(){}

void PyMainAlgWorker::run(double** arguments)
{
    try{
        //Set sys.argv[0] = config.namePyMainScript
        wchar_t* py_argv_init[1];
        QScopedArrayPointer<wchar_t*,ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter> py_argv(py_argv_init);
        py_argv[0] = Py_DecodeLocale(config.get_namePyMainScript().c_str(), nullptr);
        if(py_argv.isNull()){
            throw DecodeException(config.get_namePyMainScript());
        }
        PySys_SetArgv(1, py_argv.data());

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_NameModule(PyUnicode_DecodeFSDefault(config.get_namePyMainScript().c_str()));
        if (py_NameModule.isNull()) {
            throw DecodeException(config.get_namePyMainScript());
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Module(PyImport_Import(py_NameModule.data()));
        if (py_Module.isNull()) {
            throw ModuleNotFoundException(config.get_namePyMainScript());
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Method(PyObject_GetAttrString(py_Module.data(), config.get_namePyMainMethod().c_str()));
        if (py_Method.isNull() || !PyCallable_Check(py_Method.data())) {
            throw MethodNotFoundException(config.get_namePyMainScript(),config.get_namePyMainMethod());
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Args(PyTuple_New(4));
        if (py_Args.isNull()){
            throw CreateTupleException(4);
        }

        npy_intp dims_Ball[1] = {3};
        npy_intp dims_TeamBlue[1] = {48};
        npy_intp dims_TeamYellow[1] = {48};
        npy_intp dims_BallisInside[1] = {1};
        //Определяем обычные указатели т.к. далее функция PyTuple_SetItem забирает права владения ссылкой на объект при присваивании
        //Передача прав происходит даже при неудачном присвоении
        PyObject* py_List_Ball = PyArray_SimpleNewFromData(1,dims_Ball, NPY_FLOAT64, arguments[0]);
        PyObject* py_List_TeamBlue = PyArray_SimpleNewFromData(1, dims_TeamBlue, NPY_FLOAT64, arguments[1]);
        PyObject* py_List_TeamYellow = PyArray_SimpleNewFromData(1, dims_TeamYellow, NPY_FLOAT64, arguments[2]);
        PyObject* py_List_BallisInside = PyArray_SimpleNewFromData(1, dims_BallisInside, NPY_FLOAT64, arguments[3]);


        if (PyTuple_SetItem(py_Args.data(), 0, py_List_Ball) != 0){
            throw CouldNotSetItemTupleException("py_Args",0,"py_List_Ball");
        }
        if (PyTuple_SetItem(py_Args.data(), 1, py_List_TeamBlue) != 0){
            throw CouldNotSetItemTupleException("py_Args",1,"py_List_TeamBlue");
        }
        if (PyTuple_SetItem(py_Args.data(), 2, py_List_TeamYellow) != 0){
            throw CouldNotSetItemTupleException("py_Args",2,"py_List_TeamYellow");
        }
        if (PyTuple_SetItem(py_Args.data(), 3, py_List_BallisInside) != 0){
            throw CouldNotSetItemTupleException("py_Args",3,"py_List_BallisInside");
        }

        //Запускаем Python функцию и получаем результат
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Value(PyObject_CallObject(py_Method.data(), py_Args.data()));
        if (py_Value.isNull()){
            throw CallMethodException(config.get_namePyMainScript(),config.get_namePyMainMethod());
        }
        if (!PyList_Check(py_Value.data())){
            throw NotArrayException("py_Value");
        }
        if (PyList_Size(py_Value.data()) != config.get_CONTROL_SIGNALS_AMOUNT()){
            throw WrongSizeArrayException("py_Value",config.get_CONTROL_SIGNALS_AMOUNT(),static_cast<const int>(PyList_Size(py_Value.data())));
        }

        for (int i = 0; i < config.get_CONTROL_SIGNALS_AMOUNT(); i++){
            if (!PyList_Check(PyList_GetItem(py_Value.data(),i))) {
                throw NotArrayException("py_Value[" + to_string(i) +"]");
            }
            if (PyList_Size(PyList_GetItem(py_Value.data(),i)) != config.get_CONTROL_SIGNALS_LENGTH()) {
                throw WrongSizeArrayException("py_Value[" + to_string(i) +"]",config.get_CONTROL_SIGNALS_LENGTH(),PyList_Size(PyList_GetItem(py_Value.data(),i)));
            }
        }


        double** controlSignals = new double*[config.get_CONTROL_SIGNALS_AMOUNT()];
        for (int i = 0; i < config.get_CONTROL_SIGNALS_AMOUNT(); i++){
            controlSignals[i] = new double[config.get_CONTROL_SIGNALS_LENGTH()];
        }


        for (int i = 0; i < config.get_CONTROL_SIGNALS_AMOUNT(); i++){
            for (int j = 0; j < config.get_CONTROL_SIGNALS_LENGTH(); j++){
                controlSignals[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(py_Value.data(),i),j));
                //Если скастилось неудачно, то Python выставит ошибку и мы выбрасываем свою ошибку
                if (PyErr_Occurred()) {
                    throw CouldNotCastToDouble();
                }
            }
        }

        //На этом моменте у нас есть готовый массив массивов с управляющими сигналами для роботов

        cout<<"Return of call: "<<endl;
        for (int i = 0; i < config.get_CONTROL_SIGNALS_AMOUNT(); i++){
            cout<<" [ ";
            for (int j = 0; j < config.get_CONTROL_SIGNALS_LENGTH(); j++){
                cout<<controlSignals[i][j]<<" ";
            }
            cout<<"]"<<endl;
        }

    } catch (pyException& e) {
        cerr<<e.message()<<endl;
        if (PyErr_Occurred()) PyErr_Print();
    }
}

int PyMainAlgWorker::startPython(const char* name)
{
    /*
    wchar_t* program[1];
    QScopedArrayPointer<wchar_t*,ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter> programName(program);
    programName[0] = Py_DecodeLocale(name, nullptr);
    if (programName == nullptr) {
        cerr<<"Fatal error: cannot decode argv[0]\n"<<endl;
        exit(1);
    }
    Py_SetProgramName((programName.data())[0]);
    */

    Py_Initialize();

    /*
    //Выставляем путь до используемых скриптов(python логгера и т.д.), по умолчанию путь до директории с запускаемым приложением.
    PyRun_SimpleString(
       "import sys\n"
       "sys.path.append('C:/Users/pogre/Downloads')\n"
    );
    */

    //Использую Numpy C API
    //Вызывать только после Py_SetProgramName и Py_Initialize
    import_array();

    pythonStart = true;

    return 0;
}

void PyMainAlgWorker::stopPython()
{
    Py_Finalize();
    pythonStart = false;
}

void PyMainAlgWorker::pause_unpause(){
    try{
        //Set sys.argv[0] = config.namePyMainScript
        wchar_t* py_argv_init[1];
        QScopedArrayPointer<wchar_t*,ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter> py_argv(py_argv_init);
        py_argv[0] = Py_DecodeLocale(config.get_namePyPauseScript().c_str(), nullptr);
        if(py_argv.isNull()){
            throw DecodeException(config.get_namePyPauseScript());
        }
        PySys_SetArgv(1, py_argv.data());

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_NameModule(PyUnicode_DecodeFSDefault(config.get_namePyPauseScript().c_str()));
        if (py_NameModule.isNull()) {
            throw DecodeException(config.get_namePyPauseScript());
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Module(PyImport_Import(py_NameModule.data()));
        if (py_Module.isNull()) {
            throw ModuleNotFoundException(config.get_namePyPauseScript());
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Method(PyObject_GetAttrString(py_Module.data(), config.get_namePyPauseMethod().c_str()));
        if (py_Method.isNull() || !PyCallable_Check(py_Method.data())) {
            throw MethodNotFoundException(config.get_namePyPauseScript(), config.get_namePyPauseMethod());
        }


        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Value(PyObject_CallObject(py_Method.data(), nullptr));
        if (py_Value.isNull()){
            throw CallMethodException(config.get_namePyPauseScript(), config.get_namePyPauseMethod());
        }

    } catch (pyException& e) {
        cerr<<e.message()<<endl;
        if (PyErr_Occurred()) PyErr_Print();
    }
}
