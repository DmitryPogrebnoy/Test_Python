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

#include "pyExcept.hpp"
#include "pyScopedPointerDeleter.hpp"

using namespace std;


PyMainAlgWorker::PyMainAlgWorker(){}

PyMainAlgWorker::~PyMainAlgWorker(){}

void PyMainAlgWorker::run(double** arguments)
{
    try{
        //Set sys.argv[0] = "main_python_script"
        wchar_t* py_argv_init[1];
        QScopedArrayPointer<wchar_t*,ScopedSingleElemArrayPointerPy_DecodeLocaleDeleter> py_argv(py_argv_init);
        py_argv[0] = Py_DecodeLocale("main_python_script", nullptr);
        if(py_argv.isNull()){
            throw DecodeException("main_python_script");
        }
        PySys_SetArgv(1, py_argv.data());

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_NameModule(PyUnicode_DecodeFSDefault("main_python_script"));
        if (py_NameModule.isNull()) {
            throw DecodeException("main_python_script");
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Module(PyImport_Import(py_NameModule.data()));
        if (py_Module.isNull()) {
            throw ModuleNotFoundException("main_python_script");
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Method(PyObject_GetAttrString(py_Module.data(), "main"));
        if (py_Method.isNull() || !PyCallable_Check(py_Method.data())) {
            throw MethodNotFoundException("main_python_script","main");
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Args(PyTuple_New(4));
        if (py_Args.isNull()){
            throw CreateTupleException(4);
        }

        npy_intp dims_Ball[1] = {3};
        npy_intp dims_TeamBlue[1] = {48};
        npy_intp dims_TeamYellow[1] = {48};
        npy_intp dims_BallisInside[1] = {1};
        //Определяем обычные указатели т.к. функция PyTuple_SetItem забирает права владения ссылкой на объект при присваивании
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

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Value(PyObject_CallObject(py_Method.data(), py_Args.data()));
        if (py_Value.isNull()){
            throw CallMethodException("main_python_script","main");
        }
        if (!PyTuple_Check(py_Value.data())){
            throw NotTupleException("py_Value");
        }
        if (PyTuple_Size(py_Value.data()) != 4){
            throw WrongSizeTupleException("py_Value",4,static_cast<const int>(PyTuple_Size(py_Value.data())));
        }

        /*
         * PyTuple_GetItem возвращает заимствованную ссылку на возвращаемый объект
         * Т.к. используются умные указатели, то необходимо увеличивать счетчик ссылок на 1
         * Тем самым возвращаемое значение будет с полными правами
        */
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_List_Ball_Back(PyTuple_GetItem(py_Value.data(),0));
        Py_INCREF(py_List_Ball_Back.data());
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_List_TeamBlue_Back(PyTuple_GetItem(py_Value.data(),1));
        Py_INCREF(py_List_TeamBlue_Back.data());
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_List_TeamYellow_Back(PyTuple_GetItem(py_Value.data(),2));
        Py_INCREF(py_List_TeamYellow_Back.data());
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_List_BallisInside_Back(PyTuple_GetItem(py_Value.data(),3));
        Py_INCREF(py_List_BallisInside_Back.data());


        if (!PyArray_Check(py_List_Ball_Back.data())){
            throw NotArrayException("py_List_Ball");
        }
        if (!PyArray_Check(py_List_TeamBlue_Back.data())){
            throw NotArrayException("py_List_TeamBlue");
        }
        if (!PyArray_Check(py_List_TeamYellow_Back.data())){
            throw NotArrayException("py_List_TeamYellow");
        }
        if (!PyArray_Check(py_List_BallisInside_Back.data())){
            throw NotArrayException("py_List_BallisInside");
        }


        if (PyArray_NDIM(py_List_Ball_Back.data()) != 1){
            throw WrongDimArrayException("py_List_Ball",1,PyArray_NDIM(py_List_Ball_Back.data()));
        }
        if (PyArray_NDIM(py_List_TeamBlue_Back.data()) != 1){
            throw WrongDimArrayException("py_List_TeamBlue",1,PyArray_NDIM(py_List_TeamBlue_Back.data()));
        }
        if (PyArray_NDIM(py_List_TeamYellow_Back.data()) != 1){
            throw WrongDimArrayException("py_List_TeamYellow",1,PyArray_NDIM(py_List_TeamYellow_Back.data()));
        }
        if (PyArray_NDIM(py_List_BallisInside_Back.data()) != 1){
            throw WrongDimArrayException("py_List_BallisInside",1,PyArray_NDIM(py_List_BallisInside_Back.data()));
        }


        if (PyArray_DIMS(py_List_Ball_Back.data())[0] != 3){
            throw WrongSizeArrayException("py_List_Ball",3,PyArray_DIMS(py_List_Ball_Back.data())[0]);
        }
        if (PyArray_DIMS(py_List_TeamBlue_Back.data())[0] != 48){
            throw WrongSizeArrayException("py_List_TeamBlue",48,PyArray_DIMS(py_List_TeamBlue_Back.data())[0]);
        }
        if (PyArray_DIMS(py_List_TeamYellow_Back.data())[0] != 48){
            throw WrongSizeArrayException("py_List_TeamYellow",48,PyArray_DIMS(py_List_TeamYellow_Back.data())[0]);
        }
        if (PyArray_DIMS(py_List_BallisInside_Back.data())[0] != 1){
            throw WrongSizeArrayException("py_List_BallisInside",1,PyArray_DIMS(py_List_BallisInside_Back.data())[0]);
        }

        double* return_list_Ball = reinterpret_cast<double*>PyArray_DATA(py_List_Ball_Back.data());
        double* return_list_TeamBlue = reinterpret_cast<double*>PyArray_DATA(py_List_TeamBlue_Back.data());
        double* return_list_TeamYellow = reinterpret_cast<double*>PyArray_DATA(py_List_TeamYellow_Back.data());
        double* return_list_BallisInside = reinterpret_cast<double*>PyArray_DATA(py_List_BallisInside_Back.data());

        cout<<"Return of call: "<<endl<<" [ ";
        for (int i = 0; i < dims_Ball[0]; i++){
            cout<<return_list_Ball[i]<<" ";
        }
        cout<<"]"<<endl<<" [";
        for (int i = 0; i < dims_TeamBlue[0]; i++){
            cout<<return_list_TeamBlue[i]<<" ";
        }
        cout<<"]"<<endl<<" [";
        for (int i = 0; i < dims_TeamYellow[0]; i++){
            cout<<return_list_TeamYellow[i]<<" ";
        }
        cout<<"]"<<endl<<" [";
        for (int i = 0; i < dims_BallisInside[0]; i++){
            cout<<return_list_BallisInside[i]<<" ";
        }
        cout<<"]"<<endl;


    } catch (Exception& e) {
        cerr<<e.message()<<endl;
        if (PyErr_Occurred()) PyErr_Print();
    }
}

int PyMainAlgWorker::start(const char* name)
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
    return 0;
}

void PyMainAlgWorker::stop()
{
    Py_Finalize();
}
