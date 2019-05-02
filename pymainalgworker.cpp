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


#include <iostream>
#include <fstream>

//Мои хедeры
#include "pyExcept.hpp"
#include "pyScopedPointerDeleter.hpp"

//Хедеры из проекта
#include "message.h"
using namespace std;


PyMainAlgWorker::PyMainAlgWorker(){
    cout << "pyMainAlg worker start" << endl;
    isBallInside = false;

    timer_scope = 0;
    timer_max = 0;
    timer_check = clock();
    timer_sum = 0;

    countCallPython = 0;

    //Тут еще должна быть инициализация client если он всетаки нужен
}

PyMainAlgWorker::~PyMainAlgWorker(){}

void PyMainAlgWorker::run(double** arguments)
{
    timer_scope = clock();
    countCallPython++;

    int** controlSignals = new int*[config.get_CONTROL_SIGNALS_AMOUNT()];
    for (int i = 0; i < config.get_CONTROL_SIGNALS_AMOUNT(); i++){
        controlSignals[i] = new int[config.get_CONTROL_SIGNALS_LENGTH()];
    }

    //Инициализируем нулями
    for (int i = 0; i < config.get_CONTROL_SIGNALS_AMOUNT(); i++){
        for (int j = 0; j < config.get_CONTROL_SIGNALS_LENGTH(); j++){
            controlSignals[i][j] = 0;
        }
    }

    //Запускаем вычисление управляющих сигналов в Python скрипте
    try{
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


        /*
         * Преобразовываем значения возвращенные Python в массивы int
         * Важно, что возвращаем числовые значения, которые влезают в int иначе
         * если не влезает в long, то выбрасываем исключение
         * если влезает в long, то теряем старшие биты
         */
        for (int i = 0; i < config.get_CONTROL_SIGNALS_AMOUNT(); i++){
            for (int j = 0; j < config.get_CONTROL_SIGNALS_LENGTH(); j++){
                controlSignals[i][j] = static_cast<int>(PyLong_AsLong(PyList_GetItem(PyList_GetItem(py_Value.data(),i),j)));
                //Если скастилось неудачно, то Python выставит ошибку и мы выбрасываем исключение
                if (PyErr_Occurred()) {
                    throw CouldNotCastToDouble();
                }
            }
        }

        //На этом моменте у нас есть готовый массив массивов с управляющими сигналами для роботов
        //Важно чтобы длина и количество управляющих сигналов возвращаемых из Python скрипта
        //совпадала с настройками config иначе выбросится ошибка

    } catch (pyException& e) {
        cerr<<e.message()<<endl;
        if (PyErr_Occurred()) PyErr_Print();
    }

    //Отправка, пришедших из Python скрипта, управлющих сигналов на роботов
    /*
     * Считаем, что Python script возвращает управляющие сигналы,
     *  каждый из которых это 6 интов, где
     * 1 - номер робота
     * 2 - SpeedX
     * 3 - SpeedY
     * 4 - kickForward
     * 5 - SpeedR
     * 6 - kickUp
    */
    for(int i = 0; i < config.get_CONTROL_SIGNALS_AMOUNT(); i++){
        Message msg;

        msg.setKickVoltageLevel(12);
        msg.setKickerChargeEnable(1);

        msg.setSpeedX(controlSignals[i][1]);
        msg.setSpeedY(controlSignals[i][2]);
        msg.setKickForward(controlSignals[i][3]);
        msg.setSpeedR(controlSignals[i][4]);
        msg.setKickUp(controlSignals[i][5]);

        QByteArray command = msg.generateByteArray();

        emit pySendToConnector(i+1,command);
    }


    //Раз в секунду создаем пакет статистики и отправляем в GUI
    timer_scope = clock() - timer_scope;
    if (timer_scope > timer_max)
        timer_max = timer_scope;

    timer_sum = timer_sum + timer_scope;

    if (clock() - timer_check > CLOCKS_PER_SEC){
        timer_check = clock();

        QString temp;
        QString toStatus = "Using Python: count call = ";
        temp.setNum(countCallPython);
        toStatus = toStatus + temp;

        toStatus = toStatus + ", mean time = ";
        temp.setNum(timer_sum/countCallPython);
        toStatus = toStatus + temp;

        toStatus = toStatus + ", max time = ";
        temp.setNum(timer_max);
        toStatus = toStatus + temp;

        toStatus = toStatus + ", sum time = ";
        temp.setNum(timer_sum);
        toStatus = toStatus + temp;

        timer_sum = 0;
        timer_max = 0;
        countCallPython = 0;

        emit pyStatusMessage(toStatus);

        // !!ДАЛЬШЕ ИДЕТ КАКАЯ-ТО ДИЧь
    }

    //Освобождаем память
    for (int i = 0; i < config.get_CONTROL_SIGNALS_AMOUNT(); i++){
        delete [] controlSignals[i];
    }
    delete [] controlSignals;

    //Сообщаем ресиверу о готовности обработки нового пакета.
    emit pyMainAlgFree();
}

int PyMainAlgWorker::startPython(const QString name)
{
    //По документации нужно сохранять в промежуточный QByteArray иначе крашится
    QByteArray byte_array_name = name.toLocal8Bit();
    wchar_t* programName = Py_DecodeLocale(byte_array_name.data(), nullptr);
    if (programName == nullptr) {
        cerr<<"Fatal error: cannot decode argv[0]\n"<<endl;
        exit(1);
    }
    Py_SetProgramName(programName);
    PyMem_RawFree(programName);


    Py_Initialize();

    //Set python sys.argv[0] = "LACRmaCS Python algorithm"
    //Т.к. некоторые модули падают(например pyplot), если sys.argv пустой
    QScopedPointer<wchar_t,ScopedPointerPy_DecodeLocaleDeleter> py_argv((Py_DecodeLocale("LACRmaCS Python algorithm", nullptr)));
    if(py_argv.isNull()){
        throw DecodeException(config.get_namePyMainScript());
    }
    wchar_t* set_argv = py_argv.data();
    PySys_SetArgv(1, &set_argv);


    //Подключаем необходимые модули в питоне, чтобы потом можно было выставлять пути до папки со скриптами
    PyRun_SimpleString(
       "import sys\n"
    );

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

void PyMainAlgWorker::setPyScriptsDir(const QString dir){

    PyRun_SimpleString(
                (string("sys.path.append('")  + dir.toStdString() + "')\n").c_str());
}

void PyMainAlgWorker::changeBallStatus(bool ballStatus){
    isBallInside = ballStatus;
}
