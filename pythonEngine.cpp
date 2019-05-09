#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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

#include <QCoreApplication>
#include <QDebug>

#include "pythonEngine.h"
#include "pyScopedPointerDeleter.h"
#include "pyExcept.h"
#include "pyConfig.h"

#include "constants.h"

// Использую версию 1.16.2
#include <arrayobject.h>

PythonEngine::PythonEngine(){

    QString name = QCoreApplication::applicationFilePath();
    QByteArray byteArrayName = name.toLocal8Bit();
    wchar_t* programName = Py_DecodeLocale(byteArrayName.data(), nullptr);
    if (programName == nullptr) {
        qDebug()<<"Fatal error: cannot decode application file path"<<endl;
    }
    Py_SetProgramName(programName);
    PyMem_RawFree(programName);

    Py_Initialize();

    QScopedPointer<wchar_t,ScopedPointerPy_DecodeLocaleDeleter> py_argv((Py_DecodeLocale("Python algorithm", nullptr)));
    wchar_t* set_argv = py_argv.data();
    PySys_SetArgv(1, &set_argv);


    //Подключаем необходимые модули в питоне, чтобы потом можно было выставлять пути до папки со скриптами
    //И по дефолту ищем скрипты в /PYScripts
    //Так же перенапрявляем output и error
    //При возникновении ошибки при выполнении Python скрипта stdout становится недоступным и до него не достучаться
    //Поэтому перенаправляем stdout и stderr в один буфер, чтобы можно было посмотреть на вывод который был до возникновения ошибки
    //После вызова любой функци в конце буфер очищается.
    PyRun_SimpleString("import sys\n"
                       "import os\n"
                       "import io\n"
                       "sys.path.append(os.path.join(os.getcwd(),'PYScripts'))\n"
                       "sys.stdout = io.StringIO()\n"
                       "sys.stderr = sys.stdout\n");


    //Использую Numpy C API
    //Вызывать только после Py_SetProgramName и Py_Initialize
    import_array1();
}

PythonEngine::~PythonEngine(){
    Py_Finalize();
}

void PythonEngine::evaluate(){

    cleanOutputError();

    //Начальные позиции роботов пока так, при встраивании через sharedRes
    double** arguments = new double*[4];
    arguments[0] = new double[3];
    arguments[1] = new double[48];
    arguments[2] = new double[48];
    arguments[3] = new double[1];
    for (int i = 0; i < 3; i++) {
        arguments[0][i] = i;
    }
    for (int i = 0; i < 48; i++) {
        arguments[1][i] = i;
        arguments[2][i] = i;
    }
    arguments[3][0] = 0;


    //Промежуточный буфер для управляющих сигналов
    int** controlSignals = new int*[Constants::ruleAmount];
    for (int i = 0; i < Constants::ruleAmount; i++){
        controlSignals[i] = new int[Constants::ruleLength];
    }
    //Инициализируем нулями
    for (int i = 0; i < Constants::ruleAmount; i++){
        for (int j = 0; j < Constants::ruleLength; j++){
            controlSignals[i][j] = 0;
        }
    }


    //Запускаем вычисление управляющих сигналов в Python скрипте
    try{

        QByteArray byteArrayNamePyMainScript = PyConfig::namePyMainScript.toLocal8Bit();
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_NameModule(PyUnicode_DecodeFSDefault(byteArrayNamePyMainScript.data()));
        if (py_NameModule.isNull()) {
            throw DecodeException(PyConfig::namePyMainScript);
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Module(PyImport_Import(py_NameModule.data()));
        if (py_Module.isNull()) {
            throw ObjectNotFoundException(PyConfig::namePyMainScript);
        }

        QByteArray byteArrayNamePyMainMethod = PyConfig::namePyMainMethod.toLocal8Bit();
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Method(PyObject_GetAttrString(py_Module.data(), byteArrayNamePyMainMethod.data()));
        if (py_Method.isNull() || !PyCallable_Check(py_Method.data())) {
            throw AttrNotFoundException(PyConfig::namePyMainScript,PyConfig::namePyMainMethod);
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
            throw CouldNotSetItemTupleException(QString("py_Args"),0,QString("py_List_Ball"));
        }
        if (PyTuple_SetItem(py_Args.data(), 1, py_List_TeamBlue) != 0){
            throw CouldNotSetItemTupleException(QString("py_Args"),1,QString("py_List_TeamBlue"));
        }
        if (PyTuple_SetItem(py_Args.data(), 2, py_List_TeamYellow) != 0){
            throw CouldNotSetItemTupleException(QString("py_Args"),2,QString("py_List_TeamYellow"));
        }
        if (PyTuple_SetItem(py_Args.data(), 3, py_List_BallisInside) != 0){
            throw CouldNotSetItemTupleException(QString("py_Args"),3,QString("py_List_BallisInside"));
        }


        //Запускаем Python функцию и получаем результат
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Value(PyObject_CallObject(py_Method.data(), py_Args.data()));
        if (py_Value.isNull()){
            throw CallMethodException(PyConfig::namePyMainScript, PyConfig::namePyMainMethod);
        }
        if (!PyList_Check(py_Value.data())){
            throw NotArrayReturnValueException(PyConfig::namePyMainScript, PyConfig::namePyMainMethod);
        }
        if (PyList_Size(py_Value.data()) != Constants::ruleAmount){
            throw WrongSizeArrayReturnValueException(Constants::ruleAmount, static_cast<const int>(PyList_Size(py_Value.data())));
        }

        for (int i = 0; i < Constants::ruleAmount; i++){
            if (!PyList_Check(PyList_GetItem(py_Value.data(),i))) {
                throw NotArrayReturnValueElementException(i);
            }
            if (PyList_Size(PyList_GetItem(py_Value.data(),i)) != Constants::ruleLength) {
                throw WrongSizeArrayReturnValueElementException(i,
                                                                Constants::ruleLength ,
                                                                static_cast<const int>(PyList_Size(PyList_GetItem(py_Value.data(),i))));
            }
        }


        /*
         * Преобразовываем значения возвращенные Python в массивы int
         * Важно, что возвращаем числовые значения, которые влезают в int иначе
         * если не влезает в long, то выбрасываем исключение
         * если влезает в long, то теряем старшие биты
         */
        for (int i = 0; i < Constants::ruleAmount; i++){
            for (int j = 0; j < Constants::ruleLength; j++){
                controlSignals[i][j] = static_cast<int>(PyLong_AsLong(PyList_GetItem(PyList_GetItem(py_Value.data(),i),j)));
                //Если скастилось неудачно, то Python выставит ошибку и мы выбрасываем исключение
                if (PyErr_Occurred()) {
                    throw CouldNotCastToDouble(i,j);
                }
            }
        }

        printOutputPython();

    } catch (pyException& e) {
        emit consoleMessage(e.message());
        printErrorPython();
    }


    //Отправка, пришедших из Python скрипта, управлющих сигналов на роботов
    //
    // ???????ПРИМЕРНО???????
    //
    /*
     * Считаем, что Python script возвращает управляющие сигналы,
     *  каждый из которых это 7 интов, где
     * 0 - "бит наличия правила"
     * 1 - номер робота
     * 2 - SpeedX
     * 3 - SpeedY
     * 4 - kickForward
     * 5 - SpeedR
     * 6 - kickUp
    */
    QVector<Rule> rules(Constants::ruleAmount);

    for (int i = 0; i < Constants::ruleAmount; i++) {
        if (controlSignals[i][0] == 1) {
            rules[i].mSpeedX = controlSignals[i][3];
            rules[i].mSpeedY = controlSignals[i][2];
            rules[i].mSpeedR = controlSignals[i][5];
            rules[i].mKickUp = controlSignals[i][6];
            rules[i].mKickForward = controlSignals[i][4];
        }
    }
    emit newData(rules);
    //

    //Удаление, при замене на sharedRes этого не будет
    for (int i = 0; i < 4; i++){
        delete [] arguments[i];
    }
    delete [] arguments;

    //Освобождаем память
    for (int i = 0; i < Constants::ruleAmount; i++){
        delete [] controlSignals[i];
    }
    delete [] controlSignals;
}

void PythonEngine::pauseUnpause(){
    cleanOutputError();
    try{
        QByteArray byteArrayNamePyPauseScript = PyConfig::namePyPauseScript.toLocal8Bit();
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_NameModule(PyUnicode_DecodeFSDefault(byteArrayNamePyPauseScript.data()));
        if (py_NameModule.isNull()) {
            throw DecodeException(byteArrayNamePyPauseScript.data());
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Module(PyImport_Import(py_NameModule.data()));
        if (py_Module.isNull()) {
            throw ObjectNotFoundException(byteArrayNamePyPauseScript.data());
        }

        QByteArray byteArrayNamePyPauseMethod = PyConfig::namePyPauseMethod.toLocal8Bit();
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Method(PyObject_GetAttrString(py_Module.data(), byteArrayNamePyPauseMethod.data()));
        if (py_Method.isNull() || !PyCallable_Check(py_Method.data())) {
            throw AttrNotFoundException(byteArrayNamePyPauseScript.data(), byteArrayNamePyPauseMethod.data());
        }

        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> py_Value(PyObject_CallObject(py_Method.data(), nullptr));
        if (py_Value.isNull()){
            throw CallMethodException(byteArrayNamePyPauseScript.data(), byteArrayNamePyPauseMethod.data());
        }
        if (!PyBool_Check(py_Value.data())){
            throw NotBoolReturnValueException(PyConfig::namePyPauseScript, PyConfig::namePyPauseMethod);
        }

        emit isPause(PyObject_IsTrue(py_Value.data()));
        printOutputPython();

    } catch (pyException& e) {
        emit consoleMessage(e.message());
        printErrorPython();
    }
}

void PythonEngine::setDirectory(const QString &path){
    cleanOutputError();
    QString simpleString = QString("sys.path.append('%1')\n").arg(path);
    QByteArray byteArraySimpleString = simpleString.toLocal8Bit();
    PyRun_SimpleString(byteArraySimpleString.data());
    if (PyErr_Occurred()) {
        printErrorPython();
    }
}

void PythonEngine::printOutputPython(){
    try{
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> sysModule(PyImport_AddModule("sys"));
        if (sysModule.isNull()) {
            throw ObjectNotFoundException(QString("sys"));
        }
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> stdoutBuffer(PyObject_GetAttrString(sysModule.data(),"stdout"));
        if (stdoutBuffer.isNull()) {
            throw AttrNotFoundException(QString("sys"),QString("stdout"));
        }

        if (checkBufferForEmptiness(stdoutBuffer.data(), QString("sys.stdout"))) {
            QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> stdoutBufferGetValue(PyObject_GetAttrString(stdoutBuffer.data(), "getvalue"));
            if (stdoutBufferGetValue.isNull()) {
                throw AttrNotFoundException(QString("sys.stdout"),QString("getvalue"));
            }
            QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> value(PyObject_CallObject(stdoutBufferGetValue.data(),nullptr));
            if (value.isNull()) {
                throw CallMethodException(QString("sys.stdout"),QString("getvalue"));;
            }
            QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> outputUTF8String(PyUnicode_AsUTF8String(value.data()));
            if (outputUTF8String.isNull()) {
                throw EncodeStreamException(QString("stdout"));;
            }
            char* outputStr = PyBytes_AsString(outputUTF8String.data());

            emit consoleMessage(QString(outputStr));
        }

    } catch (pyException& e){
        emit consoleMessage(QString("Could not access the output. %1").arg(e.message()));
    }
}

void PythonEngine::printErrorPython(){
    //Если есть ошибки то выводит их в stdout
    if (PyErr_Occurred()) {
        PyErr_Print();
    }
    try{
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> sysModule(PyImport_AddModule("sys"));
        if (sysModule.isNull()) {
            throw ObjectNotFoundException(QString("sys"));
        }
        QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> stderrBuffer(PyObject_GetAttrString(sysModule.data(),"stderr"));
        if (stderrBuffer.isNull()) {
            throw AttrNotFoundException(QString("sys"),QString("stderr"));
        }

        if (checkBufferForEmptiness(stderrBuffer.data(),QString("sys.stderr"))){
            QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> stderrBufferMethod(PyObject_GetAttrString(stderrBuffer.data(), "getvalue"));
            if (stderrBufferMethod.isNull()) {
                throw AttrNotFoundException(QString("sys.stderr"),QString("getvalue"));
            }
            QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> error(PyObject_CallObject(stderrBufferMethod.data(),nullptr));
            if (error.isNull()) {
                throw CallMethodException(QString("sys.stderr"),QString("getvalue"));;
            }
            QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> errorUTF8String(PyUnicode_AsUTF8String(error.data()));
            if (errorUTF8String.isNull()) {
                throw EncodeStreamException(QString("sys.stderr"));;
            }
            char* errorStr = PyBytes_AsString(errorUTF8String.data());
            emit consoleMessage(QString(errorStr));
        }

    } catch (pyException& e){
        emit consoleMessage(QString("Could not access the error. %1").arg(e.message()));
    }
}

void PythonEngine::cleanOutputError(){
    //Если отказаться от логгера, то можно просто инициализировать буфер заново, что вроде быстрее, чем чистить.
    //С логгером этого сделать нельзя, т.к. логер пишет (в том числе) в sys.stdout, а он, инициализируясь один раз, пишет в один и тот же буфер
    //При переинициализации буфера, логер продолжает писать в старый буфер, обновить его ссылку на буфер можно только инициализируя логгер заново
    //А это дольше, чем зачистить буфер.
    PyRun_SimpleString("sys.stdout.truncate(0)\n"
                       "sys.stdout.seek(0)\n");
}


bool PythonEngine::checkBufferForEmptiness(PyObject * buffer, const QString nameBuffer){
    QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> bufferTell(PyObject_GetAttrString(buffer, "tell"));
    if (bufferTell.isNull()) {
        throw AttrNotFoundException(nameBuffer,QString("tell"));
    }
    QScopedPointer<PyObject, ScopedPointerPyObjectDeleter> bufferPos(PyObject_CallObject(bufferTell.data(),nullptr));
    if (bufferPos.isNull()) {
        throw CallMethodException(nameBuffer,QString("tell"));;
    }

    int overflow;
    long long lengthBuffer = PyLong_AsLongLongAndOverflow(bufferPos.data(), &overflow);
    if ((lengthBuffer)||(overflow)){
        return true;
    } else {
        return false;
    }
}

