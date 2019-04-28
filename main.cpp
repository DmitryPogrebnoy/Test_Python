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
#include <iostream>

#include "pyMainAlgWorker.hpp"
#include "pyRunner.hpp"

using namespace std;


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    PyMainAlgWorker pyWorker;
    pyRunner pyRunner;

    QObject::connect(&pyRunner, SIGNAL(startPython(const char*)),&pyWorker, SLOT(startPython(const char*)));
    QObject::connect(&pyRunner, SIGNAL(run(double**)),&pyWorker, SLOT(run(double**)));
    QObject::connect(&pyRunner, SIGNAL(stopPython()),&pyWorker, SLOT(stopPython()));
    QObject::connect(&pyRunner, SIGNAL(pause_unpause()),&pyWorker, SLOT(pause_unpause()));

    pyRunner.start_signal(argv[0]);

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

    PyRun_SimpleString("import pause");
    PyRun_SimpleString("print(pause.PAUSE)");

    pyRunner.pause_unpause_signal();

    pyRunner.run_signal(arguments);

    pyRunner.pause_unpause_signal();

    pyRunner.run_signal(arguments);

    PyRun_SimpleString("print(pause.PAUSE)");


    for (int i = 0; i < 4; i++){
        delete [] arguments[i];
    }
    delete [] arguments;

    pyRunner.stop_signal();

    cout<<"Ok"<<endl;

    return a.exec();
}


