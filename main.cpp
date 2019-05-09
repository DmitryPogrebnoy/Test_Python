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

#include "pythonEngine.h"
#include "printer.h"
using namespace std;


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    EngineInterface * pyEngine = new PythonEngine();

    Printer * printer = new Printer();
    QObject::connect(pyEngine,SIGNAL(consoleMessage(const QString)),printer,SLOT(printOutput(const QString)));
    QObject::connect(pyEngine,SIGNAL(isPause(bool)),printer,SLOT(pauseState(bool)));


    const QString dir= "C:/Users/pogre/Desktop/Test_Python/PYScripts/";
    pyEngine->setDirectory(dir);

    pyEngine->evaluate();
    pyEngine->evaluate();
    pyEngine->evaluate();
    pyEngine->evaluate();

    pyEngine->pauseUnpause();

    //pyEngine->evaluate();


    cout<<"Ok"<<endl;

    return a.exec();
}
