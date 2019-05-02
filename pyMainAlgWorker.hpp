#ifndef PYMAINALGWORKER_H
#define PYMAINALGWORKER_H

#include <QObject>

#include "pyAlgConfig.hpp"

struct PyMainAlgWorker : public QObject
{
    Q_OBJECT
    //Т.к. камерами мяч не детектится, то мяч детектится роботами
    //robotReceiver выставляет флаг - есть мяч на поле или нет
    bool isBallInside;
    /*
     * countCallPython - количество вызовов скрипта Python
     * timer_scope - для замера времени выполнения одного скрипта Python
     * timer_max - для замера максимального времени выполнения одного скрипта Python
     * timer_check - для того, чтобы делать сбор статистики раз в секуду в run
     * timer_sum - сумма всех выполнений скриптов Python
     */
    long countCallPython;
    clock_t timer_scope, timer_max, timer_check, timer_sum;

    //Client client;

public:
    PyMainAlgWorker();
    ~PyMainAlgWorker();

signals:
    void pyMainAlgFree();
    void pySendToConnector(int, QByteArray command);
    void pyStatusMessage(QString message);
    //этот может и не понадобиться
    void pyUpdatePauseState(QString message);

public slots:
    int startPython(const QString name);
    void stopPython();
    void run(double** arguments);
    void pause_unpause();
    void setPyScriptsDir(const QString dir);
    void changeBallStatus(bool ballStatus);

private:
    pyAlgConfig config;
    bool pythonStart;
};

#endif // PYMAINALGWORKER_H
