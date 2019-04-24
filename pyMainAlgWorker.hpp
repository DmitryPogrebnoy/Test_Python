#ifndef PYMAINALGWORKER_H
#define PYMAINALGWORKER_H

#include <QObject>

#include "pyAlgConfig.hpp"

struct PyMainAlgWorker : public QObject
{
    Q_OBJECT


public:
    PyMainAlgWorker();
    ~PyMainAlgWorker();


public slots:
    int startPython(const char* name);
    void stopPython();
    void run(double** arguments);

private:
    pyAlgConfig config;
    bool workerStart;
    bool pythonStart;
};

#endif // PYMAINALGWORKER_H
