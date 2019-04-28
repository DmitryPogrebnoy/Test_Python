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
    void pause_unpause();

private:
    pyAlgConfig config;
    bool pythonStart;
};

#endif // PYMAINALGWORKER_H
