#ifndef PYMAINALGWORKER_H
#define PYMAINALGWORKER_H

#include <QObject>

struct PyMainAlgWorker : public QObject
{
    Q_OBJECT
public:
    PyMainAlgWorker();
    ~PyMainAlgWorker();

public slots:
    int start(const char* name);
    void stop();
    void run(double** arguments);
};

#endif // PYMAINALGWORKER_H
