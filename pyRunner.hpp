#ifndef PYRUNNER_HPP
#define PYRUNNER_HPP

#include <QObject>

class pyRunner : public QObject
{
    Q_OBJECT
public:
    pyRunner(){}
    ~pyRunner(){}
    void run_signal(double** arguments){
        emit run(arguments);
    }
    void start_signal(const char* name){
        emit start(name);
    }
    void stop_signal(){
        emit stop();
    }
signals:
    void start(const char*);
    void stop();
    void run(double**);
};

#endif // PYRUNNER_HPP
