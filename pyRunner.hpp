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
        emit startPython(name);
    }
    void stop_signal(){
        emit stopPython();
    }
    void pause_unpause_signal(){
        emit pause_unpause();
    }
signals:
    void startPython(const char*);
    void stopPython();
    void run(double**);
    void pause_unpause();
};

#endif // PYRUNNER_HPP
