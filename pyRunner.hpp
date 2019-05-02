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
    void start_signal(const QString name){
        emit startPython(name);
    }
    void stop_signal(){
        emit stopPython();
    }
    void pause_unpause_signal(){
        emit pause_unpause();
    }
    void set_py_scripts_dir(const QString dir){
        emit setPyScriptsDir(dir);
    }
signals:
    void startPython(const QString);
    void stopPython();
    void run(double**);
    void pause_unpause();
    void setPyScriptsDir(const QString);
};

#endif // PYRUNNER_HPP
