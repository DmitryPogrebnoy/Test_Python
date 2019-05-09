#pragma once

#include <QObject>
using namespace std;
#include <iostream>

class Printer : public QObject
{
    Q_OBJECT

public:
    Printer(){}
    ~Printer(){}

public slots:
    void printError(const QString str){
        cout<<str.toStdString()<<"<-!ERRORS!"<<endl;
    }
    void printOutput(const QString str){
        cout<<str.toStdString()<<"<-!OUTPUT!"<<endl;
    }
    void pauseState(bool status){
        cout<<endl<<"Is pause? "<<status<<" <-PAUSESTATE"<<endl<<endl;
    }
};
