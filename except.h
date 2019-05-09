#ifndef EXCEPT_H
#define EXCEPT_H

#include <QString>

class Exception{
public:
    //Constructor
    explicit Exception(const QString &msg): msg(msg)
    {}
    //Get error message
    QString message() const {
        return msg;
    }
private:
    //Error message
    QString msg;
};
#endif // EXCEPT_H
