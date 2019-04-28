#ifndef EXCEPT_H
#define EXCEPT_H

#include <string>
using namespace std;

class Exception{
public:
    //Constructor
    explicit Exception(const string &msg): msg(msg)
    {}
    //Get error message
    string message() const {
        return msg;
    }
private:
    //Error message
    string msg;
};
#endif // EXCEPT_H
