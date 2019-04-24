#ifndef EXCEPT_H
#define EXCEPT_H

#include <string>
using namespace std;

class Exception{
public:
    //Constructor
    explicit Exception(const std::string &msg): msg(msg)
    {}
    //Get error message
    std::string message() const {
        return msg;
    }
private:
    //Error message
    std::string msg;
};
#endif // EXCEPT_H
