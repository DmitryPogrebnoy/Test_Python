#ifndef PYEXCEPTION_H
#define PYEXCEPTION_H

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

class DecodeException: public Exception
{
public:
    explicit DecodeException(const string &string) : Exception("Could not decode string: " + string)
    {}
};


class ModuleNotFoundException: public Exception
{
public:
    explicit ModuleNotFoundException(const string &moduleName) : Exception("Could not found Python module: " + moduleName)
    {}
};

class MethodNotFoundException: public Exception
{
public:
    MethodNotFoundException(const string &moduleName, const string &methodName)
        : Exception("Could not found method " + methodName + " in Python module " + moduleName)
    {}
};

class CallMethodException: public Exception
{
public:
    CallMethodException(const string &moduleName, const string &methodName)
        : Exception("An error occurred while calling the method " + methodName + " in the module " + moduleName)
    {}
};

class NotArrayException: public Exception
{
public:
    NotArrayException(const string &moduleName, const string &methodName)
        : Exception("The return value of method "+ methodName + " in module " + moduleName + " is not an array")
    {}
};

class WrongDimArrayException: public Exception
{
public:
    WrongDimArrayException(const string &moduleName, const string &methodName)
        : Exception("Return an array of method " + methodName+ " in module " + moduleName + " has the wrong dimension")
    {}
};

#endif // PYEXCEPTION_H
