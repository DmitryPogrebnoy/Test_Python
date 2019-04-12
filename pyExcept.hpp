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
    explicit DecodeException(const string &string) : Exception("Could not decode string: " + string + ".")
    {}
};


class ModuleNotFoundException: public Exception
{
public:
    explicit ModuleNotFoundException(const string &moduleName) : Exception("Could not found Python module '" + moduleName + "'.")
    {}
};

class MethodNotFoundException: public Exception
{
public:
    MethodNotFoundException(const string &moduleName, const string &methodName)
        : Exception("Could not found method '" + methodName + "' in Python module '" + moduleName + "'.")
    {}
};

class CallMethodException: public Exception
{
public:
    CallMethodException(const string &moduleName, const string &methodName)
        : Exception("An error occurred while calling the method '" + methodName + "' in the module '" + moduleName + "'.")
    {}
};

class NotArrayException: public Exception
{
public:
    NotArrayException(const string &arrayName)
        : Exception("'" + arrayName + "' is not an array")
    {}
};

class WrongSizeArrayException: public Exception
{
public:
    WrongSizeArrayException(const string &arrayName, const int &expectedValue, const int &currentValue)
        : Exception("'" + arrayName + "' array has the wrong size. Expected value: "
                    + to_string(expectedValue) + ". Current value: " + to_string(currentValue) + ".")
    {}
};

class WrongDimArrayException: public Exception
{
public:
    WrongDimArrayException(const string &arrayName, const int &expectedValue, const int &currentValue)
        : Exception("'" + arrayName + "' array has the wrong dimension. Expected value: "
                    + to_string(expectedValue) + ". Current value: " + to_string(currentValue) + ".")
    {}
};

class NotTupleException: public Exception
{
public:
    NotTupleException(const string &tupleName)
        : Exception("'" + tupleName + "' is not a tuple")
    {}
};

class WrongSizeTupleException: public Exception
{
public:
    WrongSizeTupleException(const string &tupleName, const int &expectedValue, const int &currentValue)
        : Exception("'" + tupleName + "' tuple has the wrong size. Expected value: " + to_string(expectedValue)
                    + ". Current value: " + to_string(currentValue) + ".")
    {}
};

class CouldNotSetItemTupleException: public Exception
{
public:
    CouldNotSetItemTupleException(const string &tupleName, const int &itemNumber, const string &valueName)
        : Exception("Could not set '" + valueName + "' value as item " + to_string(itemNumber) + " of '" + tupleName + "' tuple.")
    {}
};

#endif // PYEXCEPTION_H
