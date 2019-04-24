#ifndef PYEXCEPTION_H
#define PYEXCEPTION_H

#include "except.hpp"

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
    explicit MethodNotFoundException(const string &moduleName, const string &methodName)
        : Exception("Could not found method '" + methodName + "' in Python module '" + moduleName + "'.")
    {}
};

class CreateTupleException: public Exception
{
public:
    explicit CreateTupleException(const int lengthTuple)
        : Exception("Failed to create a tuple of length " + to_string(lengthTuple) +".")
    {}
};


class CallMethodException: public Exception
{
public:
    explicit CallMethodException(const string &moduleName, const string &methodName)
        : Exception("An error occurred while calling the method '" + methodName + "' in the module '" + moduleName + "'.")
    {}
};

class NotArrayException: public Exception
{
public:
    explicit NotArrayException(const string &arrayName)
        : Exception("'" + arrayName + "' is not an array")
    {}
};

class WrongSizeArrayException: public Exception
{
public:
    explicit WrongSizeArrayException(const string &arrayName, const int &expectedValue, const int &currentValue)
        : Exception("'" + arrayName + "' array has the wrong size. Expected value: "
                    + to_string(expectedValue) + ". Current value: " + to_string(currentValue) + ".")
    {}
};

class CouldNotCastToDouble: public Exception
{
public:
    explicit CouldNotCastToDouble()
        : Exception("Could not cast to double.")
    {}
};




class CouldNotSetItemTupleException: public Exception
{
public:
    explicit CouldNotSetItemTupleException(const string &tupleName, const int &itemNumber, const string &valueName)
        : Exception("Could not set '" + valueName + "' value as item " + to_string(itemNumber) + " of '" + tupleName + "' tuple.")
    {}
};

#endif // PYEXCEPTION_H
