#ifndef PYEXCEPTION_H
#define PYEXCEPTION_H

#include "except.hpp"

class pyException: public Exception
{
public:
    explicit pyException(const string &string) : Exception(string)
    {}
};

class DecodeException: public pyException
{
public:
    explicit DecodeException(const string &string) : pyException("Could not decode string: " + string + ".")
    {}
};


class ModuleNotFoundException: public pyException
{
public:
    explicit ModuleNotFoundException(const string &moduleName) : pyException("Could not found Python module '" + moduleName + "'.")
    {}
};

class MethodNotFoundException: public pyException
{
public:
    explicit MethodNotFoundException(const string &moduleName, const string &methodName)
        : pyException("Could not found method '" + methodName + "' in Python module '" + moduleName + "'.")
    {}
};

class CreateTupleException: public pyException
{
public:
    explicit CreateTupleException(const int lengthTuple)
        : pyException("Failed to create a tuple of length " + to_string(lengthTuple) +".")
    {}
};


class CallMethodException: public pyException
{
public:
    explicit CallMethodException(const string &moduleName, const string &methodName)
        : pyException("An error occurred while calling the method '" + methodName + "' in the module '" + moduleName + "'.")
    {}
};

class NotArrayException: public pyException
{
public:
    explicit NotArrayException(const string &arrayName)
        : pyException("'" + arrayName + "' is not an array")
    {}
};

class WrongSizeArrayException: public pyException
{
public:
    explicit WrongSizeArrayException(const string &arrayName, const int &expectedValue, const int &currentValue)
        : pyException("'" + arrayName + "' array has the wrong size. Expected value: "
                    + to_string(expectedValue) + ". Current value: " + to_string(currentValue) + ".")
    {}
};

class CouldNotCastToDouble: public pyException
{
public:
    explicit CouldNotCastToDouble()
        : pyException("Could not cast to double.")
    {}
};


class CouldNotSetItemTupleException: public pyException
{
public:
    explicit CouldNotSetItemTupleException(const string &tupleName, const int &itemNumber, const string &valueName)
        : pyException("Could not set '" + valueName + "' value as item " + to_string(itemNumber) + " of '" + tupleName + "' tuple.")
    {}
};

#endif // PYEXCEPTION_H
