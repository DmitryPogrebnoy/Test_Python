#ifndef PYEXCEPTION_H
#define PYEXCEPTION_H

#include "except.h"

class pyException : public Exception
{
public:
    explicit pyException(const QString &str) : Exception(str)
    {}
};

class DecodeException : public pyException
{
public:
    explicit DecodeException(const QString &str)
        : pyException(QString("Could not decode string: %1.").arg(str))
    {}
};

class EncodeStreamException : public pyException
{
public:
    explicit EncodeStreamException(const QString &str)
        : pyException(QString("Could not encode %1.").arg(str))
    {}
};


class ObjectNotFoundException : public pyException
{
public:
    explicit ObjectNotFoundException(const QString &objectName)
        : pyException(QString("Could not found Python module '%1'.").arg(objectName))
    {}
};

class AttrNotFoundException : public pyException
{
public:
    explicit AttrNotFoundException(const QString &objectName, const QString &methodName)
        : pyException(QString("Could not found attribute '%1' in Python object '%2'.").arg(methodName, objectName))
    {}
};

class CreateTupleException : public pyException
{
public:
    explicit CreateTupleException(const int lengthTuple)
        : pyException(QString("Failed to create a tuple of length %1.").arg(QString::number(lengthTuple)))
    {}
};


class CallMethodException : public pyException
{
public:
    explicit CallMethodException(const QString &objectName, const QString &methodName)
        : pyException(QString("An error occurred while calling the method '%1' of the object '%2'.").arg(methodName, objectName))
    {}
};

class NotBoolReturnValueException : public pyException
{
public:
    explicit NotBoolReturnValueException(const QString &objectName, const QString &methodName)
        : pyException(QString("The returned value of the method '%1' object '%2' is not a bool").arg(methodName, objectName))
    {}
};

class NotArrayReturnValueException : public pyException
{
public:
    explicit NotArrayReturnValueException(const QString &objectName, const QString &methodName)
        : pyException(QString("The returned value of the method '%1' object '%2' not an array").arg(methodName, objectName))
    {}
};

class NotArrayReturnValueElementException : public pyException
{
public:
    explicit NotArrayReturnValueElementException(const int elementPosition)
        : pyException(QString("The %1 element of the returned array from the Python script is not an array.").arg(QString::number(elementPosition)))
    {}
};

class WrongSizeArrayReturnValueException : public pyException
{
public:
    explicit WrongSizeArrayReturnValueException(const int expectedValue, const int currentValue)
        : pyException(QString(
                          "The returned array from the Python script has the wrong size. Expected value: %1. Current value: %2."
                          ).arg(QString::number(expectedValue), QString::number(currentValue)))
    {}
};

class WrongSizeArrayReturnValueElementException : public pyException
{
public:
    explicit WrongSizeArrayReturnValueElementException(const int elementPosition,const int expectedValue, const int currentValue)
        : pyException(QString(
                          "The %1 element of the returned array from the Python script has the wrong size. Expected value: %2. Current value: %3."
                          ).arg(QString::number(elementPosition), QString::number(expectedValue), QString::number(currentValue)))
    {}
};

class CouldNotCastToDouble : public pyException
{
public:
    explicit CouldNotCastToDouble(const int numberRule, const int numberPosition)
        : pyException(QString("Could not cast (%1,%2) element of returned array to double type."
                              ).arg(QString::number(numberRule), QString::number(numberPosition)))
    {}
};


class CouldNotSetItemTupleException : public pyException
{
public:
    explicit CouldNotSetItemTupleException(const QString &tupleName, const int &itemNumber, const QString &valueName)
        : pyException(QString("Could not set '%1' value as item %2 of '%3' tuple.").arg(valueName, QString::number(itemNumber), tupleName))
    {}
};

#endif // PYEXCEPTION_H
