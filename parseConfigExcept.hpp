#ifndef PARSECONFIGEXCEPT_H
#define PARSECONFIGEXCEPT_H

#include "except.hpp"

class parseException: public Exception
{
public:
    explicit parseException(const string &massage) :
        Exception(massage)
    {}
};

class parseConfigException: public parseException
{
public:
    explicit parseConfigException(const string &substr, const string &filename) :
        parseException("Could not found '" + substr + "' in '" + filename + "'.")
    {}
};

class notFullConfigException: public parseException
{
public:
    explicit notFullConfigException(const string &var, const string &filename) :
        parseException("Not full config file '" + filename + "'. The value of '" + var + "' is not defined.")
    {}
};
#endif // PARSECONFIGEXCEPT_H
