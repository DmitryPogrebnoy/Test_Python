#ifndef RCCONFIG_H
#define RCCONFIG_H

#include <iostream>



class pyAlgConfig
{
public:
    pyAlgConfig();
    pyAlgConfig(const pyAlgConfig &conf);

    std::string get_name() const;
    std::string get_namePyMainScript() const;
    std::string get_namePyMainMethod() const;
    std::string get_namePyPauseScript() const;
    std::string get_namePyPauseMethod() const;
    int get_CONTROL_SIGNALS_AMOUNT() const;
    int get_CONTROL_SIGNALS_LENGTH() const;

private:
    std::string name;
    std::string namePyMainScript;
    std::string namePyMainMethod;
    std::string namePyPauseScript;
    std::string namePyPauseMethod;
    int CONTROL_SIGNALS_AMOUNT;
    int CONTROL_SIGNALS_LENGTH;

};

#endif // RCCONFIG_H
