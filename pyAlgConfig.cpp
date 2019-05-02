#include "pyAlgConfig.hpp"
#include "parseConfigExcept.hpp"
#include <fstream>

using namespace std;
/* Пример конфигурации, которая спарсится:
 *
 * name=RoboFootball
 * namePyMainScript=main
 * namePyMainMethod=main
 * namePyPauseScript=pause
 * namePyPauseMethod=pause_unpause
 * CONTROL_SIGNALS_AMOUNT=4
 * CONTROL_SIGNALS_LENGTH=7
 *
 *
 * Парсится строго в этом порядке и строго в этом виде.
 * При неудачной попытке распарсить выставляются значения по умолчанию.
 */
pyAlgConfig::pyAlgConfig(){
    string line;
    ifstream configFile;
    configFile.exceptions(ifstream::failbit);
    try {
        configFile.open("pyAlgConfig.cnf");


        if (getline(configFile, line)) {
            if (line.find("name=") != string::npos){
                name = line.substr(line.find('=') + 1);
            } else throw parseConfigException("name=","pyAlgConfig.cnf");
        } else throw notFullConfigException("name", "pyAlgConfig.cnf");

        if (getline(configFile, line)) {
            if (line.find("namePyMainScript=") != string::npos){
                namePyMainScript = line.substr(line.find('=') + 1);
            } else throw parseConfigException("namePyMainScript=","pyAlgConfig.cnf");
        } else throw notFullConfigException("namePyMainScript", "pyAlgConfig.cnf");

        if (getline(configFile, line)) {
            if (line.find("namePyMainMethod=") != string::npos){
                namePyMainMethod = line.substr(line.find('=') + 1);
            } else throw parseConfigException("namePyMainMethod=","pyAlgConfig.cnf");
        } else throw notFullConfigException("namePyMainMethod", "pyAlgConfig.cnf");

        if (getline(configFile, line)) {
            if (line.find("namePyPauseScript=") != string::npos){
                namePyPauseScript = line.substr(line.find('=') + 1);
            } else throw parseConfigException("namePyPauseScript=","pyAlgConfig.cnf");
        } else throw notFullConfigException("namePyPauseScript", "pyAlgConfig.cnf");

        if (getline(configFile, line)) {
            if (line.find("namePyPauseMethod=") != string::npos){
                namePyPauseMethod = line.substr(line.find('=') + 1);
            } else throw parseConfigException("namePyPauseMethod=","pyAlgConfig.cnf");
        } else throw notFullConfigException("namePyPauseMethod", "pyAlgConfig.cnf");

        if (getline(configFile, line)) {
            if (line.find("CONTROL_SIGNALS_AMOUNT=") != string::npos){
                CONTROL_SIGNALS_AMOUNT = atoi(line.substr(line.find('=') + 1).c_str());
            } else throw parseConfigException("CONTROL_SIGNALS_AMOUNT=","pyAlgConfig.cnf");
        } else throw notFullConfigException("CONTROL_SIGNALS_AMOUNT", "pyAlgConfig.cnf");

        if (getline(configFile, line)) {
            if (line.find("CONTROL_SIGNALS_LENGTH=") != string::npos){
                CONTROL_SIGNALS_LENGTH = atoi(line.substr(line.find('=') + 1).c_str());
            } else throw parseConfigException("CONTROL_SIGNALS_LENGTH=","pyAlgConfig.cnf");
        } else throw notFullConfigException("CONTROL_SIGNALS_LENGTH", "pyAlgConfig.cnf");

        cout<<"name = "<<name<<endl
            <<"namePyMainScript = "<<namePyMainScript<<endl
            <<"namePyMainMethod = "<<namePyMainMethod<<endl
            <<"namePyPauseScript = "<<namePyPauseScript<<endl
            <<"namePyPauseMethod = "<<namePyPauseMethod<<endl
            <<"CONTROL_SIGNALS_AMOUNT = "<<CONTROL_SIGNALS_AMOUNT<<endl
            <<"CONTROL_SIGNALS_LENGTH = "<<CONTROL_SIGNALS_LENGTH<<endl
            <<"Config file successfully parsed."<<endl;
    } catch (const ifstream::failure) {
        name = "Robofootball";
        namePyMainScript = "main";
        namePyMainMethod = "main";
        namePyPauseScript = "pause";
        namePyPauseMethod = "pause_unpause";
        CONTROL_SIGNALS_AMOUNT = 6;
        CONTROL_SIGNALS_LENGTH = 6;

        cerr<<"File 'pyAlgConfig.cnf' is not found."<<endl
            <<"The default values are set:"<<endl
            <<"name = "<<name<<endl
            <<"namePyMainScript = "<<namePyMainScript<<endl
            <<"namePyMainMethod = "<<namePyMainMethod<<endl
            <<"namePyPauseScript = "<<namePyPauseScript<<endl
            <<"namePyPauseMethod = "<<namePyPauseMethod<<endl
            <<"CONTROL_SIGNALS_AMOUNT = "<<CONTROL_SIGNALS_AMOUNT<<endl
            <<"CONTROL_SIGNALS_LENGTH = "<<CONTROL_SIGNALS_LENGTH<<endl;
    } catch (const parseException e) {
        name = "Robofootball";
        namePyMainScript = "main";
        namePyMainMethod = "main";
        namePyPauseScript = "pause";
        namePyPauseMethod = "pause_unpause";
        CONTROL_SIGNALS_AMOUNT = 6;
        CONTROL_SIGNALS_LENGTH = 6;

        cerr<<e.message()<<endl
            <<"The default values are set:"<<endl
            <<"name = "<<name<<endl
            <<"namePyMainScript = "<<namePyMainScript<<endl
            <<"namePyMainMethod = "<<namePyMainMethod<<endl
            <<"namePyPauseScript = "<<namePyPauseScript<<endl
            <<"namePyPauseMethod = "<<namePyPauseMethod<<endl
            <<"CONTROL_SIGNALS_AMOUNT = "<<CONTROL_SIGNALS_AMOUNT<<endl
            <<"CONTROL_SIGNALS_LENGTH = "<<CONTROL_SIGNALS_LENGTH<<endl;
    }
}
pyAlgConfig::pyAlgConfig(const pyAlgConfig &conf) : name(conf.name),
                                                    namePyMainScript(conf.namePyMainScript),
                                                    namePyMainMethod(conf.namePyMainMethod),
                                                    namePyPauseScript(conf.namePyPauseScript),
                                                    namePyPauseMethod(conf.namePyPauseMethod),
                                                    CONTROL_SIGNALS_AMOUNT(conf.CONTROL_SIGNALS_AMOUNT),
                                                    CONTROL_SIGNALS_LENGTH(conf.CONTROL_SIGNALS_LENGTH){

}
//Геттеры для полей класса
string pyAlgConfig::get_name() const {
    return name;
}
string pyAlgConfig::get_namePyMainScript() const {
    return namePyMainScript;
}
string pyAlgConfig::get_namePyMainMethod() const {
    return namePyMainMethod;
}
string pyAlgConfig::get_namePyPauseScript() const {
    return namePyPauseScript;
}
string pyAlgConfig::get_namePyPauseMethod() const {
    return namePyPauseMethod;
}
int pyAlgConfig::get_CONTROL_SIGNALS_AMOUNT() const {
    return CONTROL_SIGNALS_AMOUNT;
}
int pyAlgConfig::get_CONTROL_SIGNALS_LENGTH() const {
    return CONTROL_SIGNALS_LENGTH;
}
