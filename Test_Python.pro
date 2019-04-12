QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
        main.cpp \
    pyMainAlgWorker.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES +=

HEADERS += \
    pyMainAlgWorker.hpp \
    pyScopedPointerDeleter.hpp \
    pyRunner.hpp \
    pyExcept.hpp


message($$(PYTHONLIB))
message($$(PYTHONINCLUDE))
win32:CONFIG(release, debug|release): LIBS += -L$$(PYTHONLIB)/ -lpython37
else:win32:CONFIG(debug, debug|release): LIBS += -L$$(PYTHONLIB)/ -lpython37d
else:unix: LIBS += -L$$(PYTHONLIB)/ -lpython3.7m

INCLUDEPATH += $$(PYTHONINCLUDE)
DEPENDPATH += $$(PYTHONINCLUDE)

#Бага Qt креатора:  редактор не отображает, что подключена библиотека
#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../AppData/Local/Programs/Python/Python37/libs/ -lpython37
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../AppData/Local/Programs/Python/Python37/libs/ -lpython37d
#else:unix: LIBS += -L$$PWD/../../AppData/Local/Programs/Python/Python37/libs/ -lpython37

#INCLUDEPATH += $$PWD/../../AppData/Local/Programs/Python/Python37/include
#DEPENDPATH += $$PWD/../../AppData/Local/Programs/Python/Python37/include

message($$(NUMPYLIB))
message($$(NUMPYINCLUDE))
win32:CONFIG(release, debug|release): LIBS += -L$$(NUMPYLIB)/ -lnpymath
else:win32:CONFIG(debug, debug|release): LIBS += -L$$(NUMPYLIB)/ -lnpymathd
else:unix:!macx: LIBS += -L$$(NUMPYLIB)/ -lnpymath

INCLUDEPATH += $$(NUMPYINCLUDE)
DEPENDPATH += $$(NUMPYINCLUDE)
