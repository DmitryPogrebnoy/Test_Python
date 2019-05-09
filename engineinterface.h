#pragma once
#include <QObject>
//#include "sharedRes.h"

typedef struct Rule
{
    int mSpeedX = 0;
    int mSpeedY = 0;
    int mSpeedR = 0;

    int mSpeedDribbler = 0;
    int mDribblerEnable = 0;

    int mKickerVoltageLevel = 12;
    int mKickerChargeEnable = 1;
    int mKickUp = 0;
    int mKickForward = 0;

    int mBeep = 0;
    int bs_kick_state = 0;
} Rule;


class EngineInterface : public QObject
{
    Q_OBJECT

public:
    //EngineInterface(SharedRes * sharedRes) : mSharedRes(sharedRes){}
    EngineInterface() {}
    virtual ~EngineInterface(){}

    /// Calls the main script of a particular engine
    virtual void evaluate() = 0;

    /// Stop and continue evaluate from GUI (most likely from a Worker)
    virtual void pauseUnpause() = 0;

    /// Setting the path where the main script is located
    virtual void setDirectory(const QString & path) = 0;

protected:
    //SharedRes * mSharedRes;
    bool mIsPause { false };

signals:

    /// Internal errors of the engine (for them, a separate field is implied)
    void algoStatus(const QString & message);

    /// Algorithm execution state changed - pause / calculated
    void isPause(bool status);

    /// Output to the console all print from scripts + internal errors (stdout + stderr)
    void consoleMessage(const QString & message);

    /// Sending new computed control signals
    void newData(const QVector<Rule> & rule);
};
