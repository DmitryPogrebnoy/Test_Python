#pragma once

class Constants
{
public:
        static const int maxNumOfRobots = 12;
        static const int maxRobotsInTeam = maxNumOfRobots; //maxNumOfRobots / 2;
        static const int robotAlgoPacketSize = 4 * maxRobotsInTeam;
        static const int ballAlgoPacketSize = 3;
        static const unsigned  SSLVisionPort = 10006;
        static const unsigned  SimVisionPort = 10020;
        static const int numOfCameras = 4;
        static const int ruleLength = 7;
        static const int ruleAmount = 12;
        static const int matlabOutputBufferSize = 2048;
};
