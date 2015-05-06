//
//  processArguments.cpp
//  EyeDetection
//
//  Created by Jakub Vlk
//
//

#include "processArguments.h"

#include <fstream>


int processArguments( int argc, const char** argv, string &file, bool &useVideo, bool &stepFrame, bool &showWindow, bool &useCamera )
{
    cout << argc << endl;
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if ((arg == "-h") || (arg == "--help"))
        {
            showUsage(argv[0]);
            return 2;
        }
        else if ((arg == "-f") || (arg == "--file"))
        {
            if (i + 1 < argc)
            {
                file = argv[++i];
                
                // just image
                if (argc == 3)
                {
                    useVideo = false;
                }
            }
            else
            {
                cerr << "--file option requires one argument." << endl;
                return -1;
            }
        }
        else if ((arg == "-v") || (arg == "--video"))
        {
            useVideo = true;
        }
        else if ((arg == "-s") || (arg == "--step"))
        {
            stepFrame = true;
            showWindow = true;
        }
        else if ((arg == "-c") || (arg == "--camera"))
        {
            useCamera = true;
        }
    }
    
    return 0;
}

void showUsage( string name )
{
    cerr << "Usage: " << name << endl
    << "Options:" << endl
    << "\t-h,--help\t\tShow this help message. Example: -h" << endl
    
    << "File:\n"
    << "\t-f,--file\t\tName of the file. If no other parameter, image is expected. For more information about video search -v in this help. Example: -f myImage.png" << endl
    
    << "Video:\n"
    << "\t-v,--video\t\tUse file name and -v for video. Example: -f myVideo.avi -v" << endl
    
    << "Step:\n"
    << "\t-s,--step\t\tUse step function for ability to move forward in video by key 'n'. Example: -f myVideo.avi -v -s You can also pause video while playing by pressing key 'p'." << endl
    
    << "Camera:\n"
    << "\t-c,--camera\t\tUse camera for streaming video from camera. Example: -c" << endl << endl
    
    << "While playing video you can use key 'p' for pause and play, 'n' for next frame in paused video and 'f' for disable drawing of eye parts"
    
    << endl;
}