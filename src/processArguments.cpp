//
//  processArguments.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 05/05/15.
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
            return 0;
        }
        else if ((arg == "-f") || (arg == "--file"))
        {
            if (i + 1 < argc)
            {
                file = argv[++i];
            }
            else
            {
                cerr << "--file option requires one argument." << endl;
                return 1;
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
    cerr << "Usage: " << name << " <option(s)> SOURCES"
    << "Options:\n"
    << "\t-h,--help\t\tShow this help message\n"
    << endl;
}