//
//  processArguments.h
//  EyeDetection
//
//  Created by Jakub Vlk on 05/05/15.
//
//

#ifndef __EyeDetection__processArguments__
#define __EyeDetection__processArguments__

#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


int processArguments( int argc, const char** argv, string &file, bool &useVideo, bool &stepFrame, bool &showWindow, bool &useCamera );
void showUsage( string name );


#endif /* defined(__EyeDetection__processArguments__) */
