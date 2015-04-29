//
//  removeReflections.h
//  EyeDetection
//
//  Created by Jakub Vlk on 29/04/15.
//
//

#ifndef __EyeDetection__removeReflections__
#define __EyeDetection__removeReflections__

#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat removeReflections(Mat &eye, string windowName, int x, int y, int frameX, int frameY);

#endif /* defined(__EyeDetection__removeReflections__) */
