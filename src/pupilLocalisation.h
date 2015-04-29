//
//  pupilLocalisation.h
//  EyeDetection
//
//  Created by Jakub Vlk on 29/04/15.
//
//

#ifndef __EyeDetection__pupilLocalisation__
#define __EyeDetection__pupilLocalisation__

#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void pupilLocalisation(Mat &eye, int kernel, string windowName, int windowX, int windowY, int frameX, int frameY, Point center, vector<Vec3f> &pupils);

void drawPupils(const vector<Vec3f> &pupils, Mat &frame);

#endif /* defined(__EyeDetection__pupilLocalisation__) */
