//
//  eyeLidsLocalisation.h
//  EyeDetection
//
//  Created by Jakub Vlk
//
//

#ifndef __EyeDetection__eyeLidsLocalisation__
#define __EyeDetection__eyeLidsLocalisation__

#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void eyeLidsLocalisation(Mat &eye, string windowName, int windowX, int windowY, int frameX, int frameY, vector<Vec4f> &eyeLids);

void drawEyeLids(const vector<Vec4f> &pupils, Mat &eyeLids);

#endif /* defined(__EyeDetection__eyeLidsLocalisation__) */
