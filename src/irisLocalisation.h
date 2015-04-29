//
//  irisLocalisation.h
//  EyeDetection
//
//  Created by Jakub Vlk on 29/04/15.
//
//

#ifndef __EyeDetection__irisLocalisation__
#define __EyeDetection__irisLocalisation__

#include <stdio.h>

#include "opencv2/opencv.hpp"

#include <iostream>

#include <fstream>

using namespace std;
using namespace cv;

Point irisLocalisation( Mat &eye, int kernel, string windowName, int windowX, int windowY, int frameX, int frameY, Point center, vector<Vec3f> &irises);

void drawIrises(const vector<Vec3f> &irises, Mat &frame);

#endif /* defined(__EyeDetection__irisLocalisation__) */
