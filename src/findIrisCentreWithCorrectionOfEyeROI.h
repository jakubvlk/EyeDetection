//
//  findIrisCentreWithCorrectionOfEyeROI.h
//  EyeDetection
//
//  Created by Jakub Vlk
//
//

#ifndef __EyeDetection__findIrisCentreWithCorrectionOfEyeROI__
#define __EyeDetection__findIrisCentreWithCorrectionOfEyeROI__

#include <stdio.h>


#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;

Point findIrisCentreWithCorrectionOfEyeROI( Mat &frame, Mat &eye, string windowName, int windowX, int windowY, int frameX, int frameY, vector<Point> &eyesCentres );

float avgIntensity(Mat mat, int x, int y, int width, int height, int maxIntensity);
int blackPixelsCount(Mat mat);

#endif /* defined(__EyeDetection__findIrisCentreWithCorrectionOfEyeROI__) */
