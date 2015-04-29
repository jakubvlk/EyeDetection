//
//  eyeDetection.h
//  EyeDetection
//
//  Created by Jakub Vlk on 19/03/15.
//
//

#ifndef __EyeDetection__eyeDetection__
#define __EyeDetection__eyeDetection__

#include <stdio.h>
#import "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


vector<Rect> eyeDetection(Mat &face, const Rect &faceRect, const Mat &originalFrame, vector<Rect> &frameEyes);
void loadEyeCascade();

#endif /* defined(__EyeDetection__eyeDetection__) */
