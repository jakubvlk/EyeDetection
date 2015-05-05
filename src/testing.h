//
//  testing.h
//  EyeDetection
//
//  Created by Jakub Vlk on 30/04/15.
//
//

#ifndef __EyeDetection__testing__
#define __EyeDetection__testing__

#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void testEyeCenterDetection(void (*fptr_detectAndShow)(void*, Mat frame), void* context, Mat &frame, Mat &originalFrame, const vector<Point> &eyesCentres);

void testIrisDetection(void (*fptr_detectAndShow)(void*, Mat frame), void* context, Mat &frame, Mat &originalFrame, const vector<Vec3f> &irises);

void testLidsDetectionvoid (void (*fptr_detectAndShow)(void*, Mat frame), void* context, Mat &frame, Mat &originalFrame, const vector<Vec4f> &eyeLids);




int digitsCount(int x);

void somefunction(void (*fptr)(void*, Mat frame), void* context, Mat frame);

void readEyeData(string fullEyeDataFilePath, vector<Point> &dataEyeCentres);

void computeEyeCentreDistances(Point dataLeftEye, Point dataRigtEye, Point myLeftEye, Point myRightEye, vector<double> &eyeCentreDistances);

double getNormalisedError(const vector<double> &eyeCentreDistances, double e);

Vec6f readMyEyeData(string fullEyeDataFilePath);

void computeIrisesDistances(Point dataLeftEye, Point dataRigtEye, double myLeftEyeCentreIrisDistance, double myRightEyeCentreIrisDistance, double dataLeftEyeCentreIrisDistance, double dataRightEyeCentreIrisDistance,vector<double> &irisesDistances);

void computeLidsDistances(Point dataLeftEye, Point dataRigtEye, const vector<Vec4f> &myLids, const Vec6f eyeData, vector<double> &irisesDistances);

#endif /* defined(__EyeDetection__testing__) */
