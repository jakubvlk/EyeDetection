//
//  eyeCentreLocalisationByMeansOfGradients.h
//  EyeDetection
//
//  Created by Jakub Vlk
//
//

#ifndef __EyeDetection__eyeCentreLocalisationByMeansOfGradients__
#define __EyeDetection__eyeCentreLocalisationByMeansOfGradients__

#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


Point eyeCentreLocalisationByMeansOfGradients( Mat &eye, string windowName, int windowX, int windowY, int frameX, int frameY, vector<Point> &eyesCentres );

Mat resizeMat(Mat src, int width);
Point unscalePoint(Point p, int origWidth, int width);
Point uncut(Point p, int cut);

void drawEyesCentres(const vector<Point> &eyesCentres, Mat &frame);

#endif /* defined(__EyeDetection__eyeCentreLocalisationByMeansOfGradients__) */
