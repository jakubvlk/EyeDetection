//
//  functions.h
//  EyeDetection
//
//  Created by Jakub Vlk
//
//

#ifndef __EyeDetection__functions__
#define __EyeDetection__functions__

#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void intenseMul( Mat &src, Mat &dst, int multiplier );
void showWindowAtPosition( string imageName, Mat &mat, int x, int y );
Mat mat2gray(const Mat &src);
vector<Rect> sortEyes(const vector<Rect> &eyes, Rect face);
Rect pickFace(vector<Rect> faces);
vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face);
Mat removeReflections(Mat &eye, string windowName, int x, int y, int frameX, int frameY);


#endif /* defined(__EyeDetection__functions__) */
