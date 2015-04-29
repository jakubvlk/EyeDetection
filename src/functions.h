//
//  functions.h
//  EyeDetection
//
//  Created by Jakub Vlk on 28/04/15.
//
//

#ifndef __EyeDetection__functions__
#define __EyeDetection__functions__

#include <stdio.h>

#include "opencv2/opencv.hpp"

using namespace cv;

void intenseMul( Mat &src, Mat &dst, int multiplier );
void showWindowAtPosition( string imageName, Mat &mat, int x, int y );
Mat mat2gray(const Mat &src);

#endif /* defined(__EyeDetection__functions__) */
