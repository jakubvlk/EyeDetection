//
//  faceDetection.h
//  EyeDetection
//
//  Created by Jakub Vlk on 19/03/15.
//
//

#ifndef __EyeDetection__faceDetection__
#define __EyeDetection__faceDetection__

#include <stdio.h>
#import "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


Mat faceDetection(Mat &frame);
void loadFaceCascade();


#endif /* defined(__EyeDetection__faceDetection__) */
