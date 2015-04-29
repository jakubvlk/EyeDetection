//
//  Draw.h
//  EyeDetection
//
//  Created by Jakub Vlk on 20/03/15.
//
//

#ifndef __EyeDetection__Draw__
#define __EyeDetection__Draw__

#include <stdio.h>
#import "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


void draw(Mat &frame, const Rect &frameFace, const vector<Rect> &frameEyes);


#endif /* defined(__EyeDetection__Draw__) */
