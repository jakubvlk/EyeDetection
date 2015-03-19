//
//  detection.h
//  EyeDetection
//
//  Created by Jakub Vlk on 19/03/15.
//
//

#ifndef __EyeDetection__detection__
#define __EyeDetection__detection__

#import <stdio.h>
#import <iostream>
#import "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void initDetection();
void detectAndFind(Mat &frame);


#endif /* defined(__EyeDetection__detection__) */
