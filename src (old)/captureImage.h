//
//  captureImage.h
//  EyeDetection
//
//  Created by Jakub Vlk on 19/03/15.
//
//

#ifndef __EyeDetection__captureImage__
#define __EyeDetection__captureImage__

//#import <stdio.h>
///#import <iostream>

#import "opencv2/opencv.hpp"

//using namespace std;
using namespace cv;


CvCapture* startCapture(string file, bool useVideo, bool useCamera);



//string windowName = "Eye Detection";

#endif /* defined(__EyeDetection__captureImage__) */

