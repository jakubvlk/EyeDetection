//
//  removeReflections.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 29/04/15.
//
//

#include "removeReflections.h"

#import "functions.h"
#import "constants.h"

Mat removeReflections(Mat &eye, string windowName, int x, int y, int frameX, int frameY)
{
    Mat gaussEye, binaryEye, eyeWihoutReflection;
    
    GaussianBlur( eye, gaussEye, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    Mat gradX, gradY;
    Mat absGradX, absGradY, grad;
    
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    // Gradient X
    Sobel( gaussEye, gradX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    // Gradient Y
    Sobel( gaussEye, gradY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    
    convertScaleAbs( gradX, absGradX );
    convertScaleAbs( gradY, absGradY );
    
    addWeighted( absGradX, 0.5, absGradY, 0.5, 0, grad );
    
    threshold(grad, binaryEye, 91, 255, CV_THRESH_BINARY);
    
    inpaint(eye, binaryEye, eyeWihoutReflection, 3, INPAINT_TELEA);
    
    return eyeWihoutReflection;
}