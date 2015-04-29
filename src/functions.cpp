//
//  functions.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 28/04/15.
//
//

#include "functions.h"



void intenseMul( Mat &src, Mat &dst, int multiplier )
{
    for (int i = 0; i < dst.cols; i++)
    {
        for (int j = 0; j < dst.rows; j++)
        {
            int intensity = src.at<uchar>(j, i) * multiplier;
            if (intensity > 255)
                intensity = 255;
            
            dst.at<uchar>(j, i) = intensity;
        }
    }
}

void showWindowAtPosition( string imageName, Mat &mat, int x, int y )
{
    imshow( imageName, mat );
    moveWindow(imageName, x, y);
}

Mat mat2gray(const Mat &src)
{
    Mat dst;
    normalize(src, dst, 0.0, 255.0, NORM_MINMAX, CV_8U);
    
    return dst;
}