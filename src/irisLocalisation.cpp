//
//  irisLocalisation.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 29/04/15.
//
//

#include "irisLocalisation.h"


#import "functions.h"
#import "constants.h"

Point irisLocalisation( Mat &eye, int kernel, string windowName, int windowX, int windowY, int frameX, int frameY, Point center, vector<Vec3f> &irises)
{

#if TIME_MEASURING
    int64 e1 = getTickCount();
#endif
    
    Mat gaussEye;
    GaussianBlur( eye, gaussEye, Size(5, 5), 0, 0, BORDER_DEFAULT );
    
    Mat intensiveEye(eye.rows, eye.cols, CV_8U);
    intenseMul(gaussEye, intensiveEye, 4);
    
    showWindowAtPosition( windowName + "intensiveEye eye hough", intensiveEye, windowX, windowY );
    
    // Gradient
    Mat gradX;
    Mat absGradX;
    
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    /// Gradient X
    Sobel( intensiveEye, gradX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    
    convertScaleAbs( gradX, absGradX );
    
    
    absGradX = mat2gray(absGradX);
    
    // radiuses
    int minRadius = lround(eye.size().width * 0.1), maxRadius = eye.size().width * 0.3; //4
    
    int gradientsCount = maxRadius - minRadius + 1;
    double gradients[kernel][kernel][gradientsCount];
    
    for (int i = 0; i < kernel; ++i)
    {
        for (int j = 0; j < kernel; ++j)
        {
            for (int k = 0; k < gradientsCount; ++k)
            {
                gradients[i][j][k] = 0;
            }
        }
    }
    
    int halfKernel = (kernel - 1) * 0.5;
    // border
    int xMin = center.x - halfKernel;
    if (xMin < 0)
        xMin = 0;
    int xMax = center.x + halfKernel;
    if (xMax > eye.size().width)
        xMax = eye.size().width;
    int yMin = center.y - halfKernel;
    if (yMin < 0)
        yMin = 0;
    int yMax = center.y + halfKernel;
    if (yMax > eye.size().height)
        yMax = eye.size().height;
    
    if (kernel == 1)
    {
        xMin = xMax = center.x;
        yMin = yMax = center.y;
        
    }
    
    double m_pi180 = M_PI / 180;
    int i = 0;
    for (int x = xMin; x <= xMax; ++x)
    {
        int j = 0;
        for (int y = yMin; y <= yMax; ++y)
        {
            int k = 0;
            for (int r = minRadius; r <= maxRadius; ++r)
            {
                double step = 2* M_PI / (r*2);
                
                int stepsCount = 0;
                for(double theta = 120 * m_pi180;  theta <= 240 * m_pi180;  theta += step)
                {
                    int circleX = lround(x + r * cos(theta));
                    int circleY = lround(y - r * sin(theta));
                    
                    gradients[i][j][k] += absGradX.at<uchar>(circleY, circleX);
                    
                    stepsCount++;
                }
                for(double theta = -60 * m_pi180;  theta <= 60 * m_pi180;  theta += step)
                {
                    int circleX = lround(x + r * cos(theta));
                    int circleY = lround(y - r * sin(theta));
                    
                    gradients[i][j][k] += absGradX.at<uchar>(circleY, circleX);
                    
                    stepsCount++;
                }
                
                gradients[i][j][k] /= stepsCount;
                
                k++;
            }
            
            j++;
        }
        
        i++;
    }
    
    
    double maxGrad = 0;
    double maxGradRad = 0;
    
    Point newCenter = center;
    
    i = 0;
    for (int x = xMin; x <= xMax; ++x)
    {
        int j = 0;
        for (int y = yMin; y <= yMax; ++y)
        {
            for (int k = 0; k < gradientsCount; ++k)
            {
                if (gradients[i][j][k] > maxGrad)
                {
                    maxGrad = gradients[i][j][k];
                    maxGradRad = k;
                    
                    newCenter.x = x;
                    newCenter.y = y;
                }
            }
            
            j++;
        }
        
        i++;
    }
    
    
    maxGradRad += minRadius;
    
    irises.push_back(Vec3f(newCenter.x + frameX, newCenter.y + frameY, maxGradRad));
    
    
#if TIME_MEASURING
    double time = (getTickCount() - e1)/ getTickFrequency();
    cout << "My hough circle time = " << time << endl;
#endif

    return newCenter;
}

void drawIrises(const vector<Vec3f> &irises, Mat &frame)
{
    for( size_t i = 0; i < irises.size(); i++ )
    {
        int radius = cvRound(irises[i][2]);
        
        Point frameCenter( cvRound(irises[i][0]), cvRound(irises[i][1]) );
        // circle outline
        Scalar color = Scalar(255, 0, 255);
        
        circle( frame, frameCenter, radius, color);
    }
}