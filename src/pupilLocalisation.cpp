//
//  pupilLocalisation.cpp
//  EyeDetection
//
//  Created by Jakub Vlk
//
//

#include "pupilLocalisation.h"

#import "functions.h"
#import "constants.h"

void pupilLocalisation(Mat &eye, int kernel, string windowName, int windowX, int windowY, int frameX, int frameY, Point center, vector<Vec3f> &pupils)
{
    
#if TIME_MEASURING
    int64 e1 = getTickCount();
#endif
    
    Mat gaussEye;
    GaussianBlur( eye, gaussEye, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    Mat intensiveEye(eye.rows, eye.cols, CV_8U);
    intenseMul(gaussEye, intensiveEye, 6);
    
    int minRadius = 3;
    int maxRadius = 7;
    
    int intensitiesCount = maxRadius - minRadius + 1;
    double intensities[intensitiesCount];
    for (int i = 0; i < intensitiesCount; i++)
    {
        intensities[i] = 0;
    }
    
    int i = 0;
    int stepsCount = 0;
    double totalIntensity = 0;
    for (int r = minRadius; r <= maxRadius; ++r)
    {
        double step = 2 * M_PI / ( r * 2 );
        
        for(double theta = 0;  theta < 2 * M_PI;  theta += step)
        {
            int circleX = lround(center.x + r * cos(theta));
            int circleY = lround(center.y - r * sin(theta));
            
            int pixelIntens = intensiveEye.at<uchar>(circleY, circleX);
            if (pixelIntens < 250)
            {
                totalIntensity += pixelIntens;
                
                stepsCount++;
            }
        }
        
        if (stepsCount == 0)
            stepsCount = 1;
        intensities[i] = totalIntensity / (r * stepsCount);

        i++;
    }
    
    double minIntens = 0;
    double minIntensRad = 0;
    for (i = 0; i < intensitiesCount; i++)
    {
        if (intensities[i] > minIntens)
        {
            minIntens = intensities[i];
            minIntensRad = i;
        }
    }
    
    minIntensRad += minRadius;
    
    pupils.push_back(Vec3f(center.x + frameX, center.y+ frameY, minIntensRad));
    
    
#if TIME_MEASURING
    double time = (getTickCount() - e1)/ getTickFrequency();
    cout << "pupilLocalisation time = " << time << endl;
#endif

}

void drawPupils(const vector<Vec3f> &pupils, Mat &frame)
{
    for( int i = 0; i < pupils.size(); i++ )
    {
        int radius = cvRound(pupils[i][2]);
        
        Point frameCenter( cvRound(pupils[i][0]), cvRound(pupils[i][1]) );
        // circle outline
        Scalar color = CV_RGB(0, 255, 0);
        
        circle( frame, frameCenter, radius, color);
    }
}