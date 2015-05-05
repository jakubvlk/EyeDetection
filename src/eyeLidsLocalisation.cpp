//
//  eyeLidsLocalisation.cpp
//  EyeDetection
//
//  Created by Jakub Vlk
//
//

#include "eyeLidsLocalisation.h"

#import "functions.h"
#import "constants.h"


void eyeLidsLocalisation(Mat &eye, string windowName, int windowX, int windowY, int frameX, int frameY, vector<Vec4f> &eyeLids)
{
    
#if TIME_MEASURING
    double time_time;
    int64 time_wholeFunc = getTickCount();
#endif
    
    Mat blurredEye;
    
    // blur
    medianBlur(eye, blurredEye, 7);
    
    Mat intensiveEye(blurredEye.rows, blurredEye.cols, CV_8U);
    intenseMul(blurredEye, intensiveEye, 7);
    // showWindowAtPosition( windowName + " post intensity", intensiveEye, windowX, windowY + 130);
    
    Mat thresholdOutput;
    threshold( intensiveEye, thresholdOutput, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
    // showWindowAtPosition( windowName + " otsu ", thresholdOutput, windowX, windowY + 260);
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    // Find contours
    findContours( thresholdOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );
    
    // Find rectangles and ellipses for each contour
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );
    
    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( Mat(contours[i]) );
        if( contours[i].size() > 5 )
        {
            
            minEllipse[i] = fitEllipse( Mat(contours[i]) );
        }
    }
    
    for( size_t i = 0; i< contours.size(); i++ )
    {
        if (minRect[i].size.width > eye.size().width * 0.15 && minRect[i].size.height > eye.size().height * 0.15 && minRect[i].center.y > eye.size().height * 0.35 && minRect[i].center.y < eye.size().height * 0.65 )
        {
            Point framePoint = Point(frameX, frameY);
            Point leftTopPoint = Point(minRect[i].center.x - 0.35f * eye.size().width,  minEllipse[i].boundingRect().tl().y);
            Point rightTopPoint = Point(minRect[i].center.x + 0.35f * eye.size().width, minEllipse[i].boundingRect().tl().y);
            
            Point leftBottomPoint = Point(minRect[i].center.x - 0.35f * eye.size().width, minEllipse[i].boundingRect().br().y);
            Point rightBottomPoint = Point(minRect[i].center.x + 0.35f * eye.size().width, minEllipse[i].boundingRect().br().y);
            
            eyeLids.push_back(Vec4f(leftTopPoint.x + framePoint.x, leftTopPoint.y + framePoint.y, rightTopPoint.x + framePoint.x, rightTopPoint.y + framePoint.y));
            eyeLids.push_back(Vec4f(leftBottomPoint.x + framePoint.x, leftBottomPoint.y + framePoint.y, rightBottomPoint.x + framePoint.x, rightBottomPoint.y + framePoint.y));
            
            break;
        }
    }
    
    
#if TIME_MEASURING
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "eyeLidsLocalisation time = " << time_time << endl;
#endif

}

void drawEyeLids(const vector<Vec4f> &eyeLids, Mat &frame)
{
    Scalar color = Scalar(255, 255, 255);
    
    for( size_t i = 0; i < eyeLids.size(); i++ )
    {
        line(frame, Point(eyeLids[i][0], eyeLids[i][1]), Point(eyeLids[i][2], eyeLids[i][3]), color);
    }
}