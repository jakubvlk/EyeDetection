//
//  findIrisCentreWithCorrectionOfEyeROI.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 28/04/15.
//
//

#include "findIrisCentreWithCorrectionOfEyeROI.h"

#import "constants.h"
#import "functions.h"


Point findIrisCentreWithCorrectionOfEyeROI( Mat &frame, Mat &eye, string windowName, int windowX, int windowY, int frameX, int frameY, vector<Point> &eyesCentres )
{
#if TIME_MEASURING
    double time_time;
    int64 time_wholeFunc = getTickCount();
#endif
    
    bool rightEye = (frameX + eye.cols * 0.5) > (frame.cols * 0.5);
    
    Mat blurredEye;
    medianBlur(eye, blurredEye, 3);
    
    // 3.1. INTENSITY SCALING 4~8
    Mat intensiveEye(blurredEye.rows, blurredEye.cols, CV_8U);
    intenseMul(blurredEye, intensiveEye, 5);
    
    //showWindowAtPosition( windowName + " post intensity", intensiveEye, windowX, windowY + 130);
    
    // otsu binarisation
    Mat binaryMat;
    double highThreshVal = threshold( intensiveEye, binaryMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
    double lowerThreshVal = highThreshVal * 0.5;
    
    Mat eyeCanny;
    Canny(binaryMat, eyeCanny, lowerThreshVal, highThreshVal);
    
    //showWindowAtPosition( windowName + " canny", eyeCanny, windowX, windowY + 390);
    
    // find contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    findContours( eyeCanny, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );
    
    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    
    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       	boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       	minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }
    
    // 3.2. IRIS CONTOUR SELECTION
    for( int i = 0; i < boundRect.size(); )
    {
        // decline square contours
        if (boundRect[i].height >= 1.5 * boundRect[i].width)
        {
            boundRect.erase(boundRect.begin() + i);
            center.erase(center.begin() + i);
            radius.erase(radius.begin() + i);
            contours.erase(contours.begin() + i);
            contours_poly.erase(contours_poly.begin() + i);
        }
        else
        {
            i++;
        }
    }
    
    // select the 2 largest bounding boxes
    int indexCandidate1 = -1, indexCandidate2 = -1, finalCandidate = 0;
    bool empty = boundRect.size() == 0;
    
    
    int minWidth = 3;
    int minHeight = 3;
    int minAre = minWidth * minHeight;
    
    if (!empty)
    {
        if (boundRect.size() > 1)
        {
            int max = -1;
            
            for( int i = 0; i < boundRect.size(); i++)
            {
                if (boundRect[i].size().area() > max)
                {
                    indexCandidate1 = i;
                    max = boundRect[i].size().area();
                }
            }
            
            max = -1;
            for( int i = 0; i < boundRect.size(); i++)
            {
                // if it isn't the first contour and the second contour doesn't have intersect with the first contour nad it's so far the biggest contour
                if (indexCandidate1 != i && (boundRect[indexCandidate1] & boundRect[i]).area() == 0 && boundRect[i].area() > minAre && boundRect[i].size().area() > max)
                {
                    if (boundRect[i].width > minWidth && boundRect[i].height > minHeight && boundRect[i].area() > minAre)
                    {
                        indexCandidate2 = i;
                        max = boundRect[i].size().area();
                    }
                }
            }
            
            if (indexCandidate2 != -1)
            {
                // Compare y cords of the centers. If they are similar, we will select final candidate by intensity. Otherwise we will select final candidate with higher y cord.
                int yDistanceThresh = 4;
                if (abs ( center[indexCandidate1].y - center[indexCandidate2].y ) <= yDistanceThresh)
                {
                    int maxIntensity = 230;
                    float intens1 = avgIntensity(eye, boundRect[indexCandidate1].tl().x, boundRect[indexCandidate1].tl().y, boundRect[indexCandidate1].width, boundRect[indexCandidate1].height, maxIntensity);
                    float intens2 = avgIntensity(eye, boundRect[indexCandidate2].tl().x, boundRect[indexCandidate2].tl().y, boundRect[indexCandidate2].width, boundRect[indexCandidate2].height, maxIntensity);
                    
                    if (intens1 < intens2)
                    {
                        finalCandidate = indexCandidate1;
                    }
                    else
                    {
                        finalCandidate = indexCandidate2;
                    }
                }
                else
                {
                    if (center[indexCandidate1].y > center[indexCandidate2].y)
                    {
                        finalCandidate = indexCandidate1;
                    }
                    else
                    {
                        finalCandidate = indexCandidate2;
                    }
                }
            }
            else
            {
                finalCandidate = indexCandidate1;
            }
        }
        
        // 3.3. ELLIPSE FITTING
        RotatedRect myEllipse;
        if (contours[finalCandidate].size() >= 5)
            myEllipse = fitEllipse( Mat(contours[finalCandidate]));
        
        
        // 3.4. CORRECTION OF DETECTED CENTER
        int boundRectHeight = boundRect[finalCandidate].height;
        
        // upper left corner of R2
        int x1 = lround( myEllipse.center.x - 0.25 * boundRectHeight);
        int y1 = boundRect[finalCandidate].tl().y;
        Rect R2 = Rect(x1, y1, boundRectHeight, boundRectHeight);
        if (R2.x < 0)
            R2.x = 0;
        if (R2.x + R2.width >= binaryMat.cols)
            R2.x = binaryMat.cols - R2.width;
                
        // upper right corner of R1
        int x2 = myEllipse.center.x;
        int y2 = boundRect[finalCandidate].tl().y;
        Rect R1 = Rect(x2 - boundRectHeight, y2, boundRectHeight, boundRectHeight);
        if (R1.x < 0)
            R1.x = 0;
        if (R1.x  + R1.width >= binaryMat.cols)
            R1.x = binaryMat.cols - R1.width;
                
        Mat r1Mat = binaryMat(R1);
        Mat r2Mat = binaryMat(R2);
        if (rightEye)
        {
            // correction is required
            if (blackPixelsCount(r1Mat) > blackPixelsCount(r2Mat))
            {
                vector<Point> correctedContour;
                for (int i = 0; i < contours[finalCandidate].size(); i++)
                {
                    if ( ( boundRect[finalCandidate].tl().x + boundRectHeight ) >  contours[finalCandidate][i].x)
                    {
                        correctedContour.push_back(contours[finalCandidate][i]);
                    }
                }
                
                if (correctedContour.size() >= 5)
                    myEllipse = fitEllipse( Mat(correctedContour));
                
            }
        }
        else
        {
            // correction is required
            if (blackPixelsCount(r1Mat) > blackPixelsCount(r2Mat))
            {
                vector<Point> correctedContour;
                for (int i = 0; i < contours[finalCandidate].size(); i++)
                {
                    if ( ( boundRect[finalCandidate].tl().x + boundRectHeight ) >  contours[finalCandidate][i].x)
                    {
                        correctedContour.push_back(contours[finalCandidate][i]);
                    }
                }
                
                if (correctedContour.size() >= 5)
                    myEllipse = fitEllipse( Mat(correctedContour));
            }
        }
        
        
        Point frameCenter(myEllipse.center.x + frameX, myEllipse.center.y + frameY);
        eyesCentres.push_back(frameCenter);
        
#if TIME_MEASURING
        time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
        cout << "find eye cenre time = " << time_time << endl;
#endif
        
        return myEllipse.center;
    }
        
#if TIME_MEASURING
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "find eye cenre time = " << time_time << endl;
#endif
    
    return Point(eye.rows * 0.5, eye.cols * 0.5);
}

float avgIntensity(Mat mat, int x, int y, int width, int height, int maxIntensity)
{
    int totalIntensity = 0;
    int pixelCount = 0;
    
    for (int i = y; i < y + height; i++)
    {
        for (int j = x; j < x + width; j++)
        {
            int intensity = mat.at<uchar>(i, j);
            
            if (intensity < maxIntensity)
            {
                totalIntensity += intensity;
                pixelCount++;
            }
        }
    }
    
    return totalIntensity / (float)pixelCount;
}

int blackPixelsCount(Mat mat)
{
    int blackPixelCount = 0;
    
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            if(mat.at<uchar>(i, j) == 0)
            {
                blackPixelCount++;
            }
        }
    }
    
    return blackPixelCount;
}