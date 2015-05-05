//
//  eyeCentreLocalisationByMeansOfGradients.cpp
//  EyeDetection
//
//  Created by Jakub Vlk
//
//

#include "eyeCentreLocalisationByMeansOfGradients.h"

#import "functions.h"
#import "constants.h"


Point eyeCentreLocalisationByMeansOfGradients( Mat &eye, string windowName, int windowX, int windowY, int frameX, int frameY, vector<Point> &eyesCentres )
{
    
#if TIME_MEASURING
    double time_time;
    int64 time_wholeFunc = getTickCount();
#endif
    
    Mat newEye = eye.clone();
    
    int fastWidth = 50;
    bool scaleMat = false;
    if (scaleMat)
    {
        newEye = resizeMat(eye, fastWidth);
    }
    else
    {
        fastWidth = 20;
        newEye = eye( Rect(fastWidth * 0.25, fastWidth * 0.6, eye.size().width - fastWidth * 0.25, eye.size().height - fastWidth * 0.6) );
    }
    
    //showWindowAtPosition( windowName + "eye cutted", eye, windowX, windowY);
    
    
    Mat intensiveEye(newEye.rows, newEye.cols, CV_8U);
    intenseMul(newEye, intensiveEye, 5);
    
    
    Mat gradX, gradY;
    
    double scale = 1;
    double delta = 0;
    int kernelSize = 3;
    int ddepth = CV_32F;
    
    /// Gradient X
    Sobel( intensiveEye, gradX, ddepth, 1, 0, kernelSize, scale, delta, BORDER_DEFAULT );
    /// Gradient Y
    Sobel( intensiveEye, gradY, ddepth, 0, 1, kernelSize, scale, delta, BORDER_DEFAULT );
    
    
    float dotProducts[eye.rows][newEye.cols];
    for (int y = 0; y < newEye.rows; y++)
    {
        for (int x = 0; x < newEye.cols; x++)
        {
            dotProducts[y][x] = 0;
        }
    }
    
    bool useWeight = true;
    Mat weight;
    GaussianBlur( intensiveEye, weight, Size( 5, 5 ), 0, 0 );
    
    // inverse color
    for (int y = 0; y < weight.rows; ++y)
    {
        unsigned char *row = weight.ptr<unsigned char>(y);
        for (int x = 0; x < weight.cols; ++x)
        {
            row[x] = (255 - row[x]);
        }
    }
    
    
    int gradientsCount = 0;
    Mat sum = Mat::zeros(newEye.rows, newEye.cols, CV_32F);
    for (int y = 0; y < newEye.rows; ++y)
    {
        const float *gradXRows = gradX.ptr<float>(y);
        const float *gradYRows = gradY.ptr<float>(y);
        
        const unsigned char *ws = weight.ptr<unsigned char>(y);
        
        for (int x = 0; x < newEye.cols; ++x)
        {
            float gX = gradXRows[x];
            float gY = gradYRows[x];
            if (gX == 0.f && gY == 0.f)
            {
                continue;
            }
            
            float gMag = sqrt( gX * gX + gY*gY );
            
            if (gMag < 310)
                continue;
            
            float gx = gX / gMag, gy = gY / gMag;
            
            for (int cy = 0; cy < newEye.rows; cy++)
            {
                float *sumRows = sum.ptr<float>(cy);
                for (int cx = 0; cx < newEye.cols; cx++)
                {
                    if (x == cx && y == cy)
                    {
                        continue;
                    }
                    
                    float dx = x - cx, dy = y - cy;
                    
                    
                    //normalize d
                    float dMagnitude = sqrt( dx * dx + dy * dy );
                    dx = dx / dMagnitude;
                    dy = dy / dMagnitude;
                    
                    float dotProduct;
                    if (useWeight)
                    {
                        dotProduct = dx*gx + dy*gy * (ws[x] / 150.f);
                    }
                    else
                    {
                        dotProduct = dx*gx + dy*gy;
                    }
                    
                    if (dotProduct < 0.f)
                        dotProduct = 0.f;
                    
                    sumRows[cx] += dotProduct * dotProduct;
                    gradientsCount++;
                }
            }
            
            
        }
    }
    
    Point centre;
    double max;
    minMaxLoc(sum, NULL, &max, NULL, &centre);
    
    
    if (scaleMat)
    {
        centre = unscalePoint(centre, eye.cols, newEye.cols);
    }
    else
    {
        centre = uncut(centre, fastWidth);
    }
    
    Point frameCenter(centre.x + frameX, centre.y + frameY);
    eyesCentres.push_back(frameCenter);
    
    
#if TIME_MEASURING
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "eyeCentreLocalisationByMeansOfGradients time = " << time_time << endl;
#endif
    
    return centre;
}

Mat resizeMat(Mat src, int width)
{
    Mat dst;
    
    double widthRatio = width / (double)src.size().width;
    
    resize(src, dst, Size(width, lround(widthRatio * src.size().height)));
    
    return dst;
}

Point unscalePoint(Point p, int origWidth, int width)
{
    float ratio = ((float)width / origWidth);
    int x = roundl(p.x / ratio);
    int y = roundl(p.y / ratio);
    
    return Point(x, y);
}

Point uncut(Point p, int cut)
{
    return Point(p.x + cut * 0.25f, p.y + cut * 0.6f);
}


void drawEyesCentres(const vector<Point> &eyesCentres, Mat &frame)
{
    Scalar color = Scalar(0, 0, 255);
    int lineLength = 6;
    
    for( size_t i = 0; i < eyesCentres.size(); i++ )
    {
        line(frame, Point(eyesCentres[i].x - lineLength*0.5, eyesCentres[i].y), Point(eyesCentres[i].x + lineLength*0.5, eyesCentres[i].y), color);
        line(frame, Point(eyesCentres[i].x, eyesCentres[i].y - lineLength*0.5), Point(eyesCentres[i].x, eyesCentres[i].y + lineLength*0.5), color);
        
        //        circle(frame, eyesCentres[i], 1, color);
    }
}