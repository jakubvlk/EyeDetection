//
//  functions.cpp
//  EyeDetection
//
//  Created by Jakub Vlk
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

// Sort from left to right
vector<Rect> sortEyes(const vector<Rect> &eyes, Rect face)
{
    vector<Rect> newEyes;
    
    // left eye
    if (eyes[0].x > (face.width * 0.5))
    {
        newEyes.push_back(eyes[0]);
        newEyes.push_back(eyes[1]);
    }
    // right eye
    else
    {
        newEyes.push_back(eyes[1]);
        newEyes.push_back(eyes[0]);
    }
    
    return newEyes;
}

// return the biggest face
Rect pickFace(vector<Rect> faces)
{
    double maxArea = 0;
    int maxIndex = -1;
    
    for (int i = 0; i < faces.size(); ++i)
    {
        int area = faces[i].size().width * faces[i].size().height;
        if (area > maxArea)
        {
            maxArea = area;
            maxIndex = i;
        }
    }
    
    if (maxIndex >= 0)
    {
        return faces[maxIndex];
    }
    else
    {
        return faces[0];
    }
}

vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face)
{
    vector<Rect> correctEyes = eyes;
    
    // eye is for sure in upper half of image
    for (int i = 0; i < correctEyes.size();)
    {
        if (correctEyes[i].y > (face.size().height * 0.5 ))
        {
            cout << "Delete the eye out of the upper half of the face" << endl;
            correctEyes.erase(correctEyes.begin() + i);
        }
        else
        {
            i++;
        }
    }
    
    // it the eye region is out of the face then delete
    for (int i = 0; i < correctEyes.size();)
    {
        // Right eye
        if ( eyes[i].x > (face.size().width * 0.5) )
        {
            if ( (eyes[i].x + eyes[i].width)  > face.size().width )
            {
                cout << "Delete region of the right eye" << endl;
                correctEyes.erase(correctEyes.begin() + i);
            }
            else
            {
                i++;
            }
        }
        // Left eye
        else
        {
            if ( eyes[i].x < 0 || (eyes[i].x + eyes[i].width)  > (face.size().width * 0.5 ) )
            {
                cout << "Delete region of the left eye" << endl;
                correctEyes.erase(correctEyes.begin() + i);
            }
            else
            {
                i++;
            }
        }
    }
    
    // delete eyes with similar centre
    for (int i = 0; i < correctEyes.size(); )
    {
        bool incressI = true;
        double distancesThresh = 0.1;
        
        for (int j = 0; j < correctEyes.size(); )
        {
            bool incressJ = true;
            
            if (i != j)
            {
                double distance = /*sqrt*/( pow(correctEyes[i].x - correctEyes[j].x, 2.) + pow(correctEyes[i].y - correctEyes[j].y, 2.) );
                
                if (face.size().width != 0)
                {
                    //cout << "distance = " << distance / face.size().width << endl;
                    if (distance / pow( face.size().width, 2.) < distancesThresh)
                    {
                        if (correctEyes[i].width > correctEyes[j].width)
                        {
                            correctEyes.erase(correctEyes.begin() + j);
                            incressJ = false;
                        }
                        else
                        {
                            correctEyes.erase(correctEyes.begin() + i);
                            incressI = false;
                        }
                    }
                }
            }
            
            if (incressJ)
                j++;
        }
        
        if (incressI)
            i++;
    }
    
    if (correctEyes.size() > 2)
        correctEyes.erase(correctEyes.begin() + 2, correctEyes.begin() + correctEyes.size());
    
    return correctEyes;
}

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