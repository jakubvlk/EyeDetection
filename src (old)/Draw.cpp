//
//  Draw.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 20/03/15.
//
//

#include "Draw.h"


Scalar faceBoxColor = CV_RGB(255, 0, 255);
Scalar eyeBoxColor = CV_RGB(255, 0, 0);

// private functions
void drawFace(Mat &frame, const Rect &frameFace);
void drawEyes(Mat &frame, const Rect &frameFace, const vector<Rect> &frameEyes);


void draw(Mat &frame, const Rect &frameFace, const vector<Rect> &frameEyes)
{
    drawFace(frame, frameFace);
    drawEyes(frame, frameFace, frameEyes);
}

void drawFace(Mat &frame, const Rect &frameFace)
{
    rectangle( frame, Rect(frameFace.x, frameFace.y, frameFace.width, frameFace.height), faceBoxColor, 4, 8, 0 );
}

// private functions
void drawEyes(Mat &frame, const Rect &frameFace, const vector<Rect> &frameEyes)
{
    for (int j = 0; j < frameEyes.size(); j++)
    {
        cout << frameEyes[j] << endl;
        
        rectangle( frame, Rect(frameFace.x + frameEyes[j].x, frameFace.y + frameEyes[j].y, frameEyes[j].width, frameEyes[j].height), eyeBoxColor, 2);
    }
}