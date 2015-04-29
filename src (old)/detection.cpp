//
//  detection.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 19/03/15.
//
//

#import "detection.h"
#import "faceDetection.h"
#import "eyeDetection.h"

void initDetection()
{
    loadFaceCascade();
    loadEyeCascade();
}

void detectAndFind(Mat &frame, const Mat &originalFrame, Rect &frameFace, vector<Rect> &frameEyes)
{
    // detekce obliceje - vraci oblicej
    Mat face = faceDetection(frame, frameFace);
    
    // detekce oci - vraci pole oci (0-2)    
    vector<Rect> eyes = eyeDetection(face, frameFace, originalFrame, frameEyes);
    

    
}



