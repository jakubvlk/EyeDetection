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

void detectAndFind(Mat &frame)
{
    // detekce obliceje - vraci oblicej
    Mat face = faceDetection(frame);
    
    // detekce oci - vraci pole oci (0-2)
    
    
    

    // ...
}



