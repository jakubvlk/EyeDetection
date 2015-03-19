//
//  captureImage.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 19/03/15.
//
//

#import "captureImage.h"
#import "detection.h"


//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/photo/photo.hpp"

string windowName = "Eye Detection";
Mat frame, originalFrame;

void startCapture(string file, bool useVideo, bool useCamera)
{
    CvCapture* capture;
    
    if (useVideo)
    {
        capture = cvCaptureFromFile(file.c_str());
    }
    else if (useCamera)
    {
        capture =  cvCaptureFromCAM( -1 );
    }
    else
    {
        frame = imread(file);
        
        if (frame.size().area() > 0)
        {
            originalFrame = frame.clone();
            detectAndFind(frame);
            
            imshow( windowName, frame );
            
            waitKey(0);
        }
        else
        {
            cerr << "Frame is empty" << endl;
        }
    }
}