//
//  irisLocalisation.cpp
//  EyeDetection
//
//  Created by Jakub Vlk
//
//

#import "constants.h"

#include "opencv2/opencv.hpp"


#include <iostream>
#include <stdio.h>

#include <fstream>

#import "findIrisCentreWithCorrectionOfEyeROI.h"
#import "eyeCentreLocalisationByMeansOfGradients.h"
#import "irisLocalisation.h"
#import "pupilLocalisation.h"
#import "eyeLidsLocalisation.h"
#import "testing.h"
#import "processArguments.h"

#import "functions.h"


using namespace std;
using namespace cv;


int loadCascades();
void detectAndShow(void*,  Mat frame );

void refreshImage();



// tests
int facesDetectedCount = 0, eyesDetectedCount = 0;

// default values
#if XCODE
String faceCascadeName = "../../../res/haarcascade_frontalface_alt.xml";  //lbpcascade_frontalface.xml    //haarcascade_frontalface_alt.xml
String eyesCascadeName = "../../../res/haarcascade_eye_tree_eyeglasses.xml";
#else
String faceCascadeName = "../res/haarcascade_frontalface_alt.xml";
String eyesCascadeName = "../res/haarcascade_eye_tree_eyeglasses.xml";
#endif

CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;

const int imageWidth = 640;

string window_name = "Eye Detection";

#if XCODE
string file = "../../../res/videos/geordi01_visible_ir_20140502.avi";
#else
string file = "../res/videos/geordi01_visible_ir_20140502.avi";
#endif

bool useVideo = true, useCamera = false, stepFrame = false, showWindow = false;
bool drawInFrame = true;
Mat frame, originalFrame;


vector<Point> eyesCentres;
vector<Vec3f> irises;
vector<Vec3f> pupils;
vector<Vec4f> eyeLids;


int main( int argc, const char** argv )
{
    
    int res = processArguments(argc, argv, file, useVideo, stepFrame, showWindow, useCamera);
    if (res == 2)
        return 0;
    
    
    CvCapture* capture;
    
   	if (loadCascades() == -1)
        return 1;
    
#if EXPERIMENTS
    
//    testEyeCenterDetection(&detectAndShow, 0, frame, originalFrame, eyesCentres);
//    testIrisDetection(&detectAndShow, 0, frame, originalFrame, irises);
    testLidsDetectionvoid(&detectAndShow, 0, frame, originalFrame, eyeLids);
    
#else

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
        
        originalFrame = frame.clone();
        detectAndShow(0, frame);
        
        waitKey(0);
    }
    
    if (useVideo || useCamera)
    {
        if( capture )
        {
            while( true )
            {
                if (stepFrame)
                {                    
                    
                    int c = waitKey(10);
                    
                    if( (char)c == 'n' || (char)c == 'N' || showWindow)
                    {
                        frame = cvQueryFrame( capture );
                        
                        if( !frame.empty() )
                        {
                            frame = resizeMat(frame, imageWidth);
                            
                            originalFrame = frame.clone();
                            
                            detectAndShow(0, frame );
                        }
                        else
                        {
                            cout << "No captured frame -- Break!" << endl;
                            break;
                        }
                        
                        c = -1;
                        showWindow = false;
                    }
                    else if( (char)c == 'f' || (char)c == 'F' || showWindow)
                    {
                        drawInFrame = !drawInFrame;
                        
                        refreshImage();
                    }
                }
                // normal state
                else
                {
                    frame = cvQueryFrame( capture );
                    
                    if( !frame.empty() )
                    {
                        frame = resizeMat(frame, imageWidth);
                        
                        originalFrame = frame.clone();
                        
                        detectAndShow(0,  frame );
                    }
                    else
                    {
                        cout << "No captured frame -- Break!" << endl;
                        break;
                    }
                }
                
                int c = waitKey(10);
                if( (char)c == 'c' || (char)c == 'C' )
                {
                    break;
                }
                else if( (char)c == 'p' || (char)c == 'P' || showWindow)
                {
                    stepFrame = !stepFrame;
                }
            }
        }
    }
    
#endif
    
    return 0;
}

int loadCascades()
{
    if( !faceCascade.load( faceCascadeName ) )
    {
        cout << "Error loading cascades" << endl;
        return -1;
    }
    
   	if( !eyesCascade.load( eyesCascadeName ) )
    {
        cout << "Error loading cascades" << endl;
        return -1;
    }
    
    return  0;
}

void detectAndShow(void*, Mat frame )
{
    vector<Rect> faces;
    Mat frame_gray;
    eyesCentres.clear();
    irises.clear();
    pupils.clear();
    eyeLids.clear();
    
    string eyeName = "eye";
    
    // convert from color to grayscale
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    
    // contrast adjustment using the image's histogram
    equalizeHist( frame_gray, frame_gray );
    
    // Detects objects (faces) of different sizes in the input image. The detected objects are returned as a list of rectangles.
    faceCascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(frame_gray.size().width / 4, frame_gray.size().height / 4) );
    
    if (faces.size() > 0)
    {
        Rect face = pickFace(faces);
        
        Point center( face.x + face.width*0.5, face.y + face.height*0.5 );
        rectangle( frame, Rect(face.x, face.y, face.width, face.height), Scalar( 255, 0, 255 ), 4, 8, 0 );
        facesDetectedCount++;
        
        Mat faceROI = frame_gray( face );
        vector<Rect> eyes;
        
        // In each face, detect eyes
        eyesCascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20) );   //20
        
        
        eyes = pickEyeRegions(eyes, faceROI);
        
        if (eyes.size() > 1)
            eyes = sortEyes(eyes, face);
        
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            eyesDetectedCount++;
            
            rectangle( frame, Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar( 0, 0, 255 ), 2);
            
            Mat newFaceROI = originalFrame(Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height));
            
            // convert from color to grayscale
            cvtColor( newFaceROI, newFaceROI, CV_BGR2GRAY );
            // contrast adjustment using the image's histogram
            equalizeHist( newFaceROI, newFaceROI );
            
            Mat eyeMat = newFaceROI;
            
            char numstr[21]; // enough to hold all numbers up to 64-bits
            sprintf(numstr, "%d", static_cast<int>(j + 1));
            
            Mat eyeWithoutReflection = removeReflections(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
            
            
            //Point eyeCenter = findIrisCentreWithCorrectionOfEyeROI(frame, eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyesCentres);
            Point eyeCenter = eyeCentreLocalisationByMeansOfGradients( eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyesCentres);
            
            irisLocalisation( eyeWithoutReflection, 3, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeCenter, irises);
            pupilLocalisation(eyeWithoutReflection, 3, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeCenter, pupils);
            
            eyeLidsLocalisation(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeLids);
        }
        
        if (drawInFrame)
        {
            drawEyesCentres(eyesCentres, frame);
            drawPupils(pupils, frame);
            drawIrises(irises, frame);
            drawEyeLids(eyeLids, frame);
        }
    }
    
    imshow( window_name, frame );
}

void refreshImage()
{
    //if (stepFrame)
    {
        frame = originalFrame.clone();
        detectAndShow(0, frame);
    }
}