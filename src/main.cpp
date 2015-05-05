#import "constants.h"

#include "opencv2/opencv.hpp"


#include <iostream>
#include <stdio.h>

#include <fstream>

#import "findIrisCentreWithCorrectionOfEyeROI.h"
#import "eyeCentreLocalisationByMeansOfGradients.h"
#import "removeReflections.h"
#import "irisLocalisation.h"
#import "pupilLocalisation.h"
#import "eyeLidsLocalisation.h"
#import "testing.h"

#import "functions.h"


using namespace std;
using namespace cv;


// Function Headers
int processArguments( int argc, const char** argv );
void showUsage( string name );

void loadCascades();
void detectAndDisplay(void*,  Mat frame );

void refreshImage();
vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face);
Rect pickFace(vector<Rect> faces);


vector<Rect> sortEyes(const vector<Rect> &eyes, Rect face);


// TESTS
int facesDetectedCount = 0, eyesDetectedCount = 0;

// default values
#if XCODE
String faceCascadeName = "../../../res/lbpcascade_frontalface.xml";  //lbpcascade_frontalface.xml    //haarcascade_frontalface_alt.xml
String eyesCascadeName = "../../../res/haarcascade_eye_tree_eyeglasses.xml";
#else
String faceCascadeName = "../res/haarcascade_frontalface_alt.xml";
String eyesCascadeName = "../res/haarcascade_eye_tree_eyeglasses.xml";
#endif

CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;

const int imageWidth = 640;

string window_name = "Eye Detection";

string file = "../../../res/pics/BioID_0000.pgm";
//string file = "../res/pics/lena.png";
bool useVideo = false, useCamera = false, stepFrame = false, showWindow = false;
bool drawInFrame = true;
Mat frame, originalFrame;


vector<Point> eyesCentres;
vector<Vec3f> irises;
vector<Vec3f> pupils;
vector<Vec4f> eyeLids;


int main( int argc, const char** argv )
{
    
    processArguments( argc, argv);
    
    CvCapture* capture;
    
   	loadCascades();
    
#if EXPERIMENTS
    
//    testEyeCenterDetection(&detectAndDisplay, 0, frame, originalFrame, eyesCentres);
//    testIrisDetection(&detectAndDisplay, 0, frame, originalFrame, irises);
//    testLidsDetectionvoid(&detectAndDisplay, 0, frame, originalFrame, eyeLids);
    
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
        detectAndDisplay(0, frame);
        
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
                        if (frame.empty())
                            break;
                        
                        frame = resizeMat(frame, imageWidth);
                        
                        originalFrame = frame.clone();
                        
                        if( !frame.empty() )
                        {
                            detectAndDisplay(0, frame );
                        }
                        else
                        {
                            printf(" --(!) No captured frame -- Break!"); break;
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
                // normalni stav
                else
                {
                    frame = cvQueryFrame( capture );
                    if (frame.empty())
                        break;
                    
                    frame = resizeMat(frame, imageWidth);
                    
                    originalFrame = frame.clone();
                    
                    if( !frame.empty() )
                    {
                        detectAndDisplay(0,  frame );
                    }
                    else
                    {
                        printf(" --(!) No captured frame -- Break!"); break;
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

int processArguments( int argc, const char** argv )
{
    cout << argc << endl;
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if ((arg == "-h") || (arg == "--help"))
        {
            showUsage(argv[0]);
            return 0;
        }
        else if ((arg == "-f") || (arg == "--file"))
        {
            if (i + 1 < argc)
            {
                file = argv[++i];
            }
            else
            {
                cerr << "--file option requires one argument." << endl;
                return 1;
            }
        }
        else if ((arg == "-v") || (arg == "--video"))
        {
            useVideo = true;
        }
        else if ((arg == "-s") || (arg == "--step"))
        {
            stepFrame = true;
            showWindow = true;
        }
        else if ((arg == "-c") || (arg == "--camera"))
        {
            useCamera = true;
        }
    }
    
    return 0;
}

void showUsage( string name )
{
    cerr << "Usage: " << name << " <option(s)> SOURCES"
    << "Options:\n"
    << "\t-h,--help\t\tShow this help message\n"
    << endl;
}

void loadCascades()
{
    if( !faceCascade.load( faceCascadeName ) )
    {
        printf("--(!)Error loading\n");
    }
    
   	if( !eyesCascade.load( eyesCascadeName ) )
    {
        printf("--(!)Error loading\n");
    }
}

// detect
void non_member(void*, int i0, int i1) {
    std::cout << "I don't need any context! i0=" << i0 << " i1=" << i1 << "\n";
}

// the function using the function pointers: - TEST
//void somefunction(void (*fptr)(void*, int, int), void* context) {
//    fptr(context, 17, 42);
//}

//somefunction(&non_member, 0);

void detectAndDisplay(void*, Mat frame )
{
    vector<Rect> faces;
    Mat frame_gray;
    eyesCentres.clear();
    irises.clear();
    pupils.clear();
    eyeLids.clear();
    //eyes.clear();
    
    // convert from color to grayscale
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    
    // contrast adjustment using the image's histogram
    equalizeHist( frame_gray, frame_gray );
    
    //Detects objects (faces) of different sizes in the input image. The detected objects are returned as a list of rectangles.
    faceCascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(frame_gray.size().width / 4, frame_gray.size().height / 4) );
    
    if (faces.size() > 0)
    {
        Rect face = pickFace(faces);
        
        Point center( face.x + face.width*0.5, face.y + face.height*0.5 );
        rectangle( frame, Rect(face.x, face.y, face.width, face.height), Scalar( 255, 0, 255 ), 4, 8, 0 );
        facesDetectedCount++;
        
        Mat faceROI = frame_gray( face );
        vector<Rect> eyes;
        
        //-- In each face, detect eyes
        eyesCascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(23, 23) );   //20
        
        
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
            
            Mat eyeMat = newFaceROI;//faceROI(Rect(eyes[j].x, eyes[j].y, eyes[j].width, eyes[j].height));
            
            char numstr[21]; // enough to hold all numbers up to 64-bits
            sprintf(numstr, "%d", static_cast<int>(j + 1));
            string eyeName = "eye";
            
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
        detectAndDisplay(0, frame);
    }
}