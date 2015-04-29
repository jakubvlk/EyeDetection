#import "constants.h"


#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/opencv.hpp"


#include <iostream>
#include <stdio.h>

#include <fstream>

#import "findIrisCentreWithCorrectionOfEyeROI.h"
#import "eyeCentreLocalisationByMeansOfGradients.h"
#import "removeReflections.h"


using namespace std;
using namespace cv;


// Function Headers
int processArguments( int argc, const char** argv );
void showUsage( string name );

void loadCascades();
void detectAndDisplay( Mat frame );

void showWindowAtPosition( string imageName, Mat mat, int x, int y );
void refreshImage();
vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face);
Rect pickFace(vector<Rect> faces);


void drawEyesCentres();
void drawPupils();
void drawIrises();
void drawEyeLids();


Mat resizeMat(Mat src, int width);
Point myHoughCircle(Mat eye, int kernel, string windowName, int x, int y, int frameX, int frameY, Point center);
void findEyeCorners ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
void VPF_eyelids ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);

void FCD(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
void findPupil(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY, Point center);
void findEyeLidsOTSU(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
void blob(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);

vector<Rect> sortEyes(const vector<Rect> &eyes, Rect face);

//trackbars
void onHCParam1Trackbar(int pos, void *);
void onHCParam2Trackbar(int pos, void *);
void onHCDpTrackbar(int pos, void *);
void onHCMinDistanceTrackbar(int pos, void *);


// TESTS
void testDetection();
void testFaceDetection();
void testEyeDetection();
void testIrisDetection();
void testLidsDetection();
void testPupilsDetection();
int facesDetectedCount = 0, eyesDetectedCount = 0;
double methodTime = 0.;

// default values
#if XCODE
String face_cascade_name = "../../../res/lbpcascade_frontalface.xml";  //lbpcascade_frontalface.xml    //haarcascade_frontalface_alt.xml
String eyes_cascade_name = "../../../res/haarcascade_eye_tree_eyeglasses.xml";
#else
String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "../res/haarcascade_eye_tree_eyeglasses.xml";
#endif

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

string file = "../../../res/pics/BioID_0000.pgm";
//string file = "../res/pics/lena.png";
bool useVideo = false, useCamera = false, stepFrame = false, showWindow = false;
bool drawInFrame = true;
Mat frame, originalFrame;

// sliders
const int sliderHCParam1max = 300;
int sliderHCParam1, HCParam1;

const int sliderHCParam2max = 300;
int sliderHCParam2, HCParam2;

const int sliderHCDpMax = 200;	// deli se 10
int sliderHCDp;
double HCDp;

const int sliderHCMinDistanceMax = 200;
int sliderHCMinDistance, HCMinDistance;

//vector<Rect> eyes;
vector<Point> eyesCentres;
vector<Vec3f> irises;
vector<Vec3f> pupils;
vector<Vec4f> eyeLids;


int main( int argc, const char** argv )
{
    
    processArguments( argc, argv);
    
    CvCapture* capture;
    // Mat frame;
    
    sliderHCParam1 = HCParam1 = 1;	//26	//35
    sliderHCParam2 = HCParam2 = 16;		//21	//30
    
    sliderHCDp = 17;	// deli se to 10...	// 30
    HCDp = 1;	// 3
    
    sliderHCMinDistance = HCMinDistance = 1;	// 170	//57
    
   	loadCascades();
    
#if EXPERIMENTS
    
    //testDetection();
    //testFaceDetection();
    //testEyeDetection();
    testIrisDetection();
    //testPupilsDetection();
    //testLidsDetection();
    
#else
    
    if (useVideo)
    {
        capture = cvCaptureFromFile(file.c_str());
        
        //cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, 60);	// posun na 60 frame
    }
    else if (useCamera)
    {
        capture =  cvCaptureFromCAM( -1 );
    }
    else
    {
        frame = imread(file);
        //frame = resizeMat(frame, 640);
        
        originalFrame = frame.clone();
        detectAndDisplay(frame);
        
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
                        frame = resizeMat(frame, 640);
                        
                        originalFrame = frame.clone();
                        
                        //-- 3. Apply the classifier to the frame
                        if( !frame.empty() )
                        {
                            detectAndDisplay( frame );
                        }
                        else
                        {
                            printf(" --(!) No captured frame -- Break!"); break;
                        }
                        
                        c = -1;
                        showWindow = false;
                    }
                    else if( (char)c == 'i' || (char)c == 'I' || showWindow)
                    {
                        cout << "HC param 1 = " << HCParam1 <<  "HC param 2 = " << HCParam2 << ", HC dp = " << HCDp << ", HC min distance = " << HCMinDistance << endl;
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
                    frame = resizeMat(frame, 640);
                    
                    originalFrame = frame.clone();
                    
                    //-- 3. Apply the classifier to the frame
                    if( !frame.empty() )
                    {
                        detectAndDisplay( frame );
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
            // Make sure we aren't at the end of argv!
            if (i + 1 < argc)
            {
                file = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
            }
            // Uh-oh, there was no argument to the destination option.
            else
            {
                std::cerr << "--file option requires one argument." << std::endl;
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
    << "\t-d,--destination DESTINATION\tSpecify the destination path"
    << std::endl;
}

void loadCascades()
{
    //-- 1. Load the cascades
   	if( !face_cascade.load( face_cascade_name ) )
    {
        printf("--(!)Error loading\n");
        //return -1;
    }
    
   	if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        printf("--(!)Error loading\n");
        //return -1;
    }
}


/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
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
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(frame_gray.size().width / 4, frame_gray.size().height / 4) );
    
    if (faces.size() > 0)
    {
        Rect face = pickFace(faces);
        
        Point center( face.x + face.width*0.5, face.y + face.height*0.5 );
        //rectangle( frame, Rect(face.x, face.y, face.width, face.height), Scalar( 255, 0, 255 ), 4, 8, 0 );
        facesDetectedCount++;
        
        Mat faceROI = frame_gray( face );
        vector<Rect> eyes;
        
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(23, 23) );   //20
        
        
        eyes = pickEyeRegions(eyes, faceROI);
        
        if (eyes.size() > 1)
            eyes = sortEyes(eyes, face);
        
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            eyesDetectedCount++;
            
            //            if (j == 0)
            //rectangle( frame, Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar( 0, 0, 255 ), 2);
            //            else
            //                rectangle( frame, Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar( 0, 255, 255 ), 2);
            
            
            //eyes.push_back(Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height));
            
            // Pokus - Zkouska, jestli equalize na ocni oblast, zlepsi kvalitu rozpoznani
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
            Point eyeCenter = eyeCentreLocalisationByMeansOfGradients(frame, eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyesCentres);
            
            myHoughCircle(eyeWithoutReflection, 3, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeCenter);
            //findPupil(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeCenter);
            
            //findEyeLidsOTSU(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
        }
        
        if (drawInFrame)
        {
            drawEyesCentres();
            //drawPupils();
            //drawIrises();
            //drawEyeLids();
            
        }
    }
    
    //FCD(frame_gray, "", 820 + 220 , 0, 0, 0);
    //findPupil(frame_gray, "", 820 + 220 , 0, 0, 0);
    //accurateEyeCentreLocalisationByMeansOfGradients(frame_gray, "", 820 + 220 , 0, 0, 0);
    
    
    imshow( window_name, frame );
    
    createTrackbar("HC param1", window_name, &sliderHCParam1, sliderHCParam1max, onHCParam1Trackbar);
    createTrackbar("HC param2", window_name, &sliderHCParam2, sliderHCParam2max, onHCParam2Trackbar);
    createTrackbar("HC dp", window_name, &sliderHCDp, sliderHCDpMax, onHCDpTrackbar);
    createTrackbar("HC min distance", window_name, &sliderHCMinDistance, sliderHCMinDistanceMax, onHCMinDistanceTrackbar);
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

Mat mat2gray(const cv::Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
    
    return dst;
}


Rect pickFace(vector<Rect> faces)
{
    
    // vrati nejvetsi oblicej
    double max = 0;
    int maxIndex = -1;
    
    for (int i = 0; i < faces.size(); ++i)
    {
        //int faceSize =
        int volume = faces[i].size().width * faces[i].size().height;
        if (volume > max)
        {
            max = volume;
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
    // moznost: vertikalne rozdelit tvar do 4 casti. Oci by mely byt v 2. casti
    
    vector<Rect> correctEyes = eyes;
    
    // prostor pro oci je urcite ve vrchni polovine obliceje...  !!! toto by se dalo udelat i zmensenim oblasti obliceje o 1/2 -> lepsi vykon!!!
    for (int i = 0; i < correctEyes.size();)
    {
        if (correctEyes[i].y > (face.size().height * 0.5 ))
        {
            cout << "Mazu! Oblast oka mimo vrchni polovinu obliceje. x,y = " << correctEyes[i].x << ", " << correctEyes[i].y << ". Polovina obliceje ma delku " << face.size().height * 0.5 << endl;
            correctEyes.erase(correctEyes.begin() + i);
        }
        else
        {
            i++;
        }
    }
    
    // odebere ocni oblasti, ktere zasahuji mimo oblicej
    for (int i = 0; i < correctEyes.size();)
    {
        // Prave oko
        if ( eyes[i].x > (face.size().width * 0.5) )
        {
            if ( (eyes[i].x + eyes[i].width)  > face.size().width )
            {
                cout << "Mazu! Oblast praveho oka je mimo oblicej. x,y = " << eyes[i].x << ", " << eyes[i].y << endl;
                correctEyes.erase(correctEyes.begin() + i);
            }
            else
            {
                i++;
            }
        }
        // Leve oko
        else
        {
            if ( eyes[i].x < 0 || (eyes[i].x + eyes[i].width)  > (face.size().width * 0.5 ) )
            {
                cout << "Mazu! Oblast leveho oka je mimo oblicej. x,y = " << eyes[i].x << ", " << eyes[i].y << endl;
                correctEyes.erase(correctEyes.begin() + i);
            }
            else
            {
                i++;
            }
        }
    }
    
    // odstrani oci s podobnym stredem
    for (int i = 0; i < correctEyes.size(); )
    {
        bool incressI = true;
        // jak jsou vzdalene stredy 2. ocnich oblasti. Pokud je to min nez treshold (relativne), tak mensi ocni oblast odstranime
        double distancesTresh = 0.1;
        
        for (int j = 0; j < correctEyes.size(); )
        {
            bool incressJ = true;
            
            if (i != j)
            {
                double distance = /*sqrt*/( pow(correctEyes[i].x - correctEyes[j].x, 2.) + pow(correctEyes[i].y - correctEyes[j].y, 2.) );
                
                if (face.size().width != 0)
                {
                    //cout << "distance = " << distance / face.size().width << endl;
                    if (distance / pow( face.size().width, 2.) < distancesTresh)
                    {
                        // smaze mensi ze 2 ocnich oblasti
                        if (correctEyes[i].width > correctEyes[j].width)
                        {
                            correctEyes.erase(correctEyes.begin() + j);
                            //cout << "mazu j " << j << endl;
                            
                            /*if ( i >= correctEyes.size())
                             i = correctEyes.size() - 1;
                             if ( j >= correctEyes.size())
                             j = correctEyes.size() - 1;*/
                            incressJ = false;
                        }
                        else
                        {
                            correctEyes.erase(correctEyes.begin() + i);
                            //cout << "mazu i " << i << endl;
                            
                            /*if ( j >= correctEyes.size())
                             j = correctEyes.size() - 1;
                             if ( i >= correctEyes.size())
                             i = correctEyes.size() - 1;
                             */
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
    
    // TMP - tvrde smazeni na 2 ocni oblasti
    if (correctEyes.size() > 2)
        correctEyes.erase(correctEyes.begin() + 2, correctEyes.begin() + correctEyes.size());
    
    // BLBOST - vrati to treba ocni oblat dole!!!
    /*if (correctEyes.size() > 0)
     {
     return correctEyes;
     }
     else
     {
     return eyes;
     }*/
    
    return correctEyes;
}

void refreshImage()
{
    //if (stepFrame)
    {
        frame = originalFrame.clone();
        detectAndDisplay(frame);
    }
}

void onHCParam1Trackbar(int pos, void *)
{
    HCParam1 = pos;
    
    cout << "HC param1 = " << HCParam1 << endl;
    
    if (HCParam1 < 1)
        HCParam1 = 1;
    
    refreshImage();
}

void onHCParam2Trackbar(int pos, void *)
{
    HCParam2 = pos;
    
    cout << "HC param2 = " << HCParam2 << endl;
    
    if (HCParam2 < 1)
        HCParam2 = 1;
    
    refreshImage();
}

void onHCDpTrackbar(int pos, void *)
{
    HCDp = pos / 10.;
    
    if (HCDp < 1)
        HCDp = 1;
    
    cout << "HC dp = " << HCDp << endl;
    
    refreshImage();
}

void onHCMinDistanceTrackbar(int pos, void *)
{
    HCMinDistance = pos;
    
    if (HCMinDistance < 1)
        HCMinDistance = 1;
    
    cout << "HC Min Distance = " << HCMinDistance << endl;
    
    refreshImage();
}

void blob(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    showWindowAtPosition( windowName + " eye centres", eye, windowX, windowY + 130);
    
    // Read image
    Mat im = eye;//imread( "/Users/jakubvlk/MyShit/BlinkRate/res/pics/blob.jpg", CV_LOAD_IMAGE_GRAYSCALE );
    
    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;
    
    // Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = HCParam1;
    params.thresholdStep = 2;
    
    //params.minDistBetweenBlobs = HCParam1;
    //cout << params.minDistBetweenBlobs << endl;
    
    params.filterByColor = true;
    params.blobColor = 0;
    
    // Filter by Area.
    //    params.filterByArea = true;
    //    params.minArea = HCParam1;
    //
    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.5;
    
    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.87;
    
    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
    
    // Set up detector with params
    SimpleBlobDetector detector(params);
    
    // Storage for blobs
    std::vector<KeyPoint> keypoints;
    
    // Detect blobs
    detector.detect( im, keypoints);
    
    // Draw detected blobs as red circles.
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
    // the size of the circle corresponds to the size of blob
    
    Mat im_with_keypoints;
    drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    
    //    showWindowAtPosition( windowName + " tresh", im_with_keypoints, windowX, windowY + 130);
    
    for (int i = 0; i < keypoints.size(); i++)
    {
        Point frameCenter(keypoints[i].pt.x + frameX, keypoints[i].pt.y + frameY);
        Scalar color = Scalar(0, 255, 0);
        
        //circle( frame, eyesCentres[i], radius, color, 3);
        int lineLength = 10;
        line(frame, Point(frameCenter.x - lineLength*0.5, frameCenter.y), Point(frameCenter.x + lineLength*0.5, frameCenter.y), color);
        line(frame, Point(frameCenter.x, frameCenter.y - lineLength*0.5), Point(frameCenter.x, frameCenter.y + lineLength*0.5), color);
        
        showWindowAtPosition( windowName + " tresh", im_with_keypoints, windowX, windowY + 260);
    }
    
}

void VPF_eyelids ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    float hy = 0.f;
    float vpf_h[eye.rows];
    for (int i = 0; i < eye.rows; i++)
    {
        vpf_h[eye.rows] = 0.f;
    }
    
    for (int y = 0; y < eye.rows; y++)
    {
        for (int x = 0; x < eye.cols; x++)
        {
            hy += eye.at<uchar>(y, x);
        }
        hy /= eye.rows;
        
        
        for (int x = 0; x < eye.cols; x++)
        {
            vpf_h[y] += powf(eye.at<uchar>(y, x) - hy, 2.f);
        }
        
        vpf_h[y] /= eye.cols;
    }
    
    float max = 0;
    for (int i = 0; i < eye.rows; i++)
    {
        if (vpf_h[i] > max)
        {
            max = vpf_h[i];
        }
    }
    
    float norm = 255 / max;
    for (int i = 0; i < eye.rows; i++)
    {
        vpf_h[i] = roundf(vpf_h[i] * norm);
        cout << vpf_h[i] << endl;
    }
    
    Mat graph = Mat::zeros(256, eye.cols, CV_8U);
    
    for (int i = 0; i < eye.rows; i++)
    {
        graph.at<uchar>(vpf_h[i], i) = 255;
    }
    
    //showWindowAtPosition( windowName + " graph", graph, windowX, windowY + 330);
}

void findEyeCorners ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    showWindowAtPosition( windowName + " orig", eye, windowX, windowY);
    
    Mat leftEyeCorner = eye(Rect(0, 0, eye.size().width * 0.5, eye.size().height));
    showWindowAtPosition( windowName + " cropped", leftEyeCorner, windowX, windowY + 110);
    
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat::zeros( eye.size(), CV_32FC1 );
    
    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    
    /// Detecting corners
    cornerHarris( leftEyeCorner, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
    
    /// Normalizing
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    
    /// Drawing a circle around corners
    for( int j = 0; j < dst_norm.rows ; j++ )
    {
        for( int i = 0; i < dst_norm.cols; i++ )
        {
            if( (int) dst_norm.at<float>(j,i) > HCParam1 )
            {
                circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }
    
    /// Showing the result
    showWindowAtPosition( windowName + " corners", dst_norm_scaled, windowX, windowY + 220);
    
}

double computeDynamicThreshold(const Mat &mat, double stdDevFactor)
{
    Scalar stdMagnGrad, meanMagnGrad;
    meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}

Mat orientationMap(const cv::Mat& mag, const cv::Mat& ori, double thresh = 1.0)
{
    Mat oriMap = Mat::zeros(ori.size(), CV_8UC3);
    Vec3b red(0, 0, 255);
    Vec3b cyan(255, 255, 0);
    Vec3b green(0, 255, 0);
    Vec3b yellow(0, 255, 255);
    for(int i = 0; i < mag.rows*mag.cols; i++)
    {
        float* magPixel = reinterpret_cast<float*>(mag.data + i*sizeof(float));
        if(*magPixel > thresh)
        {
            float* oriPixel = reinterpret_cast<float*>(ori.data + i*sizeof(float));
            Vec3b* mapPixel = reinterpret_cast<Vec3b*>(oriMap.data + i*3*sizeof(char));
            if(*oriPixel < 90.0)
                *mapPixel = red;
            else if(*oriPixel >= 90.0 && *oriPixel < 180.0)
                *mapPixel = cyan;
            else if(*oriPixel >= 180.0 && *oriPixel < 270.0)
                *mapPixel = green;
            else if(*oriPixel >= 270.0 && *oriPixel < 360.0)
                *mapPixel = yellow;
        }
    }
    
    return oriMap;
}

void FCD(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    int64 e1 = getTickCount();
    
    GaussianBlur( eye, eye, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    // Gradient
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, grad;
    
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;
    
    /// Gradient X
    Sobel( eye, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    /// Gradient Y
    Sobel( eye, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    
    Mat direction, acceptedDirectionPares;
    
    acceptedDirectionPares = eye.clone();
    // Obarveni na cerno - to by asi slo udelat lip - napr. vytvorit Mat se stejnymi rozmery jako grad, ale rovnou cerny, nebo tak neco...
    for (int i = 0; i < acceptedDirectionPares.cols; i++)
    {
        for (int j = 0; j < acceptedDirectionPares.rows; j++)
        {
            acceptedDirectionPares.at<uchar>(i, j) = 0;
        }
    }
    
    phase(grad_x, grad_y, direction, true);
    
    double m_pi180 = M_PI / 180;
    double _180m_pi = 180 / M_PI;
    
    vector<Vec4f> pairVectors;
    
    for (int i = 0; i < direction.cols; i++)
    {
        for (int j = 0; j < direction.rows; j++)
        {
            for (int k = i; k < direction.cols; k++)
            {
                for (int l = j + 1; l <= direction.rows; l++)
                {
                    //            for (int k = 0; k < direction.cols; k++)
                    //            {
                    //                for (int l = 0; l < direction.rows; l++)
                    //                {
                    // Pokud nemaji nulovou velikost gradientu
                    if (abs_grad_x.at<uchar>(i, j) != 0 && abs_grad_x.at<uchar>(k, l) != 0)
                    {
                        int opositeAngleTresh = 4;
                        // musi mit cca opacny uhel
                        if( abs( direction.at<float>(i, j) - direction.at<float>(k, l) ) > ( 180 - opositeAngleTresh) &&
                           abs( direction.at<float>(i, j) - direction.at<float>(k, l) ) < ( 180 + opositeAngleTresh) )
                        {
                            float rad1 = direction.at<float>(i, j) * m_pi180;
                            Vec2f directionVec1 = Vec2f( cos(rad1), sin(rad1));
                            
                            //                            float rad2 = direction.at<float>(k, l) * m_pi180;
                            //                            Vec2f directionVec2 = Vec2f( cos(rad2), sin(rad2));
                            //
                            //                            Vec2f p1p2 = directionVec1 - directionVec2;
                            //
                            //                            float angle = acos( ( directionVec1[0] * p1p2[0] + directionVec1[1] * p1p2[0] ) / (sqrt(pow(directionVec1[0], 2) + pow(directionVec1[1], 2))) * (sqrt(pow(p1p2[0], 2) + pow(p1p2[1], 2))) );
                            
                            
                            // ***************************************
                            // pixely - alternative 2
                            //                            Point p1p2_B = Point(l - j,k - i);
                            Point p1p2_B = Point(j - l,i - k);
                            
                            float magnitude = sqrt(pow(p1p2_B.x, 2) + pow(p1p2_B.y, 2));
                            Point2f p1p2Normalized_B = Point2f( p1p2_B.x / magnitude, p1p2_B.y / magnitude );
                            
                            Point2f v1 = Point2f(directionVec1[0], directionVec1[1]);
                            
                            float angle_B = acos( v1.dot(p1p2Normalized_B) / (sqrt ( pow(v1.x, 2) + pow(v1.y, 2)) * sqrt(pow(p1p2Normalized_B.x, 2) + pow(p1p2Normalized_B.y, 2))) ) * _180m_pi;
                            
                            //cout << "angle = " << angle_B << endl;
                            
                            // Uhel mezi p1p2 a v1 kolem 0
                            int angleTresh = 4;  // 4
                            if (abs(angle_B) < angleTresh)
                            {
                                pairVectors.push_back(Vec4f(j, i, l, k));
                                
                                // debug obarveni
                                acceptedDirectionPares.at<uchar>(i, j) = 255;
                                acceptedDirectionPares.at<uchar>(k, l) = 255;
                                //                                acceptedDirectionPares.at<uchar>(l, k) = grad.at<uchar>(l, k);
                                //                                acceptedDirectionPares.at<uchar>(j, i) = grad.at<uchar>(j, i);
                            }
                        }
                        
                        //acceptedDirectionPares.at<uchar>(k, l) = grad.at<uchar>(k, l);
                    }
                }
            }
        }
    }
    
    // *********************** TMP **************************
    //25,11 ; 46,36
    //pairVectors.clear();
    //pairVectors.push_back(Vec4f(25, 11, 46, 36));
    
    vector<Vec4f> pairVectorsWithRadius;
    int minRad = 5, maxRad = 20;
    
    for (int i = 0; i < pairVectors.size(); i++)
    {
        Vec2f vec = Vec2f(pairVectors[i][0] - pairVectors[i][2], pairVectors[i][1] - pairVectors[i][3]);
        
        float mag = sqrt(pow(vec[0], 2) + pow(vec[1], 2));
        
        if (mag > minRad*2 && mag < maxRad*2)
        {
            pairVectorsWithRadius.push_back(Vec4f(pairVectors[i][0], pairVectors[i][1], pairVectors[i][2], pairVectors[i][3]));
        }
    }
    
    // TODO: zmensit pole jenom na pouzite radiusy. Ted je tam zbytecne 0 az minRadius-1
    int accumulator[abs_grad_x.cols][abs_grad_x.rows][maxRad];
    for (int i = 0; i < abs_grad_x.cols; ++i)
    {
        for (int j = 0; j < abs_grad_x.rows; ++j)
        {
            for (int k = 0; k < maxRad; ++k)
            {
                accumulator[i][j][k] = 0;
            }
        }
    }
    
    for (int i = 0; i < pairVectorsWithRadius.size(); ++i)
    {
        // *********************** TMP **************************
        //25,11 ; 46,36
        //acceptedDirectionPares.at<uchar>(pairVectorsWithRadius[i][1], pairVectorsWithRadius[i][0]) = 255;
        //acceptedDirectionPares.at<uchar>(pairVectorsWithRadius[i][3], pairVectorsWithRadius[i][2]) = 255;
        
        Vec2f vec = Vec2f(abs(pairVectorsWithRadius[i][0] - pairVectorsWithRadius[i][2]), abs(pairVectorsWithRadius[i][1] - pairVectorsWithRadius[i][3]));   // abs??
        //acceptedDirectionPares.at<uchar>(vec[1], vec[0]) = 255;
        
        float mag = sqrt(pow(vec[0], 2) + pow(vec[1], 2));
        //        Vec2i center = Vec2i(lround(vec[0] * 0.5), lround(vec[1] * 0.5));
        //        center += Vec2i(pairVectorsWithRadius[i][0], pairVectorsWithRadius[i][1]);
        Vec2i center = Vec2f((pairVectorsWithRadius[i][0] + pairVectorsWithRadius[i][2]), (pairVectorsWithRadius[i][1] + pairVectorsWithRadius[i][3])) * 0.5;
        
        if (center[0] >= abs_grad_x.cols)
            center[0] = abs_grad_x.cols -1;
        if (center[1] >= abs_grad_x.rows)
            center[1] = abs_grad_x.rows -1;
        
        accumulator[center[0]][center[1]][lround(mag * 0.5)]++;
        //acceptedDirectionPares.at<uchar>(center[1], center[0]) = 255;
    }
    
    //    for (int i = 0; i < grad.cols; ++i)
    //    {
    //        for (int j = 0; j < grad.rows; ++j)
    //        {
    //            for (int k = minRad; k < maxRad; ++k)
    //            {
    //                if (accumulator[i][j][k] != 0)
    //                    cout << i << ", " << j << ", " << k << " = " << accumulator[i][j][k] << endl;
    //            }
    //        }
    //    }
    
    Vec3i bestCircle;
    int max = 0;
    for (int i = 0; i < abs_grad_x.cols; ++i)
    {
        for (int j = 0; j < abs_grad_x.rows; ++j)
        {
            for (int k = minRad; k < maxRad; ++k)
            {
                if (accumulator[i][j][k] > max)
                {
                    max = accumulator[i][j][k];
                    bestCircle = Vec3i(i, j, k);
                }
            }
            
            //circle(frame, Point(bestCircle[0], bestCircle[1]), bestCircle[2], CV_RGB(0, 0, 255));
        }
    }
    cout << "best cirlce is " << bestCircle << " with max = " << max << endl;
    circle(frame, Point(bestCircle[0] + frameX, bestCircle[1] + frameY), bestCircle[2], CV_RGB(255, 0, 0));
    
    //    // zobrazeni akumulatoru
    //    Mat accMat = grad.clone();
    //    for (int i = 0; i < accMat.cols; ++i)
    //    {
    //        max = 0;
    //        for (int j = 0; j < accMat.rows; ++j)
    //        {
    //            for (int k = minRad; k < maxRad; ++k)
    //            {
    //                if (accumulator[i][j][k] > max)
    //                {
    //                    max = accumulator[i][j][k];
    //                    bestCircle = Vec3i(i, j, k);
    //                }
    //            }
    //
    //            accMat.at<uchar>(j, i) = max;
    //        }
    //    }
    
    
    double time = (getTickCount() - e1)/ getTickFrequency();
    cout << endl << "time = " << time << endl;
    
    //showWindowAtPosition( windowName + "grad", abs_grad_x, windowX, windowY);
    //showWindowAtPosition( windowName + "_direction", mat2gray(direction), windowX, windowY + 130);
    //showWindowAtPosition( windowName + "_accepted dir", acceptedDirectionPares, windowX, windowY + 260);
    //showWindowAtPosition( windowName + "_acc", accMat, windowX, windowY + 390);
    
}

int tmpLids = 0;
void findEyeLidsOTSU(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
#if TIME_MEASURING
    double time_time;
    int64 time_wholeFunc = getTickCount();
#endif
    
    //showWindowAtPosition( windowName + " pre", eye, windowX, windowY);
    
    Mat blurredEye;
    
    // blur
    medianBlur(eye, blurredEye, 7); //5
    //GaussianBlur( eye, blurredEye, Size(5, 5), 0, 0, BORDER_DEFAULT );
    
    Mat intensiveEye(blurredEye.rows, blurredEye.cols, CV_8U);
    int intesMul = 7;   //5
    
    for (int i = 0; i < intensiveEye.cols; i++)
    {
        for (int j = 0; j < intensiveEye.rows; j++)
        {
            int intensity = blurredEye.at<uchar>(j, i) * intesMul;
            if (intensity > 255)
                intensity = 255;
            intensiveEye.at<uchar>(j, i) = intensity;
        }
    }
    
    //showWindowAtPosition( windowName + " post intensity", intensiveEye, windowX, windowY + 130);
    
    
    Mat threshold_output;
    threshold( intensiveEye, threshold_output, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
    // showWindowAtPosition( windowName + " otsu ", threshold_output, windowX, windowY + 260);
    
    
    
    //Mat srcOutput = eye.clone();
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    /// Find contours
    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );
    
    /// Find the rotated rectangles and ellipses for each contour
    vector<RotatedRect> minRect( contours.size() );
    vector<RotatedRect> minEllipse( contours.size() );
    
    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( Mat(contours[i]) );
        if( contours[i].size() > 5 )
        {
            
            minEllipse[i] = fitEllipse( Mat(contours[i]) );
            //cout << minEllipse[i].angle << endl;
        }
    }
    
    /// Draw contours + rotated rects + ellipses
    //    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    //cout << "contours.size() " << contours.size() << endl;
    for( size_t i = 0; i< contours.size(); i++ )
    {
        //        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        // contour
        //        drawContours( drawing, contours, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        // ellipse
        //        ellipse( drawing, minEllipse[i], color, 2, 8 );
        //        rectangle(drawing, minRect[i].boundingRect().tl(), minRect[i].boundingRect().br(), color);
        
        
        // rotated rectangle
        //        cout << i << " minRect[i].size().width " << minRect[i].size.width << " > " << eye.size().width * 0.15 << endl;
        //        cout << i << " minRect[i].size().height " << minRect[i].size.height << " > " << eye.size().height * 0.15 << endl;
        
        /*float pomer = 0.f;
         if (minRect[i].size.width > minRect[i].size.height)
         pomer = minRect[i].size.width / minRect[i].size.height;
         if (minRect[i].size.height > minRect[i].size.width)
         pomer = minRect[i].size.height / minRect[i].size.width;
         cout << i <<  " pomer: " << pomer << endl;
         */
        
        /*cout
         << (minRect[i].size.width > eye.size().width * 0.15)
         << (minRect[i].size.height > eye.size().height * 0.15)
         << (minRect[i].center.y > eye.size().height * 0.35)
         << (minRect[i].center.y < eye.size().height * 0.65 ) << endl;
         */
        
        if (minRect[i].size.width > eye.size().width * 0.15 && minRect[i].size.height > eye.size().height * 0.15 && minRect[i].center.y > eye.size().height * 0.35 && minRect[i].center.y < eye.size().height * 0.65 )
        {
            //drawContours( srcOutput, contours, (int)i, color, 2, 8, vector<Vec4i>(), 0, Point() );
            //namedWindow( windowName + "srcOutput", WINDOW_AUTOSIZE );
            //imshow( windowName + "srcOutput", srcOutput );
            
            //            ellipse( eye, minEllipse[i], CV_RGB(255, 255, 255), 2, 8 );
            //rectangle(eye, minRect[i].boundingRect().tl(), minRect[i].boundingRect().br(), CV_RGB(255, 255, 255));
            
            Point framePoint = Point(frameX, frameY);
            Point leftTopPoint = Point(minRect[i].center.x - 0.35f * eye.size().width,  minEllipse[i].boundingRect().tl().y);
            Point rightTopPoint = Point(minRect[i].center.x + 0.35f * eye.size().width, minEllipse[i].boundingRect().tl().y);
            line( eye, leftTopPoint, rightTopPoint, CV_RGB(255, 255, 255), 2, 8 );
            //line( frame, leftTopPoint + framePoint, rightTopPoint + framePoint, CV_RGB(255, 255, 255), 2, 8 );
            
            
            Point leftBottomPoint = Point(minRect[i].center.x - 0.35f * eye.size().width, minEllipse[i].boundingRect().br().y);
            Point rightBottomPoint = Point(minRect[i].center.x + 0.35f * eye.size().width, minEllipse[i].boundingRect().br().y);
            line( eye, leftBottomPoint, rightBottomPoint, CV_RGB(255, 255, 255), 2, 8 );
            //line( frame, leftBottomPoint + framePoint, rightBottomPoint + framePoint, CV_RGB(255, 255, 255), 2, 8 );
            
            eyeLids.push_back(Vec4f(leftTopPoint.x + framePoint.x, leftTopPoint.y + framePoint.y, rightTopPoint.x + framePoint.x, rightTopPoint.y + framePoint.y));
            eyeLids.push_back(Vec4f(leftBottomPoint.x + framePoint.x, leftBottomPoint.y + framePoint.y, rightBottomPoint.x + framePoint.x, rightBottomPoint.y + framePoint.y));
            
            
            break;
        }
    }
    
    //    namedWindow( windowName + " Contours", WINDOW_AUTOSIZE );
    //    imshow( windowName + " Contours", drawing );
    
    //showWindowAtPosition( windowName + " tmp", eye, windowX, windowY + 380);
    
    
#if TIME_MEASURING
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "find eye lids otsu time = " << time_time << endl;
#endif
    tmpLids++;
    //methodTime += time_time;
}

void findPupil(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY, Point center)
{
#if TIME_MEASURING
    int64 e1 = getTickCount();
#endif
    
    //showWindowAtPosition( windowName + " eye", eye, windowX, windowY + 0);
    
    Mat gaussEye;
    GaussianBlur( eye, gaussEye, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    Mat intensiveEye = gaussEye.clone();
    //Mat intensiveEye = eye.clone();
    const int intesMul = 6;
    
    for (int i = 0; i < intensiveEye.cols; i++)
    {
        for (int j = 0; j < intensiveEye.rows; j++)
        {
            int intensity = intensiveEye.at<uchar>(j, i) * intesMul;
            if (intensity > 255)
                intensity = 255;
            intensiveEye.at<uchar>(j, i) = intensity;
        }
    }
    
    //showWindowAtPosition( windowName + " intensive eye", intensiveEye, windowX, windowY + 120);
    
    int minRadius = 3;
    int maxRadius = 7;
    
    int intensitiesCount = maxRadius - minRadius + 1;
    double intensities[intensitiesCount];
    for (int i = 0; i < intensitiesCount; i++)
    {
        intensities[i] = 0;
    }
    
    int i = 0;
    int stepsCount = 0;
    double totalIntensity = 0;
    for (int r = minRadius; r <= maxRadius; ++r)
    {
        double step = 2* M_PI / (r*2);
        
        for(double theta = 0;  theta < 2 * M_PI;  theta += step)
            //for(double theta = M_PI;  theta <= 2 * M_PI;  theta += step)    // Spodni oblouk - neni stineny rasami
        {
            int circleX = lround(center.x + r * cos(theta));
            int circleY = lround(center.y - r * sin(theta));
            
            int pixelIntens = intensiveEye.at<uchar>(circleY, circleX);
            if (pixelIntens < 250)
            {
                totalIntensity += pixelIntens;
                
                //cout << pixelIntens << endl;
                stepsCount++;
            }
        }
        
        if (stepsCount == 0)
            stepsCount = 1;
        intensities[i] = totalIntensity / (r * stepsCount);
        
        //cout << intensities[i] << endl;
        
        i++;
    }
    //cout << endl;
    
    double minIntens = 0;
    double minIntensRad = 0;
    for (i = 0; i < intensitiesCount; i++)
    {
        if (intensities[i] > minIntens)
        {
            minIntens = intensities[i];
            minIntensRad = i;
        }
    }
    
    minIntensRad += minRadius;
    
    pupils.push_back(Vec3f(center.x + frameX, center.y+ frameY, minIntensRad));
    
    cvtColor(intensiveEye, intensiveEye, CV_GRAY2BGR);
    circle(intensiveEye, center, minIntensRad, CV_RGB(0, 255, 0));
    //showWindowAtPosition( windowName + " pupil", intensiveEye, windowX, windowY + 240);
    
    cvtColor(eye, eye, CV_GRAY2BGR);
    circle(eye, center, minIntensRad, CV_RGB(0, 255, 0));
    //showWindowAtPosition( windowName + " E pupil", eye, windowX, windowY + 360);
    
#if TIME_MEASURING
    double time = (getTickCount() - e1)/ getTickFrequency();
    cout << "find pupil time = " << time << endl;
#endif
    
    //methodTime += time;
}

Point myHoughCircle(Mat eye, int kernel, string windowName, int windowX, int windowY, int frameX, int frameY, Point center)
{
#if TIME_MEASURING
    int64 e1 = getTickCount();
#endif
    
    //showWindowAtPosition( windowName + "PRE eye hough", eye, windowX, windowY );
    
    Mat gaussEye;
    GaussianBlur( eye, gaussEye, Size(5, 5), 0, 0, BORDER_DEFAULT );
    
    Mat intensiveEye = gaussEye.clone();
    int intesMul = 4;
    
    for (int i = 0; i < intensiveEye.cols; i++)
    {
        for (int j = 0; j < intensiveEye.rows; j++)
        {
            int intensity = intensiveEye.at<uchar>(j, i) * intesMul;
            if (intensity > 255)
                intensity = 255;
            intensiveEye.at<uchar>(j, i) = intensity;
        }
    }
    
    //showWindowAtPosition( windowName + "intensiveEye eye hough", intensiveEye, windowX, windowY );
    
    // Gradient
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, grad;
    
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    /// Gradient X
    Sobel( intensiveEye, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    /// Gradient Y
    Sobel( intensiveEye, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    //showWindowAtPosition( windowName + "grad", grad, windowX, windowY + 120);
    
    
    grad = mat2gray(abs_grad_x);
    //showWindowAtPosition( windowName + "grad X", grad, windowX, windowY + 240);
    
    /*
     eye iris normalised error for 0.05 = 39.7959%
     eye iris normalised error for 0.1 = 94.898%
     eye iris normalised error for 0.15 = 100%
     eye iris normalised error for 0.2 = 100%
     eye iris normalised error for 0.25 = 100%
     */
    
    // polomery
    int minRadius = 4, maxRadius = eye.size().width * 0.3	;
    
    int gradientsCount = maxRadius - minRadius + 1;
    double gradients[kernel][kernel][gradientsCount];
    
    // nulovani
    for (int i = 0; i < kernel; ++i)
    {
        for (int j = 0; j < kernel; ++j)
        {
            for (int k = 0; k < gradientsCount; ++k)
            {
                gradients[i][j][k] = 0;
            }
        }
    }
    
    int halfKernel = (kernel - 1) * 0.5;
    // hranice
    int xMin = center.x - halfKernel;
    if (xMin < 0)
        xMin = 0;
    int xMax = center.x + halfKernel;
    if (xMax > eye.size().width)
        xMax = eye.size().width;
    int yMin = center.y - halfKernel;
    if (yMin < 0)
        yMin = 0;
    int yMax = center.y + halfKernel;
    if (yMax > eye.size().height)
        yMax = eye.size().height;
    
    // cout << "center = " << center.x << ", " << center.y << endl;
    // cout << xMin << ", " << xMax << ", " << yMin << ", " << yMax << endl;
    
    if (kernel == 1)
    {
        xMin = xMax = center.x;
        yMin = yMax = center.y;
        
    }
    
    double m_pi180 = M_PI / 180;
    //Mat tmpMat = grad.clone(); //Mat::zeros(grad.size(), CV_8U);
    int i = 0;
    for (int x = xMin; x <= xMax; ++x)
    {
        int j = 0;
        for (int y = yMin; y <= yMax; ++y)
        {
            int k = 0;
            for (int r = minRadius; r <= maxRadius; ++r)
            {
                double step = 2* M_PI / (r*2);
                
                int stepsCount = 0;
                //for(double theta = 0;  theta < 2 * M_PI;  theta += step)
                for(double theta = 120 * m_pi180;  theta <= 240 * m_pi180;  theta += step)
                {
                    int circleX = lround(x + r * cos(theta));
                    int circleY = lround(y - r * sin(theta));
                    
                    gradients[i][j][k] += grad.at<uchar>(circleY, circleX);
                    //tmpMat.at<uchar>(circleY, circleX) = 127;
                    
                    stepsCount++;
                }
                for(double theta = -60 * m_pi180;  theta <= 60 * m_pi180;  theta += step)
                {
                    int circleX = lround(x + r * cos(theta));
                    int circleY = lround(y - r * sin(theta));
                    
                    gradients[i][j][k] += grad.at<uchar>(circleY, circleX);
                    //tmpMat.at<uchar>(circleY, circleX) = 127;
                    
                    stepsCount++;
                }
                
                gradients[i][j][k] /= stepsCount;
                
                k++;
            }
            
            j++;
        }
        
        i++;
    }
    
    
    double maxGrad = 0;
    double maxGradRad = 0;
    
    Point newCenter = center;
    
    i = 0;
    for (int x = xMin; x <= xMax; ++x)
    {
        int j = 0;
        for (int y = yMin; y <= yMax; ++y)
        {
            for (int k = 0; k < gradientsCount; ++k)
            {
                if (gradients[i][j][k] > maxGrad)
                {
                    maxGrad = gradients[i][j][k];
                    maxGradRad = k;
                    
                    newCenter.x = x;
                    newCenter.y = y;
                }
            }
            
            j++;
        }
        
        i++;
    }
    
    
    maxGradRad += minRadius;
    
    //showWindowAtPosition( windowName + " TMP", tmpMat, windowX, windowY + 130);
    
    //cout << "max grad = " << maxGrad << " s rad = " << maxGradRad << endl << endl;
    
    // drawing
    //showWindowAtPosition( windowName + "_nova oblast", grad, windowX, windowY + 230);
    
    //cvtColor(grad, grad, CV_GRAY2BGR);
    
    //	Scalar color = Scalar(0, 0, 255);
    //	int lineLength = 10;
    //	line(grad, Point(center.x - lineLength*0.5, center.y), Point(center.x + lineLength*0.5, center.y), color);
    //	line(grad, Point(center.x, center.y - lineLength*0.5), Point(center.x, center.y + lineLength*0.5), color);
    
    // min max radius circle
    //circle( grad, newCenter, minRadius, CV_RGB(0, 0, 255));
    //circle( grad, newCenter, maxRadius, CV_RGB(0, 0, 255));
    
    //circle( grad, newCenter, maxGradRad, CV_RGB(255, 0, 0));
    
    irises.push_back(Vec3f(newCenter.x + frameX, newCenter.y + frameY, maxGradRad));
    
    
    //showWindowAtPosition( windowName + "_nova oblast + cicles", grad, windowX, windowY + 230 );
    
    
    //	cvtColor(eye, eye, CV_GRAY2BGR);
    //	circle(eye, newCenter, maxGradRad, color);
    //	showWindowAtPosition( windowName + "_nova oblast + eye", eye, windowX, windowY + 330  );
    
    //showWindowAtPosition( windowName + "POST eye hough", eye, windowX, windowY + 390);
    //findPupil(eye(Rect(center.x - maxGradRad, center.y - maxGradRad, maxGradRad*2, maxGradRad*2)), windowName, windowX, windowY, frameX, frameY);
    //findPupil(frame(Rect(frameX + center.x - maxGradRad, frameY + center.y - maxGradRad, maxGradRad*2, maxGradRad*2)), windowName, windowX, windowY, frameX, frameY);
    //showWindowAtPosition( windowName +  + "eye", frame(Rect(frameX, frameY, eye.size().width, eye.size().height)), windowX, windowY + 650);
    
#if TIME_MEASURING
    double time = (getTickCount() - e1)/ getTickFrequency();
    cout << "My hough circle time = " << time << endl;
#endif
    
    //    Point frameCenter(newCenter.x + frameX, newCenter.y + frameY);
    //    eyesCentres.push_back(frameCenter);
    
    
    //methodTime += time;
    return newCenter;
}


void drawIrises()
{
    for( size_t i = 0; i < irises.size(); i++ )
    {
        int radius = cvRound(irises[i][2]);
        
        Point frameCenter( cvRound(irises[i][0]), cvRound(irises[i][1]) );
        // circle outline
        Scalar color = Scalar(255, 0, 255);
        
        circle( frame, frameCenter, radius, color);
    }
}

void drawPupils()
{
    for( int i = 0; i < pupils.size(); i++ )
    {
        int radius = cvRound(pupils[i][2]);
        
        Point frameCenter( cvRound(pupils[i][0]), cvRound(pupils[i][1]) );
        // circle outline
        Scalar color = CV_RGB(0, 255, 0);
        
        circle( frame, frameCenter, radius, color);
    }
}

void drawEyesCentres()
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

void drawEyeLids()
{
    Scalar color = Scalar(255, 255, 255);
    
    for( size_t i = 0; i < eyeLids.size(); i++ )
    {
        line(frame, Point(eyeLids[i][0], eyeLids[i][1]), Point(eyeLids[i][2], eyeLids[i][3]), color);
    }
}

void showWindowAtPosition( string imageName, Mat mat, int x, int y )
{
    imshow( imageName, mat );
    moveWindow(imageName, x, y);
}



// ************ Testing ************

// Just for 4 digits number
int digitsCount(int x)
{
    x = abs(x);
    return (x < 10 ? 1 :
            (x < 100 ? 2 :
             (x < 1000 ? 3 :
              (x < 10000 ? 4 :
               5))));
}

void readEyeData(string fullEyeDataFilePath, vector<Point> &dataEyeCentres)
{
    string line;
    ifstream myfile (fullEyeDataFilePath);
    if (myfile.is_open())
    {
        int lineNum = 0;
        int lx, ly, rx, ry;
        while ( getline (myfile,line) )
        {
            if (lineNum == 1)
            {
                //cout << line << endl;
                char lineBuff[64];
                strcpy(lineBuff, line.c_str());
                
                sscanf (lineBuff," %d %d %d %d", &lx, &ly, &rx, &ry);
                //printf ("%d %d %d %d", lx, ly, rx, ry);
                
                dataEyeCentres.push_back(Point(lx, ly));
                dataEyeCentres.push_back(Point(rx, ry));
                
                break;
            }
            
            lineNum++;
        }
        
        myfile.close();
    }
    else
    {
        cout << "Unable to open file";
    }
}

void computeEyeCentreDistances(Point dataLeftEye, Point dataRigtEye, Point myLeftEye, Point myRightEye, vector<double> &eyeCentreDistances)
{
    double el = norm(dataLeftEye - myLeftEye);
    double er = norm(dataRigtEye - myRightEye);
    double distance = norm(dataLeftEye - dataRigtEye);
    distance = 0. ? 1. : distance;
    
    eyeCentreDistances.push_back(MAX(el, er) / distance);
}

double getAvgEyeCentreNormalisedError(const vector<double> &eyeCentreDistances)
{
    double sum = 0.;
    
    for (int i = 0; i < eyeCentreDistances.size(); i++)
    {
        sum += eyeCentreDistances[i];
    }
    
    return sum / eyeCentreDistances.size();
}

double getNormalisedError(const vector<double> &eyeCentreDistances, double e)
{
    int count = 0;
    
    for (int i = 0; i < eyeCentreDistances.size(); i++)
    {
        if (eyeCentreDistances[i] <= e)
            count++;
    }
    
    
    return count / (double)eyeCentreDistances.size();
}

void testFaceDetection()
{
    double time_time;
    int64 time_wholeFunc = getTickCount();
    
#if XCODE
    string folder = "../../../res/pics/BioID-FaceDatabase-V1.2/";
#else
    string folder = "../res/pics/BioID-FaceDatabase-V1.2/";
#endif
    
    string prefix = "BioID_";
    string imgSuffix = ".pgm", dataSuffix = ".eye";
    
    int fileCount = 1521;   //1521;
    vector<double> eyeCentreDistances;
    
    char numstr[21]; // enough to hold all numbers up to 64-bits
    string imgFullFilePath = "", dataFullFilePath = "";
    for (int i = 0; i < fileCount; i++)
    {
        vector<Point> dataEyeCentres;
        
        switch (digitsCount(i))
        {
            case 1:
                sprintf(numstr, "000%d", static_cast<int>(i));
                break;
                
            case 2:
                sprintf(numstr, "00%d", static_cast<int>(i));
                break;
            case 3:
                sprintf(numstr, "0%d", static_cast<int>(i));
                break;
            default:
                sprintf(numstr, "%d", static_cast<int>(i));
                break;
        }
        
        imgFullFilePath = folder + prefix + numstr + imgSuffix;
        
        frame = imread(imgFullFilePath);
        
        originalFrame = frame.clone();
        detectAndDisplay(frame);
        
        imgFullFilePath = folder + "results/" + prefix + numstr +"_result" + imgSuffix;
        //imwrite(imgFullFilePath, frame);
    }
    
    cout << "face count = " << facesDetectedCount << ", that's " << facesDetectedCount / (double)fileCount * 100 << "% face detection" << endl;
    
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "faceTestDetection time = " << time_time << endl;
}

void testEyeDetection()
{
    double time_time;
    int64 time_wholeFunc = getTickCount();
    
#if XCODE
    string folder = "../../../res/pics/BioID-FaceDatabase-V1.2/";
#else
    string folder = "../res/pics/BioID-FaceDatabase-V1.2/";
#endif
    
    string prefix = "BioID_";
    string imgSuffix = ".pgm", dataSuffix = ".eye";
    
    int fileCount = 1521;   //1521;
    vector<double> eyeCentreDistances;
    
    char numstr[21]; // enough to hold all numbers up to 64-bits
    string imgFullFilePath = "", dataFullFilePath = "";
    for (int i = 0; i < fileCount; i++)
    {
        vector<Point> dataEyeCentres;
        
        switch (digitsCount(i))
        {
            case 1:
                sprintf(numstr, "000%d", static_cast<int>(i));
                break;
                
            case 2:
                sprintf(numstr, "00%d", static_cast<int>(i));
                break;
            case 3:
                sprintf(numstr, "0%d", static_cast<int>(i));
                break;
            default:
                sprintf(numstr, "%d", static_cast<int>(i));
                break;
        }
        
        imgFullFilePath = folder + prefix + numstr + imgSuffix;
        
        frame = imread(imgFullFilePath);
        
        originalFrame = frame.clone();
        detectAndDisplay(frame);
        
        imgFullFilePath = folder + "results/" + prefix + numstr +"_result" + imgSuffix;
        //imwrite(imgFullFilePath, frame);
    }
    
    cout << "face count = " << facesDetectedCount << ", that's " << facesDetectedCount / (double)fileCount * 100 << "% face detection" << endl;
    
    cout << "eye count = " << eyesDetectedCount << ", that's " << eyesDetectedCount / (double)(facesDetectedCount * 2 ) * 100 << "% eye detection" << endl;
    
    
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "testEyeDetection time = " << time_time << endl;
}


void testDetection()
{
    double time_time;
    int64 time_wholeFunc = getTickCount();
    
#if XCODE
    string folder = "../../../res/pics/BioID-FaceDatabase-V1.2/";
#else
    string folder = "../res/pics/BioID-FaceDatabase-V1.2/";
#endif
    
    string prefix = "BioID_";
    string imgSuffix = ".pgm", dataSuffix = ".eye";
    
    int fileCount = 1521;   //1521;
    vector<double> eyeCentreDistances;
    vector<int> veryPrecciousEye;
    
    char numstr[21]; // enough to hold all numbers up to 64-bits
    string imgFullFilePath = "", dataFullFilePath = "";
    for (int i = 0; i < fileCount; i++)
    {
        vector<Point> dataEyeCentres;
        
        switch (digitsCount(i))
        {
            case 1:
                sprintf(numstr, "000%d", static_cast<int>(i));
                break;
                
            case 2:
                sprintf(numstr, "00%d", static_cast<int>(i));
                break;
            case 3:
                sprintf(numstr, "0%d", static_cast<int>(i));
                break;
            default:
                sprintf(numstr, "%d", static_cast<int>(i));
                break;
        }
        
        //TMP HAAAAK  *****************************************************************************  !!!!!!!!!!!!!!!!! ******************************
        //sprintf(numstr, "00%d", static_cast<int>(11));
        
        imgFullFilePath = folder + prefix + numstr + imgSuffix;
        
        
        
        
        frame = imread(imgFullFilePath);
        
        originalFrame = frame.clone();
        detectAndDisplay(frame);
        
        imgFullFilePath = folder + "results/" + prefix + numstr +"_result" + imgSuffix;
        //imwrite(imgFullFilePath, frame);
        
        // test data
        dataFullFilePath = folder + prefix + numstr + dataSuffix;
        // read cords from file
        readEyeData(dataFullFilePath, dataEyeCentres);
        
        
        // my cords
        Point myLeftEye, myRightEye;
        /*for (int j = 0; j < eyesCentres.size(); j++)
         {
         if (eyesCentres[j].x < frame.cols * 0.5)
         {
         myRightEye = eyesCentres[j];
         }
         else
         {
         myLeftEye = eyesCentres[j];
         }
         }*/
        
        
        
        if (eyesCentres.size() == 2)
        {
            myLeftEye = eyesCentres[0];
            myRightEye = eyesCentres[1];
            
            //            cout << myLeftEye << endl;
            //            cout << myRightEye << endl << endl;
            
            computeEyeCentreDistances(dataEyeCentres[0], dataEyeCentres[1], myLeftEye, myRightEye, eyeCentreDistances);
            
            if (eyeCentreDistances[eyeCentreDistances.size() - 1] <= 0.022)
            {
                veryPrecciousEye.push_back(i);
            }
        }
        
    }
    
    //    cout << "eye AVG centre normalised error = " << getAvgEyeCentreNormalisedError(eyeCentreDistances) << endl;
    //    cout << "eye centre normalised error = " << getEyeCentreNormalisedError(eyeCentreDistances, 0.1) * 100 << "%" << endl;
    
    double smallestE = 0.05;
    for (int i = 1; i <= 5; i++)
    {
        cout << "eye centre normalised error for " << smallestE * i << " = " <<  getNormalisedError(eyeCentreDistances, smallestE * i) * 100 << "%" << endl;
    }
    
    // vypis pro tabulku
    smallestE = 0.01;
    for (int i = 1; i <= 25; i++)
    {
        cout <<  getNormalisedError(eyeCentreDistances, smallestE * i) * 100 << endl;
    }
    
    
    cout<<endl;
    for (int i = 0; i < veryPrecciousEye.size(); i++)
    {
        cout << veryPrecciousEye[i] << ", " ;
    }
    
    cout << "size = " << veryPrecciousEye.size() << endl;
    
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "testDetection time = " << time_time << endl;
    
    cout << "method time = " << methodTime << endl;
    cout << "avg method time = " << methodTime / eyesDetectedCount << endl;
    
}


Vec6f readMyEyeData(string fullEyeDataFilePath)
{
    Vec6f myEyeData(1, 1, 1, 1, 1 , 1);
    
    string line;
    ifstream myfile (fullEyeDataFilePath);
    if (myfile.is_open())
    {
        //int lineNum = 0;
        // left diameter of iris, right diameter of iris, left upper lid, left bottom lid, right upper led, right bottom lid
        float ld, rd, lul, lbl, rul, rbl;
        
        while ( getline (myfile,line) )
        {
            //if (lineNum == 1)
            {
                //cout << line << endl;
                char lineBuff[64];
                strcpy(lineBuff, line.c_str());
                
                sscanf (lineBuff,"%f %f %f %f %f %f", &ld, &rd, &lul, &lbl, &rul, &rbl);
                
                myEyeData[0] = ld;
                myEyeData[1] = rd;
                myEyeData[2] = lul;
                myEyeData[3] = lbl;
                myEyeData[4] = rul;
                myEyeData[5] = rbl;
                
                cout << fullEyeDataFilePath << endl;
                cout << myEyeData << endl;
                
                break;
            }
            
            //lineNum++;
        }
        
        myfile.close();
    }
    else
    {
        cout << "Unable to open file";
    }
    
    return myEyeData;
}

void computeIrisesDistances(Point dataLeftEye, Point dataRigtEye, double myLeftEyeCentreIrisDistance, double myRightEyeCentreIrisDistance, double dataLeftEyeCentreIrisDistance, double dataRightEyeCentreIrisDistance,vector<double> &irisesDistances)
{
    double el = abs(myLeftEyeCentreIrisDistance - dataLeftEyeCentreIrisDistance);
    double er = abs(myRightEyeCentreIrisDistance - dataRightEyeCentreIrisDistance);
    
    double distance = norm(dataLeftEye - dataRigtEye);
    distance = 0. ? 1. : distance;
    
    irisesDistances.push_back(MAX(el, er) / distance);
}

void testIrisDetection()
{
    double time_time;
    int64 time_wholeFunc = getTickCount();
    
#if XCODE
    string folder = "../../../res/pics/BioID-FaceDatabase-V1.2/";
#else
    string folder = "../res/pics/BioID-FaceDatabase-V1.2/";
#endif
    
    string prefix = "BioID_";
    string imgSuffix = ".pgm", dataSuffix = ".eye", myDataSuffix = ".myEye";
    vector<double> irisesDistances;
    
    int fileCount = 1521;
    //int files[] = { 10, 16, 17, 27, 43, 50, 52, 58, 62, 64, 65, 66, 71, 73, 75, 77, 78, 82, 87, 101, 102, 104, 107, 110, 111, 128, 167, 175, 187, 188, 197, 212, 213, 218, 219, 220, 249, 265, 266, 267, 268, 280, 281, 284, 285, 286, 300, 305, 373, 420, 433, 444, 446, 482, 488, 513, 560, 574, 576, 577, 580, 581, 582, 585, 598, 604, 617, 630, 649, 662, 670, 672, 674, 675, 677, 685, 823, 827, 832, 854, 875, 880, 928, 943, 1080, 1093, 1095, 1096, 1153, 1157, 1160, 1175, 1177, 1209, 1227, 1237, 1245, 1264, 1289, 1290};
    
    char numstr[21]; // enough to hold all numbers up to 64-bits
    string imgFullFilePath = "", dataFullFilePath = "", myDataFullFilePath = "";
    for (int i = 0; i < fileCount; i++)
    {
        vector<Point> dataEyeCentres;
        
        switch (digitsCount(i))
        {
            case 1:
                sprintf(numstr, "000%d", static_cast<int>(i));
                break;
                
            case 2:
                sprintf(numstr, "00%d", static_cast<int>(i));
                break;
            case 3:
                sprintf(numstr, "0%d", static_cast<int>(i));
                break;
            default:
                sprintf(numstr, "%d", static_cast<int>(i));
                break;
        }
        
        
        imgFullFilePath = folder + prefix + numstr + imgSuffix;
        
        
        frame = imread(imgFullFilePath);
        
        originalFrame = frame.clone();
        detectAndDisplay(frame);
        
        imgFullFilePath = folder + "results/" + prefix + numstr +"_result" + imgSuffix;
        imwrite(imgFullFilePath, frame);
        
        // test data
        dataFullFilePath = folder + prefix + numstr + dataSuffix;
        // read cords from file
        readEyeData(dataFullFilePath, dataEyeCentres);
        
        myDataFullFilePath = folder + prefix + numstr + myDataSuffix;
        Vec6f myEyeData = readMyEyeData(myDataFullFilePath);
        
        
        if (irises.size() == 2)
        {
            //computeIrisesDistances(dataEyeCentres[0], dataEyeCentres[1], irises[0][2], irises[1][2], myEyeData[0], myEyeData[1], irisesDistances);
        }
        
    }
    
    double smallestE = 0.05;
    for (int i = 1; i <= 5; i++)
    {
        cout << "eye iris normalised error for " << smallestE * i << " = " <<  getNormalisedError(irisesDistances, smallestE * i) * 100 << "%" << endl;
    }
    
    // vypis pro tabulku
    smallestE = 0.01;
    for (int i = 1; i <= 25; i++)
    {
        cout <<  getNormalisedError(irisesDistances, smallestE * i) * 100 << endl;
    }
    
    
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "testIrisDetection time = " << time_time << endl;
    
    cout << "method time = " << methodTime << endl;
    cout << "avg method time = " << methodTime / eyesDetectedCount << endl;
}

void testPupilsDetection()
{
    double time_time;
    int64 time_wholeFunc = getTickCount();
    
#if XCODE
    string folder = "../../../res/pics/BioID-FaceDatabase-V1.2/";
#else
    string folder = "../res/pics/BioID-FaceDatabase-V1.2/";
#endif
    
    string prefix = "BioID_";
    string imgSuffix = ".pgm", dataSuffix = ".eye", myDataSuffix = ".myEye";
    vector<double> irisesDistances;
    
    int fileCount = 1521;
    
    char numstr[21]; // enough to hold all numbers up to 64-bits
    string imgFullFilePath = "", dataFullFilePath = "", myDataFullFilePath = "";
    for (int i = 0; i < fileCount; i++)
    {
        vector<Point> dataEyeCentres;
        
        switch (digitsCount(i))
        {
            case 1:
                sprintf(numstr, "000%d", static_cast<int>(i));
                break;
                
            case 2:
                sprintf(numstr, "00%d", static_cast<int>(i));
                break;
            case 3:
                sprintf(numstr, "0%d", static_cast<int>(i));
                break;
            default:
                sprintf(numstr, "%d", static_cast<int>(i));
                break;
        }
        
        
        imgFullFilePath = folder + prefix + numstr + imgSuffix;
        
        
        frame = imread(imgFullFilePath);
        
        originalFrame = frame.clone();
        detectAndDisplay(frame);
    }
    
    
    
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    
    cout << "method time = " << methodTime << endl;
    cout << "avg method time = " << methodTime / eyesDetectedCount << endl;
}

void computeLidsDistances(Point dataLeftEye, Point dataRigtEye, const vector<Vec4f> &myLids, const Vec6f eyeData, vector<double> &irisesDistances)
{
    cout << myLids[0][1] << ", " << myLids[1][1] << ", " << myLids[2][1] << ", " << myLids[3][1] << endl;
    double elu = abs(myLids[0][1] - eyeData[2]);
    double elb = abs(myLids[1][1] - eyeData[3]);
    double eru = abs(myLids[2][1] - eyeData[4]);
    double erb = abs(myLids[3][1] - eyeData[5]);
    
    double distance = norm(dataLeftEye - dataRigtEye);
    distance = 0. ? 1. : distance;
    
    irisesDistances.push_back(MAX(elu, elb) / distance);
    irisesDistances.push_back(MAX(eru, erb) / distance);
}



void testLidsDetection()
{
    double time_time;
    int64 time_wholeFunc = getTickCount();
    
#if XCODE
    string folder = "../../../res/pics/BioID-FaceDatabase-V1.2/";
#else
    string folder = "../res/pics/BioID-FaceDatabase-V1.2/";
#endif
    
    string prefix = "BioID_";
    string imgSuffix = ".pgm", dataSuffix = ".eye", myDataSuffix = ".myEye";
    vector<double> lidsDistances;
    
    int fileCount = 1521;
    
    char numstr[21]; // enough to hold all numbers up to 64-bits
    string imgFullFilePath = "", dataFullFilePath = "", myDataFullFilePath = "";
    for (int i = 0; i < fileCount; i++)
    {
        vector<Point> dataEyeCentres;
        
        switch (digitsCount(i))
        {
            case 1:
                sprintf(numstr, "000%d", static_cast<int>(i));
                break;
                
            case 2:
                sprintf(numstr, "00%d", static_cast<int>(i));
                break;
            case 3:
                sprintf(numstr, "0%d", static_cast<int>(i));
                break;
            default:
                sprintf(numstr, "%d", static_cast<int>(i));
                break;
        }
        
        
        imgFullFilePath = folder + prefix + numstr + imgSuffix;
        
        
        frame = imread(imgFullFilePath);
        
        originalFrame = frame.clone();
        detectAndDisplay(frame);
        
        
    }
    
    
    
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "testLidsDetection time = " << time_time << endl;
    
    cout << "method time = " << methodTime << endl;
    cout << "avg method time = " << methodTime / eyesDetectedCount << endl;
}
