#import "constants.h"


/*#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"*/
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

#import "functions.h"


using namespace std;
using namespace cv;


// Function Headers
int processArguments( int argc, const char** argv );
void showUsage( string name );

void loadCascades();
void detectAndDisplay( Mat frame );

void refreshImage();
vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face);
Rect pickFace(vector<Rect> faces);


vector<Rect> sortEyes(const vector<Rect> &eyes, Rect face);


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
    
    //testDetection();
    //testFaceDetection();
    //testEyeDetection();
    //testIrisDetection();
    //testPupilsDetection();
    //testLidsDetection();c
    
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
                        frame = resizeMat(frame, imageWidth);
                        
                        originalFrame = frame.clone();
                        
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
                    frame = resizeMat(frame, imageWidth);
                    
                    originalFrame = frame.clone();
                    
                    if( !frame.empty() )
                    {
                        detectAndDisplay( frame );
                        
                        imshow( window_name, frame );
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
    //-- 1. Load the cascades
   	if( !faceCascade.load( faceCascadeName ) )
    {
        printf("--(!)Error loading\n");
    }
    
   	if( !eyesCascade.load( eyesCascadeName ) )
    {
        printf("--(!)Error loading\n");
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
            //findEyeLidsOTSU(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
            eyeLidsLocalisation(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeLids);
        }
        
        if (drawInFrame)
        {
            drawEyesCentres(eyesCentres, frame);
            //drawPupils(pupils, frame);
            drawIrises(irises, frame);
            drawEyeLids(eyeLids, frame);
        }
    }
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
            cout << "Mazu! Oblast oka mimo vrchni polovinu obliceje. x,y = " << correctEyes[i].x << ", " << correctEyes[i].y << ". Polovina obliceje ma delku " << face.size().height * 0.5 << endl;
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
                cout << "Mazu! Oblast praveho oka je mimo oblicej. x,y = " << eyes[i].x << ", " << eyes[i].y << endl;
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
                cout << "Mazu! Oblast leveho oka je mimo oblicej. x,y = " << eyes[i].x << ", " << eyes[i].y << endl;
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

void refreshImage()
{
    //if (stepFrame)
    {
        frame = originalFrame.clone();
        detectAndDisplay(frame);
    }
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