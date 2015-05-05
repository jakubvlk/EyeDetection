//
//  testing.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 30/04/15.
//
//

#include "testing.h"

#import "functions.h"
#import "constants.h"

#include <fstream>

#if XCODE
string folder = "../../../res/pics/BioID-FaceDatabase-V1.2/";
#else
string folder = "../res/pics/BioID-FaceDatabase-V1.2/";
#endif
string prefix = "BioID_";


// the function using the function pointers: - TEST
void somefunction(void (*fptr)(void*, Mat frame), void* context, Mat frame)
{
    fptr(0, frame);
    
    
}


void testEyeCenterDetection(void (*fptr_detectAndShow)(void*, Mat frame), void* context, Mat &frame, Mat &originalFrame, const vector<Point> &eyesCentres)
{
    double time_time;
    int64 time_wholeFunc = getTickCount();
    
    
    string imgSuffix = ".pgm", dataSuffix = ".eye";
    
    int fileCount = 1521;
    vector<double> eyeCentreDistances;
    
    char numstr[21];
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
        fptr_detectAndShow(0, frame);
        
        imgFullFilePath = folder + "results/" + prefix + numstr +"_result" + imgSuffix;
        //imwrite(imgFullFilePath, frame);
        
        // test data
        dataFullFilePath = folder + prefix + numstr + dataSuffix;
        // read cords from file
        readEyeData(dataFullFilePath, dataEyeCentres);
        
        
        // my cords
        Point myLeftEye, myRightEye;

        
        
        
        if (eyesCentres.size() == 2)
        {
            myLeftEye = eyesCentres[0];
            myRightEye = eyesCentres[1];
            
            //            cout << myLeftEye << endl;
            //            cout << myRightEye << endl << endl;
            
            computeEyeCentreDistances(dataEyeCentres[0], dataEyeCentres[1], myLeftEye, myRightEye, eyeCentreDistances);

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
    
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "testDetection time = " << time_time << endl;
}

void testIrisDetection(void (*fptr_detectAndShow)(void*, Mat frame), void* context, Mat &frame, Mat &originalFrame, const vector<Vec3f> &irises)
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
    
    int files[] = { 10, 16, 17, 27, 43, 50, 52, 58, 62, 64, 65, 66, 71, 73, 75, 77, 78, 82, 87, 101, 102, 104, 107, 110, 111, 128, 167, 175, 187, 188, 197, 212, 213, 218, 219, 220, 249, 265, 266, 267, 268, 280, 281, 284, 285, 286, 300, 305, 373, 420, 433, 444, 446, 482, 488, 513, 560, 574, 576, 577, 580, 581, 582, 585, 598, 604, 617, 630, 649, 662, 670, 672, 674, 675, 677, 685, 823, 827, 832, 854, 875, 880, 928, 943, 1080, 1093, 1095, 1096, 1153, 1157, 1160, 1175, 1177, 1209, 1227, 1237, 1245, 1264, 1289, 1290};
    int fileCount = 100;
    
    char numstr[21]; // enough to hold all numbers up to 64-bits
    string imgFullFilePath = "", dataFullFilePath = "", myDataFullFilePath = "";
    for (int i = 0; i < fileCount; i++)
    {
        vector<Point> dataEyeCentres;
        
        switch (digitsCount(files[i]))
        {
            case 1:
                sprintf(numstr, "000%d", static_cast<int>(files[i]));
                break;
                
            case 2:
                sprintf(numstr, "00%d", static_cast<int>(files[i]));
                break;
            case 3:
                sprintf(numstr, "0%d", static_cast<int>(files[i]));
                break;
            default:
                sprintf(numstr, "%d", static_cast<int>(files[i]));
                break;
        }
        
        
        imgFullFilePath = folder + prefix + numstr + imgSuffix;
        
        
        frame = imread(imgFullFilePath);
        
        originalFrame = frame.clone();
        fptr_detectAndShow(0, frame);
        
        imgFullFilePath = folder + "results/" + prefix + numstr +"_result" + imgSuffix;
        //imwrite(imgFullFilePath, frame);
        
        // test data
        dataFullFilePath = folder + prefix + numstr + dataSuffix;
        // read cords from file
        readEyeData(dataFullFilePath, dataEyeCentres);
        
        myDataFullFilePath = folder + prefix + numstr + myDataSuffix;
        Vec6f myEyeData = readMyEyeData(myDataFullFilePath);
        
        
        if (irises.size() == 2)
        {
            computeIrisesDistances(dataEyeCentres[0], dataEyeCentres[1], irises[0][2], irises[1][2], myEyeData[0], myEyeData[1], irisesDistances);
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
}

void testLidsDetectionvoid (void (*fptr_detectAndShow)(void*, Mat frame), void* context, Mat &frame, Mat &originalFrame, const vector<Vec4f> &eyeLids)
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
    
    int fileCount = 100;
    int files[] = { 10, 16, 17, 27, 43, 50, 52, 58, 62, 64, 65, 66, 71, 73, 75, 77, 78, 82, 87, 101, 102, 104, 107, 110, 111, 128, 167, 175, 187, 188, 197, 212, 213, 218, 219, 220, 249, 265, 266, 267, 268, 280, 281, 284, 285, 286, 300, 305, 373, 420, 433, 444, 446, 482, 488, 513, 560, 574, 576, 577, 580, 581, 582, 585, 598, 604, 617, 630, 649, 662, 670, 672, 674, 675, 677, 685, 823, 827, 832, 854, 875, 880, 928, 943, 1080, 1093, 1095, 1096, 1153, 1157, 1160, 1175, 1177, 1209, 1227, 1237, 1245, 1264, 1289, 1290 };
    
    char numstr[21]; // enough to hold all numbers up to 64-bits
    string imgFullFilePath = "", dataFullFilePath = "", myDataFullFilePath = "";
    for (int i = 0; i < fileCount; i++)
    {
        vector<Point> dataEyeCentres;
        
        switch (digitsCount(files[i]))
        {
            case 1:
                sprintf(numstr, "000%d", static_cast<int>(files[i]));
                break;
                
            case 2:
                sprintf(numstr, "00%d", static_cast<int>(files[i]));
                break;
            case 3:
                sprintf(numstr, "0%d", static_cast<int>(files[i]));
                break;
            default:
                sprintf(numstr, "%d", static_cast<int>(files[i]));
                break;
        }
        
        
        imgFullFilePath = folder + prefix + numstr + imgSuffix;
        
        
        frame = imread(imgFullFilePath);
        
        originalFrame = frame.clone();
        fptr_detectAndShow(0, frame);
        
        imgFullFilePath = folder + "results/" + prefix + numstr +"_result" + imgSuffix;
        //imwrite(imgFullFilePath, frame);
        
        // test data
        dataFullFilePath = folder + prefix + numstr + dataSuffix;
        // read cords from file
        readEyeData(dataFullFilePath, dataEyeCentres);
        
        myDataFullFilePath = folder + prefix + numstr + myDataSuffix;
        Vec6f myEyeData = readMyEyeData(myDataFullFilePath);
        
        
        if (eyeLids.size() == 4)
        {
            computeLidsDistances(dataEyeCentres[0], dataEyeCentres[1], eyeLids, myEyeData, lidsDistances);
        }
        
    }
    
    double smallestE = 0.05;
    for (int i = 1; i <= 5; i++)
    {
        cout << "eye lids normalised error for " << smallestE * i << " = " <<  getNormalisedError(lidsDistances, smallestE * i) * 100 << "%" << endl;
    }
    
    // vypis pro tabulku
    smallestE = 0.01;
    for (int i = 1; i <= 25; i++)
    {
        cout <<  getNormalisedError(lidsDistances, smallestE * i) * 100 << endl;
    }
    
    
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "testLidsDetection time = " << time_time << endl;
}




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
