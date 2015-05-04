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

#if XCODE
string folder = "../../../res/pics/BioID-FaceDatabase-V1.2/";
#else
string folder = "../res/pics/BioID-FaceDatabase-V1.2/";
#endif
string prefix = "BioID_";
/*
void testDetection(Mat frame, Mat originalFrame)
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
        detectAndDisplay(frame);
        
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
    
    cout << "method time = " << methodTime << endl;
    cout << "avg method time = " << methodTime / eyesDetectedCount << endl;
    
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
}*/