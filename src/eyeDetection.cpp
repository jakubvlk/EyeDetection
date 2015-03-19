//
//  eyeDetection.cpp
//  EyeDetection
//
//  Created by Jakub Vlk on 19/03/15.
//
//

#include "eyeDetection.h"
#import "constants.h"

#if XCODE
string eyeCascadeName = "../../../res/haarcascade_eye_tree_eyeglasses.xml";
#else
string eyeCascadeName = "../res/haarcascade_eye_tree_eyeglasses.xml";
#endif

CascadeClassifier eyeCascade;


// private functions
vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face);


Mat eyeDetection(Mat &face)
{
    std::vector<Rect> eyes;
    
    // Detect eyes
    eyeCascade.detectMultiScale( face, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    eyes = pickEyeRegions(eyes, face);
    
    for( size_t j = 0; j < eyes.size(); j++ )
    {
        //rectangle( frame, Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar( 0, 0, 255 ), 2);
        
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
    }
}

void loadEyeCascade()
{
    if( !eyeCascade.load( eyeCascadeName ) )
    {
        cerr << "Can't load eye cascade " + eyeCascadeName << endl;
    }
}


vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face)
{
    // moznost: vertikalne rozdelit tvar do 4 casti. Oci by mely byt v 2. casti
    
    vector<Rect> correctEyes = eyes;
    
    // prostor pro oci je urcite ve vrchni polovine obliceje...  !!! toto by se dalo udelat i zmensenim oblasti obliceje o 1/2 -> lepsi vykon!!!
    for (int i = 0; i < correctEyes.size(); ++i)
    {
        if (correctEyes[i].y > (face.size().height * 0.5 ))
        {
            cout << "Mazu! Oblast oka mimo vrchni polovinu obliceje. x,y = " << correctEyes[i].x << ", " << correctEyes[i].y << ". Polovina obliceje ma delku " << face.size().height * 0.5 << endl;
            correctEyes.erase(correctEyes.begin() + i);
        }
    }
    
    // odebere ocni oblasti, ktere zasahuji mimo oblicej
    for (int i = 0; i < correctEyes.size(); ++i)
    {
        // Prave oko
        if ( eyes[i].x > (face.size().width * 0.5) )
        {
            if ( (eyes[i].x + eyes[i].width)  > face.size().width )
            {
                cout << "Mazu! Oblast praveho oka je mimo oblicej. x,y = " << eyes[i].x << ", " << eyes[i].y << endl;
                correctEyes.erase(correctEyes.begin() + i);
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
        }
    }
    
    // odstrani oci s podobnym stredem
    for (int i = 0; i < correctEyes.size(); ++i)
    {
        // jak jsou vzdalene stredy 2. ocnich oblasti. Pokud je to min nez treshold (relativne), tak mensi ocni oblast odstranime
        double distancesTresh = 0.1;
        
        for (int j = 0; j < correctEyes.size(); ++j)
        {
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
                            
                            if ( i >= correctEyes.size())
                                i = correctEyes.size() - 1;
                            if ( j >= correctEyes.size())
                                j = correctEyes.size() - 1;
                        }
                        else
                        {
                            correctEyes.erase(correctEyes.begin() + i);
                            //cout << "mazu i " << i << endl;
                            
                            if ( j >= correctEyes.size())
                                j = correctEyes.size() - 1;
                            if ( i >= correctEyes.size())
                                i = correctEyes.size() - 1;
                            
                        }
                    }
                }
            }
        }
    }
    
    // TMP - tvrde smazeni na 2 ocni oblasti
    if (correctEyes.size() > 2)
        correctEyes.erase(correctEyes.begin() + 2, correctEyes.begin() + correctEyes.size());  
    
    if (correctEyes.size() > 0)
    {
        return correctEyes;
    }
    else
    {
        return eyes;
    }
}