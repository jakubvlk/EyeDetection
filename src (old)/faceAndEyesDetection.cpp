// Hough transform - na rozpoznavani usecek a dalsich vecich, jako treba kruznic (kruhovych oblouku)
/*
TODO: 1) Podivat se jak detekuji oci a a pokud nedetekuji kazde vzlast, tak udelat trenovani.
			- pozitivni / negativni vzorky
	  2) Rozjet detekci vicek Houghovou trans. nebo prijit s jinym konceptem
	  3) Rohovka a duhovka - asi zase Houghova trans., takze by se to dalo vzit s vickem

	  // mrl.cs.vsb.cz/eyes/dataset/video
*/

#define TIME_MEASURING  0

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/opencv.hpp"


#include <iostream>
#include <stdio.h>

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


Mat resizeMat(Mat src, int width);
Point setEyesCentres ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
Point findEyeCentre ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
Point myHoughCircle(Mat eye, int kernel, string windowName, int x, int y, int frameX, int frameY, Point center);
Mat removeReflections(Mat eye, string windowName, int x, int y, int frameX, int frameY);
void findEyeCorners ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
void VPF_eyelids ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);

void FCD(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
Point accurateEyeCentreLocalisationByMeansOfGradients(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
void findPupil(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY, Point center);
void findEyeLids(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
void blob(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);



//trackbars
void onHCParam1Trackbar(int pos, void *);
void onHCParam2Trackbar(int pos, void *);
void onHCDpTrackbar(int pos, void *);
void onHCMinDistanceTrackbar(int pos, void *);



// default values
String face_cascade_name = "../../../res/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "../../../res/haarcascade_eye_tree_eyeglasses.xml";
//String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
//String eyes_cascade_name = "../res/haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

string file = "lena.png";
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

vector<Point> eyesCentres;
vector<Vec3f> irises;
vector<Vec3f> pupils;

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

	// convert from color to grayscale
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	
	// contrast adjustment using the image's histogram
  	equalizeHist( frame_gray, frame_gray );
	
  	//Detects objects (faces) of different sizes in the input image. The detected objects are returned as a list of rectangles.
  	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(frame_gray.size().width / 6, frame_gray.size().height / 6) );

  	if (faces.size() > 0)
  	{
  		Rect face = pickFace(faces);

    	Point center( face.x + face.width*0.5, face.y + face.height*0.5 );
    	rectangle( frame, Rect(face.x, face.y, face.width, face.height), Scalar( 255, 0, 255 ), 4, 8, 0 );

    	Mat faceROI = frame_gray( face );
    	std::vector<Rect> eyes;

    	//-- In each face, detect eyes
    	eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );


    	eyes = pickEyeRegions(eyes, faceROI);

    	for( size_t j = 0; j < eyes.size(); j++ )
     	{
            rectangle( frame, Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar( 0, 0, 255 ), 2);
            
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

//			Point eyeCenter = setEyesCentres(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
            
            //Point eyeCenter = findEyeCentre(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
            Point eyeCenter = accurateEyeCentreLocalisationByMeansOfGradients(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);

			//eyeCenter = myHoughCircle(eyeWithoutReflection, 11, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeCenter);
            myHoughCircle(eyeWithoutReflection, 3, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeCenter);
            findPupil(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeCenter);
//            findEyeCorners(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
            //VPF_eyelids(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
            

            //findEyeLids(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
			//FCD(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
            //blob(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
     	}

		if (drawInFrame)
		{
	     	drawEyesCentres();
            drawPupils();
            //drawIrises();
 
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

Mat resizeMat(Mat src, int width)
{
    Mat resizedMat;
    
    double widthRatio = width / (double)src.size().width;
    
    resize(src, resizedMat, Size(width, lround(widthRatio * src.size().height)));
    
    return resizedMat;
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

float avgIntensity(Mat mat, int x, int y, int width, int height, int maxIntensity)
{
    int totalIntensity = 0;
    int pixelCount = 0;
    
    for (int i = y; i < y + height; i++)
    {
        for (int j = x; j < x + width; j++)
        {
            int intensity = mat.at<uchar>(i, j);
            
            if (intensity < maxIntensity)
            {
                totalIntensity += intensity;
                pixelCount++;
            }
            else
            {
                cout << "damn" << endl;
            }

        }
    }
    
//    Mat newMat = mat(Rect(x, y, width, height));
//    showWindowAtPosition( "roi", newMat, 600, 0);
//    
//    
//    rectangle( mat, Rect(x, y, width, height), CV_RGB(255, 255, 0));
//    showWindowAtPosition( "mat", mat, 600, 100);
    
    return totalIntensity / (float)pixelCount;
}

int blackPixelsCount(Mat mat)
{
    int blackPixelCount = 0;
    
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            if(mat.at<uchar>(i, j) == 0)
            {
                blackPixelCount++;
            }
        }
    }
    
    return blackPixelCount;
}

// IND
Point findEyeCentre ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
#if TIME_MEASURING
    double time_time;
    int64 time_wholeFunc = getTickCount();
#endif
    
    bool rightEye = (frameX + eye.cols * 0.5) > (frame.cols * 0.5);
    
    Mat blurredEye;
    medianBlur(eye, blurredEye, 5); //5
    
    //showWindowAtPosition( windowName + " eye", eye, windowX, windowY);
    
    // 3.1. INTENSITY SCALING 4~8
    Mat intensiveEye(blurredEye.rows, blurredEye.cols, CV_8U);
    int intesMul = 5;
    
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
    
    /*
    // histogram
    // Initialize parameters
    int histSize = 256;    // bin size
    float range[] = { 0, 256 };
    const float *ranges[] = { range };
    // Calculate histogram
    MatND hist;
    calcHist( &intensiveEye, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );
    // Show the calculated histogram in command window
    double total;
    total = intensiveEye.rows * intensiveEye.cols;
    for( int h = 0; h < histSize; h++ )
    {
        float binVal = hist.at<float>(h);
        cout<<" "<<binVal << endl;
    }
    // Plot the histogram
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage( hist_h, hist_w, CV_8U, Scalar( 0,0,0) );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
    }
    showWindowAtPosition( windowName + " hist", histImage, windowX, windowY + 130);
    */
     
    
    // otsu tresh
    Mat uselessMat;
    double otsu_thresh_val = threshold( intensiveEye, uselessMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
    string e = "";
    if (rightEye)
        e = "r";
    else
        e = "l";
    
    //showWindowAtPosition( windowName + " otsu " + e, uselessMat, windowX, windowY + 260);
    
    double high_thresh_val  = otsu_thresh_val;
    double lower_thresh_val = otsu_thresh_val * 0.5;
    //cout << "Computed tresholds = " << high_thresh_val << ", " << lower_thresh_val << endl;
    
    Mat eyeCanny;
    Canny(uselessMat, eyeCanny, lower_thresh_val, high_thresh_val);
    
    //showWindowAtPosition( windowName + " canny", eyeCanny, windowX, windowY + 390);
    
    // najit kontury
    vector<vector<Point> > contours;
    Mat threshold_output = eyeCanny.clone();
    vector<Vec4i> hierarchy;
    
    findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );
    //cout << "pocet kontur = " << contours.size() << endl;
    
    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    
    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       	boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       	minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }
    
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
        //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    }
    
//    showWindowAtPosition( windowName + " draw", drawing, windowX, windowY + 680);
    
    // 3.2. IRIS CONTOUR SELECTION
    // Pokud boundRect.height >= 3/2 * boundRect.width tak tuto konturu nepozujeme za korektni
    
    double threeHalfs = 3/2.;
    for( int i = 0; i < boundRect.size(); )
    {
        if (boundRect[i].height >= threeHalfs * boundRect[i].width)
        {
            boundRect.erase(boundRect.begin() + i);
            center.erase(center.begin() + i);
            radius.erase(radius.begin() + i);
            contours.erase(contours.begin() + i);
            contours_poly.erase(contours_poly.begin() + i);
            //cout << "Rejected" << endl;
        }
        else
        {
            i++;
        }
    }
    
    // Vybereme 2 nejvetsi bounding boxy (muze byt oko a oboci)
    
    int indexCandidate1 = -1, indexCandidate2 = -1, finalCandidate = 0;
    bool empty = boundRect.size() == 0;
    
    int minWidth = 3;
    int minHeight = 3;
    int minAre = minWidth * minHeight;
    
    if (!empty)
    {
        if (boundRect.size() > 1)
        {
            int max = -1;
            
            for( int i = 0; i < boundRect.size(); i++)
            {
                if (boundRect[i].size().area() > max)
                {
                    indexCandidate1 = i;
                    max = boundRect[i].size().area();
                }
            }
            
            max = -1;
            for( int i = 0; i < boundRect.size(); i++)
            {
                // Pokud to neni prvni kontura, neprekryva se s prvni konturou, je vetsi nez minArea a je (zatim) nejvetsi
                if (indexCandidate1 != i && (boundRect[indexCandidate1] & boundRect[i]).area() == 0 && boundRect[i].area() > minAre && boundRect[i].size().area() > max)
                {
                    if (boundRect[i].width > minWidth && boundRect[i].height > minHeight && boundRect[i].area() > minAre)
                    {
                        indexCandidate2 = i;
                        max = boundRect[i].size().area();
                    }
                }
            }
            
            if (indexCandidate2 != -1)
            {
                // Porovname Y souradnici stredu. Pokud maji podobnou, tak urcime finalniho kandidata pomoci intensity. Pokud nemaji podobnou Y souradnici stredu, tak finalni kandidat bude ten s vyssi Y souradnici stredu.
                int yDistanceThresh = 5;
                if (abs ( center[indexCandidate1].y - center[indexCandidate2].y ) <= yDistanceThresh)
                {
                    // NAPAD: Mozna radsi podle obsahu BB kontury???
                    
                    int maxIntensity = 250;
                    float intens1 = avgIntensity(eye, boundRect[indexCandidate1].tl().x, boundRect[indexCandidate1].tl().y, boundRect[indexCandidate1].width, boundRect[indexCandidate1].height, maxIntensity);
                    float intens2 = avgIntensity(eye, boundRect[indexCandidate2].tl().x, boundRect[indexCandidate2].tl().y, boundRect[indexCandidate2].width, boundRect[indexCandidate2].height, maxIntensity);
                    
                    if (intens1 < intens2)
                    {
                        finalCandidate = indexCandidate1;
                    }
                    else
                    {
                        finalCandidate = indexCandidate2;
                    }
                }
                else
                {
                    if (center[indexCandidate1].y > center[indexCandidate2].y)
                    {
                        finalCandidate = indexCandidate1;
                    }
                    else
                    {
                        finalCandidate = indexCandidate2;
                    }
                }
            }
            else
            {
                finalCandidate = indexCandidate1;
            }
        }
        
        // 3.3. ELLIPSE FITTING
        RotatedRect myEllipse = fitEllipse( Mat(contours[finalCandidate]));
        
        /*ellipse( drawing, myEllipse, CV_RGB(255, 255, 255));
        int lineLength2 = 6;
        line(drawing, Point(myEllipse.center.x - lineLength2*0.5, myEllipse.center.y), Point(myEllipse.center.x + lineLength2*0.5, myEllipse.center.y), CV_RGB(255, 255, 255));
        line(drawing, Point(myEllipse.center.x, myEllipse.center.y - lineLength2*0.5), Point(myEllipse.center.x, myEllipse.center.y + lineLength2*0.5), CV_RGB(255, 255, 255));
        showWindowAtPosition( windowName + " draw", drawing, windowX, windowY + 680);
        
        Mat tmpMat = uselessMat.clone();
        cvtColor(tmpMat, tmpMat, CV_GRAY2BGR);
        ellipse( tmpMat, myEllipse, CV_RGB(255, 0, 0));
        line(tmpMat, Point(myEllipse.center.x - lineLength2*0.5, myEllipse.center.y), Point(myEllipse.center.x + lineLength2*0.5, myEllipse.center.y), CV_RGB(255, 0, 0));
        line(tmpMat, Point(myEllipse.center.x, myEllipse.center.y - lineLength2*0.5), Point(myEllipse.center.x, myEllipse.center.y + lineLength2*0.5), CV_RGB(255, 0, 0));
        showWindowAtPosition( windowName + " otsu " + e, tmpMat, windowX, windowY + 260);
        
        Mat correctEye = eye.clone();
        cvtColor(eye, eye, CV_GRAY2BGR);
        line(eye, Point(myEllipse.center.x - lineLength2*0.5, myEllipse.center.y), Point(myEllipse.center.x + lineLength2*0.5, myEllipse.center.y), CV_RGB(255, 0, 0));
        line(eye, Point(myEllipse.center.x, myEllipse.center.y - lineLength2*0.5), Point(myEllipse.center.x, myEllipse.center.y + lineLength2*0.5), CV_RGB(255, 0, 0));
        showWindowAtPosition( windowName + " eye", eye, windowX, windowY);
         */
        
        
        
        
//        // TMP DRAW
//        Mat eye2 = eye.clone();
//        Scalar ellipseColor2 = CV_RGB(255, 255, 0);
//        cvtColor(eye2, eye2, CV_GRAY2BGR);
//        ellipse( eye2, myEllipse, ellipseColor2);
//        int lineLength2 = 10;
//        line(eye2, Point(myEllipse.center.x - lineLength2*0.5, myEllipse.center.y), Point(myEllipse.center.x + lineLength2*0.5, myEllipse.center.y), ellipseColor2);
//        line(eye2, Point(myEllipse.center.x, myEllipse.center.y - lineLength2*0.5), Point(myEllipse.center.x, myEllipse.center.y + lineLength2*0.5), ellipseColor2);
//        
//        // cebter if ellipse to frame
//        line(frame, Point(frameX + myEllipse.center.x - lineLength2*0.5, frameY + myEllipse.center.y), Point(frameX + myEllipse.center.x + lineLength2*0.5, frameY + myEllipse.center.y), ellipseColor2);
//        line(frame, Point(frameX + myEllipse.center.x, frameY + myEllipse.center.y - lineLength2*0.5), Point(frameX + myEllipse.center.x, frameY + myEllipse.center.y + lineLength2*0.5), ellipseColor2);

        
        
        // 3.4. CORRECTION OF DETECTED CENTER
        int boundRectHeight = boundRect[finalCandidate].height;
        
        // upper left corner of R2
        int x1 = lround( myEllipse.center.x - 0.25 * boundRectHeight);
        int y1 = boundRect[finalCandidate].tl().y;
        Rect R2 = Rect(x1, y1, boundRectHeight, boundRectHeight);
        if (R2.x < 0)
            R2.x = 0;
        if (R2.x + R2.width >= uselessMat.cols)
            R2.x = uselessMat.cols - R2.width;
        
        // upper right corner of R1
        int x2 = myEllipse.center.x;
        int y2 = boundRect[finalCandidate].tl().y;
        Rect R1 = Rect(x2 - boundRectHeight, y2, boundRectHeight, boundRectHeight);
        if (R1.x < 0)
            R1.x = 0;
        if (R1.x  + R1.width >= uselessMat.cols)
            R1.x = uselessMat.cols - R1.width;
        
        Mat r1Mat = uselessMat(R1);
        Mat r2Mat = uselessMat(R2);
        if (rightEye)
        {
            // Je nutna oprava stredu
            if (blackPixelsCount(r1Mat) > blackPixelsCount(r2Mat))
            {
                cout << "Oprava R" << endl;
                
                vector<Point> correctedContour;
                for (int i = 0; i < contours[finalCandidate].size(); i++)
                {
                    if ( ( boundRect[finalCandidate].tl().x + boundRectHeight ) >  contours[finalCandidate][i].x)
                    {
                        correctedContour.push_back(contours[finalCandidate][i]);
                    }
                }
                
                myEllipse = fitEllipse( Mat(correctedContour));
            }
        }
        else
        {
            // Je nutna oprava stredu
            if (blackPixelsCount(r1Mat) > blackPixelsCount(r2Mat))
            {
                cout << "Oprava L" << endl;
                
                vector<Point> correctedContour;
                for (int i = 0; i < contours[finalCandidate].size(); i++)
                {
                    if ( ( boundRect[finalCandidate].tl().x + boundRectHeight ) >  contours[finalCandidate][i].x)
                    {
                        correctedContour.push_back(contours[finalCandidate][i]);
                    }
                }
                
                myEllipse = fitEllipse( Mat(correctedContour));
            }
        }
        
        /*cvtColor(correctEye, correctEye, CV_GRAY2BGR);
        line(correctEye, Point(myEllipse.center.x - lineLength2*0.5, myEllipse.center.y), Point(myEllipse.center.x + lineLength2*0.5, myEllipse.center.y), CV_RGB(255, 0, 0));
        line(correctEye, Point(myEllipse.center.x, myEllipse.center.y - lineLength2*0.5), Point(myEllipse.center.x, myEllipse.center.y + lineLength2*0.5), CV_RGB(255, 0, 0));
        showWindowAtPosition( windowName + " correct eye", correctEye, windowX, windowY + 130);
        
        tmpMat = uselessMat.clone();
        cvtColor(tmpMat, tmpMat, CV_GRAY2BGR);
        ellipse( tmpMat, myEllipse, CV_RGB(255, 0, 0));
        line(tmpMat, Point(myEllipse.center.x - lineLength2*0.5, myEllipse.center.y), Point(myEllipse.center.x + lineLength2*0.5, myEllipse.center.y), CV_RGB(255, 0, 0));
        line(tmpMat, Point(myEllipse.center.x, myEllipse.center.y - lineLength2*0.5), Point(myEllipse.center.x, myEllipse.center.y + lineLength2*0.5), CV_RGB(255, 0, 0));
        showWindowAtPosition( windowName + " correct otsu" + e, tmpMat, windowX, windowY + 360);*/


        
        //myEllipse = fitEllipse( Mat(contours[finalCandidate]));
        
        // TMP DRAW
        /*//drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        cvtColor(eye, eye, CV_GRAY2BGR);
        Scalar color = CV_RGB(0, 0, 255);
        rectangle( eye, boundRect[indexCandidate1].tl(), boundRect[indexCandidate1].br(), color);

        if (indexCandidate2 != -1)
            rectangle( eye, boundRect[indexCandidate2].tl(), boundRect[indexCandidate2].br(), color);
        
        Mat finalDrawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
        //Scalar color2 = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        //drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        rectangle( eye, boundRect[finalCandidate].tl(), boundRect[finalCandidate].br(), CV_RGB(255, 0, 0));
        
         */
        // ellipse
        Scalar ellipseColor = CV_RGB(0, 255, 0);
        //ellipse( eye, myEllipse, ellipseColor);
        // center of ellipse
        // circle outline
        
        int lineLength = 6;
        line(eye, Point(myEllipse.center.x - lineLength*0.5, myEllipse.center.y), Point(myEllipse.center.x + lineLength*0.5, myEllipse.center.y), ellipseColor);
        line(eye, Point(myEllipse.center.x, myEllipse.center.y - lineLength*0.5), Point(myEllipse.center.x, myEllipse.center.y + lineLength*0.5), ellipseColor);
        
        // cebter if ellipse to frame
        line(frame, Point(frameX + myEllipse.center.x - lineLength*0.5, frameY + myEllipse.center.y), Point(frameX + myEllipse.center.x + lineLength*0.5, frameY + myEllipse.center.y), ellipseColor);
        line(frame, Point(frameX + myEllipse.center.x, frameY + myEllipse.center.y - lineLength*0.5), Point(frameX + myEllipse.center.x, frameY + myEllipse.center.y + lineLength*0.5), ellipseColor);
    
        //showWindowAtPosition( windowName + " final", eye, windowX, windowY + 580);
        //showWindowAtPosition( windowName + " PRE final", eye2, windowX, windowY + 680);
         
        
        //Point frameCenter(myEllipse.center.x + frameX, myEllipse.center.y + frameY);
        //eyesCentres.push_back(frameCenter);
        
#if TIME_MEASURING
        time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
        cout << "find eye cenre time = " << time_time << endl;
#endif
        
        return myEllipse.center;
    }
    
#if TIME_MEASURING
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "find eye cenre time = " << time_time << endl;
#endif
    
    
    return Point(eye.rows * 0.5, eye.cols * 0.5);
    
    // TODO: jeste porovnat s mojim predchozim resenim; pak nejak at se to da lehce prepiant. Pak zkusit zornicky
}

Point setEyesCentres ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    //showWindowAtPosition( windowName + " eye centres", eye, windowX, windowY);
    
	Mat tmp, medianBlurMat;

	medianBlur(eye, medianBlurMat, 7);

	//pokusne orezani oboci
	double eyeTrimHeight = medianBlurMat.size().height * 0.2;
	tmp = medianBlurMat(Rect(0, eyeTrimHeight, medianBlurMat.size().width, medianBlurMat.size().height - (eyeTrimHeight)));
	
    int erosion_size = 1;  // 2
    Mat ErosElement = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size) );
    erode( tmp, tmp, ErosElement );
    
    threshold( tmp, tmp, 4, 255, CV_THRESH_BINARY_INV);  // 8
    //showWindowAtPosition( windowName + " tresh center", tmp, windowX, windowY + 130);

	vector<vector<Point> > contours;
    Mat threshold_output = tmp.clone();
	vector<Vec4i> hierarchy;

    findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly( contours.size() );
	//vector<Rect> boundRect( contours.size() );
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );

	for( int i = 0; i < contours.size(); i++ )
    { 
    	approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       	//boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       	minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }

    Point correctCenter = Point(tmp.size().width * 0.5, tmp.size().height * 0.5);
    // Pokud najdeme vice kontur, tak nechame jen tu nejvetsi
    if (contours.size() > 0)
    {
	    correctCenter = center[0];
	    if (contours.size() > 1)
	    {
	    	int maxRadius = 0;
	    	int maxRadiusIndex = 0;

	    	for (int i = 0; i < contours.size(); ++i)
	    	{
	    		if (radius[i] > maxRadius)
		    	{
		    		maxRadius = radius[i];
		    		maxRadiusIndex = i;
		    	}	
	    	}

	    	correctCenter = center[maxRadiusIndex];
	    }



//		Point frameCenter(correctCenter.x + frameX, correctCenter.y + frameY + eyeTrimHeight);
//		eyesCentres.push_back(frameCenter);
	}

    return Point(correctCenter.x, correctCenter.y + eyeTrimHeight);
}

Mat removeReflections(Mat eye, string windowName, int x, int y, int frameX, int frameY)
{
//    showWindowAtPosition( windowName + " eye refl", eye, x, y);
    
	Mat gaussEye, binaryEye;

	GaussianBlur( eye, gaussEye, Size(3,3), 0, 0, BORDER_DEFAULT );

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, grad;
	
	int scale = 1;
  	int delta = 0;
  	int ddepth = CV_16S;

	/// Gradient X
	Sobel( gaussEye, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( gaussEye, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );

	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	threshold(grad, binaryEye, 91, 255, CV_THRESH_BINARY);    //
    //showWindowAtPosition( windowName + " eye bin", binaryEye, x, y);
	//	Mozna jeste pridat erozi???


	Mat reparedEye;
	inpaint(eye, binaryEye, reparedEye, 3, INPAINT_TELEA);

	// Draw
	//showWindowAtPosition( windowName + " - grad", grad, x - 200, y + 260);
	//showWindowAtPosition( windowName + " - bez odlesku", reparedEye, x - 200, y + 390);
    //showWindowAtPosition( windowName + " - grad x", mat2gray(grad_x), x - 200, y + 390);
    //showWindowAtPosition( windowName + " - grad y", mat2gray(grad_y), x - 200, y + 490);
	
	return reparedEye;
}



Point unscalePoint(Point p, int origWidth, int width)
{
    float ratio = ((float)width / origWidth);
    int x = roundl(p.x / ratio);
    int y = roundl(p.y / ratio);

    return Point(x, y);
}

Point uncut(Point p, int cut)
{
    return Point(p.x + cut * 0.25f, p.y + cut * 0.6f);
}

Point accurateEyeCentreLocalisationByMeansOfGradients(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    //showWindowAtPosition( windowName + "eye", eye, windowX, windowY);
    
#if TIME_MEASURING
    double time_time;
    int64 time_wholeFunc = getTickCount();
#endif
    
    Mat originalEye = eye.clone();
    
    int fastWidth = 50;
    bool scaleMat = false;
    if (scaleMat)
    {
        eye = resizeMat(eye, fastWidth);
    }
    else
    {
        fastWidth = 20;
        eye = originalEye( Rect(fastWidth * 0.25, fastWidth * 0.6, originalEye.size().width - fastWidth * 0.25, originalEye.size().height - fastWidth * 0.6) );

    }
    
    
    Mat gaussEye;
    
    GaussianBlur( eye, gaussEye, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    Mat intensiveEye = gaussEye.clone();
    int intesMul = 5;
    for (int y = 0; y < intensiveEye.rows; y++)
    {
        uchar *intens = intensiveEye.ptr<uchar>(y);
        for (int x = 0; x < intensiveEye.cols; x++)
        {
            int intensity = intens[x] * intesMul;
            if (intensity > 255)
                intensity = 255;

            intens[x] = intensity;
        }
    }
    
    Mat grad_x, grad_y;
    
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;
    
    /// Gradient X
    Sobel( intensiveEye, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    /// Gradient Y
    Sobel( intensiveEye, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    
    float dotProducts[eye.rows][eye.cols];
    for (int y = 0; y < eye.rows; y++)
    {
        for (int x = 0; x < eye.cols; x++)
        {
            dotProducts[y][x] = 0;
        }
    }
    
    int gradientsCount = 0;
    Mat sum = Mat::zeros(eye.rows, eye.cols, CV_32F);
    for (int y = 0; y < eye.rows; ++y)
    {
        const float *gradXRows = grad_x.ptr<float>(y);
        const float *gradYRows = grad_y.ptr<float>(y);
        
        for (int x = 0; x < eye.cols; ++x)
        {
            float gX = gradXRows[x];
            float gY = gradYRows[x];
            if (gX == 0.f && gY == 0.f)
            {
                continue;
            }
            
            float gMag = sqrt( gX * gX + gY*gY );
            
            float gx = gX / gMag, gy = gY / gMag;
            
            for (int cy = 0; cy < eye.rows; cy++)
            {
                float *sumRows = sum.ptr<float>(cy);
                for (int cx = 0; cx < eye.cols; cx++)
                {
                    if (x == cx && y == cy)
                    {
                        continue;
                    }
                    
                    float dx = x - cx, dy = y - cy;
                
                    
                    //normalize d
                    float dMagnitude = sqrt( dx * dx + dy * dy );
                    dx = dx / dMagnitude;
                    dy = dy / dMagnitude;

                    float dotProduct = dx*gx + dy*gy;
                    
                    if (dotProduct < 0.f)
                        dotProduct = 0.f;
                    
                    //dotProducts[cy][cx] += dotProduct*dotProduct;
                    sumRows[cx] += dotProduct * dotProduct;
                    gradientsCount++;
                }
            }
            
            
        }
    }
    
    Point centre;
    double max;
    minMaxLoc(sum, NULL, &max, NULL, &centre);
    
    
#if TIME_MEASURING
    time_time = (getTickCount() - time_wholeFunc)/ getTickFrequency();
    cout << "accurateEyeCentreLocalisationByMeansOfGradients time = " << time_time << endl;
#endif
    
    //cout << centre << endl;
    
    cvtColor(eye, eye, CV_GRAY2BGR);
    //circle(eye, centre, 1, Scalar(0,0, 255));
    Scalar color = CV_RGB(255, 0, 0);
    int lineLength = 6;
    line(eye, Point(centre.x - lineLength*0.5, centre.y), Point(centre.x + lineLength*0.5, centre.y), color);
    line(eye, Point(centre.x, centre.y - lineLength*0.5), Point(centre.x, centre.y + lineLength*0.5), color);
    
    if (scaleMat)
    {
        centre = unscalePoint(centre, originalEye.cols, eye.rows);
    }
    else
    {
        centre = uncut(centre, fastWidth);
    }
    
    Point frameCenter(centre.x + frameX, centre.y + frameY);
    eyesCentres.push_back(frameCenter);
    
    
    
    //showWindowAtPosition( windowName + " final eye", eye, windowX, windowY + 230);
    //showWindowAtPosition( windowName + " final", mat2gray(sum), windowX, windowY + 330);
    
    
    return centre;
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

void findEyeLids(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    showWindowAtPosition( windowName + "PRE eye lids", eye, windowX, windowY );
    
    GaussianBlur( eye, eye, Size(7,7), 0, 0, BORDER_DEFAULT );
    
    // Gradient
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, grad;
    
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    /// Gradient X
    Sobel( eye, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    /// Gradient Y
    Sobel( eye, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    // Je to lepsi???????????????
    //grad = mat2gray(abs_grad_x);
    
//    showWindowAtPosition( windowName + "grad", grad, windowX, windowY + 130 );
    showWindowAtPosition( windowName + "grad Y", abs_grad_y, windowX, windowY + 260);
//    showWindowAtPosition( windowName + "grad x", abs_grad_x, windowX, windowY + 390);
    
    Mat bestEye = abs_grad_y.clone();
    Mat uselessMat;
    double otsu_thresh_val = threshold( bestEye, uselessMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
    double high_thresh_val  = otsu_thresh_val;
    double lower_thresh_val = otsu_thresh_val * 0.5;
    //cout << "Computed tresholds = " << high_thresh_val << ", " << lower_thresh_val << endl;	//140,70
    
    // Mat eyeCanny;
    //Canny(gaussienEye, gaussienEye, lower_thresh_val, high_thresh_val);
    //showWindowAtPosition( windowName + "_canny", gaussienEye, x, y );
    
    // polomery
    int minRadius = cvRound(bestEye.size().width * 0.4);
    int maxRadius = cvRound(bestEye.size().width * 1);
    vector<Vec3f> circles;
    HoughCircles( bestEye, circles, CV_HOUGH_GRADIENT, 1.8, 129, high_thresh_val, 56, minRadius, maxRadius);	//eyeCanny.rows / 8, high_thresh_val
    
    cvtColor(bestEye, bestEye, CV_GRAY2BGR);
    /// Draw the circles detected
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        
        // circle outline
        circle( bestEye, center, radius, Scalar(0, 255, 0), 0, 8, 0 );
        
        circle( bestEye, center, minRadius, Scalar(255, 0, 0));
        circle( bestEye, center, maxRadius, Scalar(255, 0, 0));
        
        
        Point frameCenter(cvRound(circles[i][0]) + frameX, cvRound(circles[i][1]) + frameY);
        // circle outline
        circle( frame, frameCenter, radius, Scalar(0,255,0));
        circle( frame, frameCenter, minRadius, Scalar(255,0,0));
        circle( frame, frameCenter, maxRadius, Scalar(255,0,0));
        
    }
    
    showWindowAtPosition( windowName, bestEye, windowX, windowY + 520);
    
    // TODO: Pohrat si s polomery.
}

void findPupil(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY, Point center)
{
    showWindowAtPosition( windowName + " eye", eye, windowX, windowY + 0);
    
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
    
    showWindowAtPosition( windowName + " intensive eye", intensiveEye, windowX, windowY + 120);
    
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
    showWindowAtPosition( windowName + " pupil", intensiveEye, windowX, windowY + 240);
}

Point myHoughCircle(Mat eye, int kernel, string windowName, int windowX, int windowY, int frameX, int frameY, Point center)
{
#if TIME_MEASURING
     int64 e1 = getTickCount();
#endif
    
    //showWindowAtPosition( windowName + "PRE eye hough", eye, windowX, windowY );
    
    Mat gaussEye;
	GaussianBlur( eye, gaussEye, Size(5,5), 0, 0, BORDER_DEFAULT );
    
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

    
    // Je to lepsi???????????????
    grad = mat2gray(abs_grad_x);
    //showWindowAtPosition( windowName + "grad X", grad, windowX, windowY + 240);


	// polomery
	int minRadius = 8, maxRadius = eye.size().width * 0.3	;

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
	for( size_t i = 0; i < eyesCentres.size(); i++ )
	{
		// circle outline
		Scalar color = Scalar(0, 0, 255);

		int lineLength = 6;
		line(frame, Point(eyesCentres[i].x - lineLength*0.5, eyesCentres[i].y), Point(eyesCentres[i].x + lineLength*0.5, eyesCentres[i].y), color);
		line(frame, Point(eyesCentres[i].x, eyesCentres[i].y - lineLength*0.5), Point(eyesCentres[i].x, eyesCentres[i].y + lineLength*0.5), color);
        
//        circle(frame, eyesCentres[i], 1, color);
	}
}

void showWindowAtPosition( string imageName, Mat mat, int x, int y )
{
	imshow( imageName, mat );
	moveWindow(imageName, x, y);
}