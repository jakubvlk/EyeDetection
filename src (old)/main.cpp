/* TODO:
            1) dopsat processArguments a k nemu help
            2) napsat aspon velmi lehkou dokumentaci jak buildit a co dela ktery parametr - textak
            3) zkusit nejak udelat, ze pri chybe (napr spatne nacteni kaskad) to propadne do main a skonci program... return X...
 
*/


#import <stdio.h>
#import <iostream>

#import "captureImage.h"
#import "detection.h"
#import "constants.h"

using namespace std;



#if XCODE
string file = "../../../res/pics/lena.png";
#else
string file = "../res/pics/lena.png";
#endif

bool useVideo = false, useCamera = false, stepFrame = false, showWindow = false;
bool drawInFrame = true;


// private functions
void processArguments( int argc, const char** argv );
void showUsage( string name );



int main( int argc, const char** argv )
{
    processArguments( argc, argv);
    
    initDetection();
    
    CvCapture *capture = startCapture(file, useVideo, useCamera);
    
	return 0;
}

void processArguments( int argc, const char** argv )
{
    cout << argc << endl;
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        if ((arg == "-h") || (arg == "--help"))
        {
            showUsage(argv[0]);
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
                cerr << "--file option requires one argument." << endl;
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
}

void showUsage( string name )
{
    cerr << "Usage: " << name << " <option(s)> SOURCES"
    << "Options:\n"
    << "\t-h,--help\t\tShow this help message\n"
    << "\t-d,--destination DESTINATION\tSpecify the destination path"
    << std::endl;
}

