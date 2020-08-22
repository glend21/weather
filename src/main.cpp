/*
 * Use optical flowing tracking to predict rain movements from radar images
 *
 * Usage: rain -a|--algo <algo> [-t|--train <param-out-file>] | [-r|--run <param-in-file>] <src-dir> <dest-dir>
 *
 */

/*
TODOs:
    SORT OF proper cmd-line
    - output of generated images to subdir for visual inspection
        - specify the output dir
    - permutations of o-flow parameters (from input file)
    - output of parameter file
    - find max of the SSIM scalar (isn't it really a vector though?)

    - [p] download of images
        - all available for a radar site for training purposes
        - only the last 2 for the site for production run
    - call this exe from the python driver
*/


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <string.h>
#include <dirent.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdarg>
#include <thread>

#include "ssim.hpp"
#include "opticalflow.hpp"


#define _DEBUG
#define IMG_TYPE ".png"     // This gets used directly to build the input file list

// Types
typedef enum { Blue, Green, Red } CHANNELS;
typedef std::vector< cv::Mat > IMGVEC;

struct CmdLineParams {
    char* algo;
    char* srcDir;
    char* destDir;
    char* paramFile;
    bool doTrain;

    CmdLineParams()
    {
        this->algo = NULL;
        this->srcDir = NULL;
        this->destDir = NULL;
        this->paramFile = NULL;
        this->doTrain = false;
    }
};

struct FarnebackParams {
    float scale;
    int levels,
        smoothingSize,
        iterations,
        polyArea;
    float polyWidth;

    cv::Scalar ssimScore;
};


// Fwd declarations, what a thing!
bool processCmdLine( int argc, char** argv, struct CmdLineParams& params );
bool train( const CmdLineParams& options );
int run( const CmdLineParams& options );
/*
bool processTriple( const std::string& fname1, 
                    const std::string& fname2, 
                    const std::string& testFname,
                    std::vector< FarnebackParams* > params );
*/
bool processWithParam( const cv::Mat& img1, 
                       const cv::Mat& img2, 
                       const cv::Mat& test, 
                       OpticalFlowABC& proc );
void processChannel_t( const cv::Mat& img1,
                       const cv::Mat& img2,
                       cv::Mat& dest,
                       OpticalFlowABC& proc );
void getChannels_t( const std::string& fname, IMGVEC& img );


void buildParams( const std::string& fname, std::vector< FarnebackParams* >& params );
void cleanupParams( std::vector< FarnebackParams* >& params );
std::vector< cv::Mat > getChannels( const cv::String& filename );
void generateFlow( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& flow, FarnebackParams* params );
void remapImage( const cv::Mat& src, cv::Mat& dest, const cv::Mat& flow );
void dbg( std::string fmt, ... );
void usage();


using namespace std;

int main( int argc, char **argv )
{
    int retval = 0;

    /*// - - -
    try
    {
        for ( long n = 0l; ; ++n )
        {
            OpticalFlowABC& fb = OpticalFlowABC::generate( "fb", 50 );
            fb.save( std::string( "training.csv" ) );
            delete &fb;
            dbg( "%d", n );
        }
    }
    catch ( RainException& ex )
    {
        std::cout << "The End." << std::endl;
    }
    return 0;
    // - - -
    */

    CmdLineParams options;
    bool b = processCmdLine( argc, argv, options );
    std::cout << "res: " << b << std::endl;
    std::cout << "algo: " << options.algo << std::endl;
    std::cout << "train? " << options.doTrain << std::endl;
    std::cout << "param file name: " << options.paramFile << std::endl;
    std::cout << "src dir: " << options.srcDir << std::endl;
    //std::cout << "dest dir: " << options.destDir << std::endl;
    if ( ! b )
    {
        std::cout << "Usage: rain -a|--algo algo [-t|--train <param-out-file>] | [-r|--run <param-in-file>] <src-dir> [<dest-dir>]" << std::endl;
        return -1;
    }

    if ( options.doTrain )
    {
        return train( options );
    }
    else
    {
        return run( options );
    }
}


bool processCmdLine( int argc, char** argv, struct CmdLineParams& params )
{
    if (argc > 1)
    {
        for ( ++argv; *argv; ++argv )
        {
            if ( ! strcmp( *argv, "-a" ) || ! strcmp( *argv, "--algo" ) )
            {
                params.algo = *(++argv);
            }
            else if ( ! strcmp( *argv, "-t" ) || ! strcmp( *argv, "--train" ) )
            {
                params.doTrain = true;
                params.paramFile = *(++argv);
            }
            else if ( ! strcmp( *argv, "-r" ) || ! strcmp( *argv, "--run" ) )
            {
                params.paramFile = *(++argv);
            }
            else 
            {
                if ( ! params.srcDir )
                {
                    params.srcDir = *argv;
                }
                else
                {
                    params.destDir = *argv;
                }
            }
        }

        dbg( "A: %s", params.algo );
        dbg( "P: %s", params.paramFile );
        dbg( "S: %s", params.srcDir );
        dbg( "D: %s", params.destDir );
        return ( params.algo && params.paramFile && params.srcDir );
    }

    return false;
}


bool train( const CmdLineParams& options )
{
    dbg( "Training ..." );

    // Get all the src images.
    std::vector< std::string > srcNames;
    IMGVEC srcData;

    DIR* dir = opendir( options.srcDir );
    if ( dir )
    {
        struct dirent *ent;

        while (ent = readdir( dir ))
        {
            const std::string fname( ent->d_name );
            std::size_t pos = fname.rfind( IMG_TYPE );
            if (pos == fname.size() - 4)
            {
                std::stringstream path;
                path << options.srcDir << '/' << fname;
                dbg( "adding %s to src image list", path.str().c_str() );
                srcNames.push_back( path.str() );
                srcData.push_back( cv::imread( path.str(), cv::IMREAD_COLOR ) );
            }
        }

        closedir( dir );
    }
    else
    {
        // could not open directory
        dbg( "could not open %s", options.srcDir );
        return EXIT_FAILURE;
    }

    if (srcNames.size() < 3)
    {
        std::cout << "Need at least 3 images in the source dir";
        return EXIT_FAILURE;
    }

    dbg( "There are %d src images.", srcNames.size() );
    // New algo starts here
    std::ofstream ofh;
    std::string line;

    try
    {
        ofh.open( options.paramFile, std::ios::app );
        if ( ofh.fail() )
        {
            throw RainException( "Could not open parameter file" );
        }

        for ( long n = 0l; ; ++n )
        {
            OpticalFlowABC& proc = OpticalFlowABC::generate( options.algo, 5 );

            // Write header to the parameter file on first iteration
            if (n == 0l)
            {
                proc.paramHeaders( line );
                ofh << line << ','
                    << "image_1" << ','
                    << "image_2" << ','
                    << "image_test" << std::endl;
            }

            // Iterate over all image pairs (bar the last) for this set of algorithm parameters
            for ( int idx = 0; idx < srcData.size() - 2; ++idx )
            {
                if ( srcData[ idx ].data == NULL) dbg( "READ ERROR 1" );
                if ( srcData[ idx + 1 ].data == NULL) dbg( "READ ERROR 2" );
                if ( srcData[ idx + 2 ].data == NULL) dbg( "READ ERROR 3" );

                processWithParam( srcData[ idx ], 
                                  srcData[ idx + 1 ], 
                                  srcData[ idx + 2 ],
                                  proc );

                proc.params( line );
                ofh << line << ','
                    << srcNames[ idx ] << ','
                    << srcNames[ idx + 1 ] << ','
                    << srcNames[ idx + 2 ] << std::endl;
            }

            delete &proc;
            dbg( "%d", n );
        }

        ofh.close();
    }
    catch ( RainException& ex )
    {
        std::cout << "Exception: " << ex.getMsg() << std::endl;
        ofh.close();
    }
    // - - -

    /*
    std::vector< FarnebackParams* > params;
    buildParams( options.paramFile, params );

    for ( int idx = 0; idx < srcList.size() - 2; ++idx )
    {
        retval = processTriple( srcList[ idx ], srcList[ idx + 1 ], srcList[ idx + 2 ], params );

        dbg( "SSIM vector: (%f, %f, %f)", 
             params[0]->ssimScore[ 0 ],
             params[0]->ssimScore[ 1 ],
             params[0]->ssimScore[ 2 ] );
        break;
    }
    */

    return 0;
}


int run( const CmdLineParams& options )
{
    return -1;
}


/*
bool processTriple( const std::string& fname1, 
                    const std::string& fname2, 
                    const std::string& testFname,
                    std::vector< FarnebackParams* > params )
{
    static bool first = true;
    IMGVEC img1(3), img2(3);
    cv::Mat imgTest;

    if ( first )
    {
        // I don't actually think I want to thread here
        std::cout << "about to thread" << std::endl;
        std::thread readFirst( getChannels_t, fname1, std::ref( img1 ) );
        std::thread readSecond( getChannels_t, fname2, std::ref( img2 ) );

        readFirst.join();
        readSecond.join();
        first = false;
    }
    else
    {
        // FIXME what did I intend to put here?
    }  

    imgTest = cv::imread( testFname, cv::IMREAD_COLOR );
    if (img1.empty() || img2.empty() || imgTest.empty())
    {
        return false;
    }

    // Run the algorithm
    std::for_each( params.begin(),
                   params.end(),
                   [img1, img2, imgTest] ( FarnebackParams* parm ) mutable
        {
            processWithParam( img1, img2, imgTest, parm );
        }
    );

    // FIXME error handling
    return true;
}
*/

// Accepts 2 images, split into their BGR channels, a test image and a processor object
// Calls the processor to generate the output channels, combines them into a BGR output
//  image, compares it to the test image.
bool processWithParam( const cv::Mat& img1, 
                       const cv::Mat& img2, 
                       const cv::Mat& test, 
                       OpticalFlowABC& proc )
{
    IMGVEC chnls1( 3 ), chnls2( 3 );    // per-channel image data
    IMGVEC dest( 3 );       // generated single-channel image (NOT 1 3-channel image)
    std::vector< std::thread > tasks;  // pre-channel threads

    cv::split( img1, chnls1 );
    cv::split( img2, chnls2 );

    // Per-channel processing
    for( int chnl = 0; chnl < 3; ++chnl )
    {
        dbg( "processing channel %d", chnl );

        std::thread task( processChannel_t, 
                          std::ref( chnls1[ chnl ] ),
                          std::ref( chnls2[ chnl ] ),
                          std::ref( dest[ chnl ] ),
                          std::ref( proc ) );
        tasks.push_back( std::move( task ) );
    }

    std::for_each( tasks.begin(), tasks.end(), []( std::thread& p ) { p.join(); } ); 

    // I should now have the 3 channels of an output image. Let's see
    cv::Mat fwd;
    cv::merge( dest, fwd );

    cv::imshow( "FWD", fwd );
    cv::waitKey( 0 );

    // And use the full-colour structural similarity algorithm to determine our "fit"
    proc.storeFit( getMSSIM( fwd, test ) );

/*
    // Save the generated image for reference / amusement
    std::stringstream outName;
    outName << "GEN." << imageFiles[ idx + 2 ];
    dbg( outName.str().c_str() );
    cv::imwrite( outName.str().c_str(), fwd );
*/

    return true;
}


void processChannel_t( const cv::Mat& img1,
                       const cv::Mat& img2,
                       cv::Mat& dest,
                       OpticalFlowABC& proc )
{
    proc.execute( img1, img2, dest );
}


void getChannels_t( const std::string& fname, IMGVEC& img )
{
    std::for_each( img.begin(), img.end(), []( cv::Mat& chnl ) { chnl.release();} );
    cv::Mat colour = cv::imread( fname, cv::IMREAD_COLOR );

    // cv::imshow( "getChannels() Colour", colour );
    // cv::waitKey(0);

    if (colour.empty())
    {
        dbg( "Could not read image from %s", fname );
    }
    else
    {
        dbg( "read colour %s", fname.c_str() );
        cv::split( colour, img );
        dbg( "split into channels" );
    }
}

// TODO DELETE ME
void buildParams( const std::string& fname, std::vector< FarnebackParams* >& params )
{
    for( int iscale = 1; iscale < 10; ++iscale )
    {
        FarnebackParams* pParm = new FarnebackParams;
        pParm->scale = (double) iscale / 10.0;

        pParm->levels = 3;
        pParm->smoothingSize = 15;
        pParm->iterations = 3;
        pParm->polyArea = 5;
        pParm->polyWidth = 1.2;

        params.push_back( pParm );
        break;
    }
}


std::vector< cv::Mat > getChannels( const cv::String& filename )
{
    std::vector< cv::Mat > channels( 3 );
    cv::Mat colour = cv::imread( filename, cv::IMREAD_COLOR );

    // cv::imshow( "getChannels() Colour", colour );
    // cv::waitKey(0);

    if (colour.empty())
    {
        dbg( "Could not read image from %s", filename );
    }
    else
    {
        dbg( "read colour %s", filename.c_str() );
        cv::split( colour, channels );
        dbg( "split into channels" );
    }

    return channels;
}


// TODO DELETE ME
void generateFlow( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& flow, FarnebackParams* params )
{
    dbg( "scale is %f", params->scale );

    // cv::imshow( "generateFlow() 1 gs", img1 );
    // cv::waitKey(0);
    // cv::imshow( "generateFlow() 2 gs", img2 );
    // cv::waitKey(0);

    cv::calcOpticalFlowFarneback( img1,                 // An input image
                                  img2,                 // Image immediately subsequent to 'prevImg'
                                  flow,                 // Flow vectors will be recorded here
                                  params->scale,         // Scale between pyramid levels (< '1.0')
                                  params->levels,        // Number of pyramid levels
                                  params->smoothingSize, // Size of window for pre-smoothing pass
                                  params->iterations,    // Iterations for each pyramid level
                                  params->polyArea,      // Area over which polynomial will be fit
                                  params->polyWidth,     // Width of fit polygon, usually '1.2*polyN'
                                  0 );                  // Option flags, combine with OR operator

}


void remapImage( const cv::Mat& src, cv::Mat& dest, const cv::Mat& flow )
{
    cv::remap( src,            // starting image for the extrapolation
               dest,            // output image
               flow,            // mapping matrix
               cv::Mat(),       // y mapping matrix, not needed here as flow is (x,y) data
               cv::INTER_LINEAR,   // interpolation method
               cv::BORDER_TRANSPARENT,  // border mode for extrapolations
               0 );             // border value, not needed here
}


void dbg( const std::string fmt, ... )
{
#ifdef _DEBUG
    char buf[1024];         // cross fingers 
    va_list args;
 
    va_start( args, fmt );
    vsprintf( buf, fmt.c_str(), args );
    std::cout << buf << std::endl;
    va_end( args );
#endif
}

void usage()
{
    std::cout << "Usage: rain [-t|--train <param-out-file>] | [-r|--run <param-in-file>] <src-dir>" << std::endl;
}