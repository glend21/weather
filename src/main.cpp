/*
 * Use optical flowing tracking to predict rain movements from radar images
 *
 * This exe is called from a driver program, from which the args are guaranteed to be:
 *
 *  rain exe <algo> <t|r> <param_file> <src> <dest>
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
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <string.h>
#include <dirent.h>

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
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
bool processCmdLine_2(int argc, char** argv, struct CmdLineParams& params );
bool train( const CmdLineParams& options );
int run( const CmdLineParams& options );
bool processWithParam( const cv::Mat& img1, 
                       const cv::Mat& img2, 
                       const cv::Mat& test, 
                       cv::Mat& outImg,
                       OpticalFlowABC& proc );
void processChannel_t( const cv::Mat& img1,
                       const cv::Mat& img2,
                       cv::Mat& dest,
                       OpticalFlowABC& proc );
void addTransparency( IMGVEC& channels );

void usage();
void dbg( std::string fmt, ... );
bool dbgImage( const cv::Mat& img );


using namespace std;

int main( int argc, char **argv )
{
    int retval = 0;

    CmdLineParams options;
    bool b = processCmdLine_2( argc, argv, options );
    std::cout << "res: " << b << std::endl;
    std::cout << "algo: " << options.algo << std::endl;
    std::cout << "train? " << options.doTrain << std::endl;
    std::cout << "param file name: " << options.paramFile << std::endl;
    std::cout << "src dir: " << options.srcDir << std::endl;
    std::cout << "dest dir: " << options.destDir << std::endl;
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


bool processCmdLine_2(int argc, char** argv, struct CmdLineParams& params )
{
    params.algo = *(++argv);
    params.doTrain = **(++argv) == 't';
    params.paramFile = *(++argv);
    params.srcDir = *(++argv);
    params.destDir = *(++argv);

    return true;
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
                srcNames.push_back( path.str() );
                srcData.push_back( cv::imread( path.str(), cv::IMREAD_UNCHANGED ) );
                dbg( "Read an image of (%d, %d, %d)", srcData.back().rows, srcData.back().cols, srcData.back().channels() );
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
    std::stringstream outName;
    std::ofstream ofh;

    // Start timer
    const double begin_all = (double) cv::getTickCount();
    try
    {
        outName << options.destDir << '/' << options.paramFile;
        ofh.open( outName.str() );
        if ( ofh.fail() )
        {
            throw RainException( "Could not open parameter file" );
        }

        // Iterate over all parameter combinations, until generate() throws an exception
        double begin_param;
        for ( long n = 0l; ; ++n )
        {
            dbg( "Param iteration: %d", n );
            begin_param = (double) cv::getTickCount();

            OpticalFlowABC& proc = OpticalFlowABC::generate( options.algo, 1 );

            // Write header to the parameter file on first iteration
            if (n == 0l)
            {
                ofh << proc.paramHeaders() << ','
                    << "image_1" << ','
                    << "image_2" << ','
                    << "image_test" << std::endl;
            }

            // Iterate over all image pairs (bar the last) for this set of algorithm parameters
            for ( int idx = 0; idx < srcData.size() - 2; ++idx )
            {
                dbg( " Image-pair: %d %d", idx, idx + 1 );

                if ( srcData[ idx ].data == NULL) dbg( "READ ERROR 1" );
                if ( srcData[ idx + 1 ].data == NULL) dbg( "READ ERROR 2" );
                if ( srcData[ idx + 2 ].data == NULL) dbg( "READ ERROR 3" );

                // Generate the output image
                cv::Mat newImg( srcData[ idx ].size(), CV_8UC4 );   // BGRA output image
                processWithParam( srcData[ idx ], 
                                  srcData[ idx + 1 ], 
                                  srcData[ idx + 2 ],
                                  newImg,
                                  proc );

                // Save the output image to the correct location
                outName.seekp( 0 );
                outName << options.destDir << '/'
                        << std::setfill( '0' ) << std::setw( 2 ) << std::right << idx + 1
                        << '_' 
                        << std::setfill( '0' ) << std::setw( 2 ) << std::right << idx + 2
                        << '/' 
                        << std::setfill( '0' ) << std::setw( 5 ) << std::right << n
                        << IMG_TYPE;
                dbg( "Output filename ~%s~", outName.str().c_str() );

                if ( cv::imwrite( outName.str(), newImg ) )
                {
                    dbg( "Data saved (%d %d %d).", newImg.rows, newImg.cols, newImg.channels() );
                }
                else
                {
                    dbg( "NOT SAVED." );
                }

                // Save the parameter set
                ofh << proc.params() << ','
                    << srcNames[ idx ] << ','
                    << srcNames[ idx + 1 ] << ','
                    << srcNames[ idx + 2 ] << std::endl;
            }

            delete &proc;

            // End timer for this parameter set
            std::cout << "Time for 1 parameter set: " 
                      << (cv::getTickCount() - begin_param) / cv::getTickFrequency()
                      << std::endl;
        }   // for each parameter combination

        ofh.close();
    }
    catch ( RainException& ex )
    {
        std::cout << "Exception: " << ex.getMsg() << std::endl;
        ofh.close();
    }

    // End timer for all data
    std::cout << "Time for all processing: " 
              << (cv::getTickCount() - begin_all) / cv::getTickFrequency()
              << std::endl;

    return 0;
}


int run( const CmdLineParams& options )
{
    return -1;
}


// Accepts 2 images, split into their BGR channels, a test image and a processor object
// Calls the processor to generate the output channels, combines them into a BGR output
//  image, compares it to the test image.
bool processWithParam( const cv::Mat& img1, 
                       const cv::Mat& img2, 
                       const cv::Mat& test, 
                       cv::Mat& outImg,
                       OpticalFlowABC& proc )
{
    IMGVEC chnls1( 3 ), chnls2( 3 );    // per-channel image data
    IMGVEC outChnls( 3 );                   // generated BGRA image channels
    std::vector< std::thread > tasks;  // pre-channel threads

    cv::split( img1, chnls1 );
    cv::split( img2, chnls2 );

    // Per-colour channel processing
    for( int chnl = 0; chnl < 3; ++chnl )
    {
        std::thread task( processChannel_t, 
                          std::ref( chnls1[ chnl ] ),
                          std::ref( chnls2[ chnl ] ),
                          std::ref( outChnls[ chnl ] ),
                          std::ref( proc ) );
        tasks.push_back( std::move( task ) );
    }

    std::for_each( tasks.begin(), tasks.end(), []( std::thread& p ) { p.join(); } ); 

    // I should now have the 3 channels of an output image. Let's see
    addTransparency( outChnls );
    cv::merge( outChnls, outImg );
    dbg( "Created image of shape (%d, %d, %d)", outImg.rows, outImg.cols, outImg.channels() );
    dbg( "Created image of type %d", outImg.type() );

    // And use the full-colour structural similarity algorithm to determine our "fit"
    proc.storeFit( getMSSIM( outImg, test ) );

    return true;
}


void processChannel_t( const cv::Mat& img1,
                       const cv::Mat& img2,
                       cv::Mat& dest,
                       OpticalFlowABC& proc )
{
    proc.execute( img1, img2, dest );
}


void addTransparency( IMGVEC& channels )
{
    cv::Mat tmp( channels[ 0 ].size(), channels[ 0 ].depth() ),
            grey( channels[ 0 ].size(), channels[ 0 ].depth() ),
            alpha( channels[ 0 ].size(), channels[ 0 ].depth() );

    // Merge the input channels and convert to greyscale. 
    cv::merge( channels, tmp );
    cv::cvtColor( tmp, grey, cv::COLOR_BGRA2GRAY );

    // Threshold it so that any pixel with a colour value is converted to the max value
    cv::threshold( grey, alpha, 0, 255, cv::THRESH_BINARY );

    // Add that image as the alpha channel to outImg
    channels.push_back( alpha );
}


void usage()
{
    std::cout << "Usage: rain [-t|--train <param-out-file>] | [-r|--run <param-in-file>] <src-dir>" << std::endl;
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


// Return true to continue
bool dbgImage( const cv::Mat& img )
{
    cv::imshow( "dbgImage", img );
    return cv::waitKey( 0 ) != 'q';
}
