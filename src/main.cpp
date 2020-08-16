/*
 * Use optical flowing tracking to predict rain movements from radar images
 *
 * Usage: rain [-t|--train <param-out-file>] | [-r|--run <param-in-file>] <src-dir>
 *
 */

/*
TODOs:
    - proper cmd-line
    - output of generated images to subdir for visual inspection
        - specify the output dir
    - permutations of o-flow parameters
    - output of parameter file
    - find max of the SSIM scalar (isn't it really a vector though?)
    - multi-thread the individual channel processing

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

#include <dirent.h>

#include <iostream>
#include <string>
#include <vector>
#include <cstdarg>
#include <thread>

#include "ssim.hpp"


#define _DEBUG
#define IMG_TYPE ".png"     // This gets used directly to build the input file list

// Types
typedef enum { Blue, Green, Red } CHANNELS;
typedef std::vector< cv::Mat > IMGVEC;

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
bool processTriple( const std::string& fname1, 
                    const std::string& fname2, 
                    const std::string& testFname,
                    std::vector< FarnebackParams* > params );
bool processWithParam( const IMGVEC& img1, 
                       const IMGVEC& img2, 
                       const cv::Mat& test, 
                       FarnebackParams* parm );
void processChannel_t( const cv::Mat& img1,
                       const cv::Mat& img2,
                       cv::Mat& dest,
                       FarnebackParams*& parm );
void getChannels_t( const std::string& fname, IMGVEC& img );

int process( const std::vector< std::string>& imageFiles, std::vector< FarnebackParams* > params );
void buildParams( std::vector< FarnebackParams* >& params );
void cleanupParams( std::vector< FarnebackParams* >& params );
std::vector< cv::Mat > getChannels( const cv::String& filename );
std::vector< cv::Mat > createImageVector( const cv::Size& sz );
void generateFlow( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& flow, FarnebackParams* params );
void remapImage( const cv::Mat& src, cv::Mat& dest, const cv::Mat& flow );
void dbg( std::string fmt, ... );
void usage();


using namespace std;

int main( int argc, char **argv )
{
    int retval = 0;

    /*
        Iterate over a list of radar images
        For each consecutive pair - 1 (ie until first image is last - 2)
            Load both images
            Separate each into B, G, R channels -> 3 greyscale images per input image
            Iterate over parameter sets
                Generate optical flow output between each pair of colour channels
                    Store the parameters used 
                Remap the 3 flow images with their channel base to produce 3 channel output images
                Combine the 3 output channels into a single BGR output
                Load the next image as BGR <- test image
                Separate it into channels
                Diff (SSIM) each calculated channel image with the corresponding test channel image
                    Store the diff scores next to the Farneback parameters
            Note the param set with the minimised diff scores

            Move the second BGR image into the first slot, the test image into the second slot
    */

    // TODO proper cmd-line handling
    if (argc == 1)
    {
        usage();
        return -1;
    }

    // Get all the src images. Easier if we cache the names, as iterating over them is non-trivial
    std::vector< std::string > srcList;
    DIR* dir = opendir( argv[1] );
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
                path << argv[1] << '/' << fname;
                dbg( "adding %s to src image list", path.str().c_str() );
                srcList.push_back( path.str() );
            }
        }

        closedir( dir );
    }
    else
    {
        // could not open directory
        dbg( "could not open %s", argv[1] );
        return EXIT_FAILURE;
    }

    if (srcList.size() < 3)
    {
        std::cout << "Need at least 3 images in the source dir";
        return EXIT_FAILURE;
    }

    std::vector< FarnebackParams* > params;
    buildParams( params );

    for ( int idx = 0; idx < srcList.size() - 2; ++idx )
    {
        retval = processTriple( srcList[ idx ], srcList[ idx + 1 ], srcList[ idx + 2 ], params );

        dbg( "SSIM vector: (%f, %f, %f)", 
             params[0]->ssimScore[ 0 ],
             params[0]->ssimScore[ 1 ],
             params[0]->ssimScore[ 2 ] );
        break;
    }

    //retval = process( srcList, params );
    return retval;
}


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


bool processWithParam( const IMGVEC& img1, 
                       const IMGVEC& img2, 
                       const cv::Mat& test, 
                       FarnebackParams* parm )
{
    IMGVEC dest( 3 );       // generated single-channel image
    std::vector< std::thread > tasks;  // pre-channel threads

    // Per-channel processing
    for( int chnl = 0; chnl < 3; ++chnl )
    {
        std::thread task( processChannel_t, 
                          std::ref( img1[ chnl ] ),
                          std::ref( img2[ chnl ] ),
                          std::ref( dest[ chnl ] ),
                          std::ref( parm ) );
        tasks.push_back( std::move( task ) );
    }

    std::for_each( tasks.begin(), tasks.end(), []( std::thread& p ) { p.join(); } ); 

    // I should now have the 3 channels of an output image. Let's see
    cv::Mat fwd;
    cv::merge( dest, fwd );

    cv::imshow( "FWD", fwd );
    cv::waitKey( 0 );

    // And use the full-colour structural similarity algorithm to determine our "fit"
    parm->ssimScore = getMSSIM( fwd, test );

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
                   FarnebackParams*& parm )
{
    cv::Mat flow;       // optical flow data
    generateFlow( img1, img2, flow, parm );
    dbg( "flow generated" );

    remapImage( img2, dest, flow );
    dbg( "remapped" );
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

// - - - - - - - - - - - - - - -
int process( const std::vector< std::string >& imageFiles, std::vector< FarnebackParams* > params )
{
    bool first = true;

    for ( int idx = 0; idx < imageFiles.size() - 2; ++idx )
    {
        IMGVEC img1;
        IMGVEC img2;
        cv::Mat imgRef;
        IMGVEC imgTest;

        if (first)
        {
            // Split each input image into BGR channels
            img1 = getChannels( imageFiles[ idx ] );
            img2 = getChannels( imageFiles[ idx + 1 ] );
            first = false;
        }
        else
        {
            std::copy( img2.begin(), img2.end(), img1.begin() );
            cv::split( imgRef, imgTest );
            /*
            for( int idx = 0; idx < 3; ++idx )
            {
                // Copy 2 to 1, Test to 2
                img2[ idx ].copyTo( img1[ idx ] );
                imgTest[ idx ].copyTo( img2[ idx ] );
            }
            */
        }

        // Always fetch the next test image
        dbg( "pre" );
        imgRef = cv::imread( imageFiles[ idx + 2 ], cv::IMREAD_COLOR);

        if (img1.empty() || img2.empty() || imgRef.empty())
        {
            return -1;
        }

        // cv::imshow( "B 1", img1[ 0 ] );
        // cv::imshow( "G 1", img1[ 1 ] );
        // cv::imshow( "R 1", img1[ 2 ] );
        // cv::waitKey( 0 );
        
        // Iterate over all parameter sets, generating an optical flow array for each channel
        //std::for_each( params.begin(),
        //               params.end(),
        //               [ &img1, &img2, &imgRef, &imgTest ]( FarnebackParams* parm ) mutable
        for ( std::vector< FarnebackParams* >::iterator parm = params.begin(); parm != params.end(); ++parm )
        {
            // This is going to be one hell of a lambda
            IMGVEC flow( 3 );       // = createImageVector( img1[ 0 ].size() );    // optical flow data
            IMGVEC dest( 3 );       // = createImageVector( img1[ 0 ].size() );    // generated single-channel image

            // Generate the flow data for each colour channel
            for( int chnl = 0; chnl < 3; ++chnl )
            {
                generateFlow( img1[ chnl ], img2[ chnl ], flow[ chnl ], *parm );
                dbg( "flow generated %d", chnl );

                remapImage( img2[ chnl ], dest[ chnl ], flow[ chnl ] );
                dbg( "remapped %d", chnl );

            }

            // cv::imshow( "Remap B", dest[ Blue ] );
            // cv::imshow( "Remap G", dest[ Green ] );
            // cv::imshow( "Remap R", dest[ Red ] );
            // cv::waitKey( 0 );

            // TODO
            // Now we combine the separate channels back into a BGR image
            cv::Mat fwd;
            cv::merge( dest, fwd );

            cv::imshow( "FWD", fwd );
            cv::waitKey( 0 );

            // And use the full-colour structural similarity algorithm to determine our "fit"
            (*parm)->ssimScore = getMSSIM( fwd, imgRef );
            dbg( "SSIM vector: (%f, %f, %f)", 
                 (*parm)->ssimScore[ 0 ],
                 (*parm)->ssimScore[ 1 ],
                 (*parm)->ssimScore[ 2 ] );

            // Save the generated image for reference / amusement
            std::stringstream outName;
            outName << "GEN." << imageFiles[ idx + 2 ];
            dbg( outName.str().c_str() );
            cv::imwrite( outName.str().c_str(), fwd );
      }

      break;
    }
}


void buildParams( std::vector< FarnebackParams* >& params )
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


std::vector< cv::Mat > createImageVector( const cv::Size& sz )
{
    std::vector< cv::Mat > vec;
    cv::Mat chnl = cv::Mat::zeros( sz, CV_8UC1 );

    for( int i = 0; i < 3; ++i )
    {
        vec.push_back( chnl );
    }

    return vec;
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