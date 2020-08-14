/*
 * Use optical flowing tracking to predict rain movements from radar images
 *
 * Usage: rain [-t|--train <param-out-file>] | [-r|--run <param-in-file>] <src-dir>
 *
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

#include "ssim.hpp"


#define _DEBUG
#define IMG_TYPE ".png"     // This gets used directly to build the input file list

// Types
typedef enum { Blue, Green, Red } CHANNELS;

struct FarnebackParams {
    float scale;
    int levels,
        smoothingSize,
        iterations,
        polyArea;
    float polyWidth;

    float ssimScores[3];
};


// Fwd declarations, what a thing!
int process( const std::vector< std::string>& images );
void buildParams( std::vector< FarnebackParams* >& params );
void cleanupParams( std::vector< FarnebackParams* >& params );
void getChannels( const cv::String& filename, cv::Mat* channels );
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
    DIR *dir = opendir( argv[1] );
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

    retval = process( srcList );
    return retval;
}


int process( const std::vector< std::string>& images )
{
    std::vector< FarnebackParams* > params;
    buildParams( params );

    cv::Mat channels1[ 3 ];
    cv::Mat channels2[ 3 ];
    cv::Mat channelsTest[ 3 ];
    bool first = true;

    for ( int idx = 0; idx < images.size() - 2; ++idx )
    {
        if (first)
        {
            // Split each input image into BGR channels
            getChannels( images[ idx ], channels1 );
            getChannels( images[ idx + 1 ], channels2 );
            first = false;
        }
        else
        {
            for( int idx = 0; idx < 3; ++idx )
            {
                // Copy 2 to 1, Test to 2
                channels2[ idx ].copyTo( channels1[ idx ] );
                channelsTest[ idx ].copyTo( channels2[ idx ] );
            }
        }
        // Always fetch the test image
        dbg( "pre" );
        getChannels( images[ idx + 2 ], channelsTest );

        // Iterate over all parameter sets, generating an optical flow array for each channel
        std::for_each( params.begin(),
                       params.end(),
                       [ &channels1, &channels2, &channelsTest ]( FarnebackParams* parm ) mutable
                       {
                            // This is going to be one hell of a lmbda
                            cv::Mat flow[3];

                            // Repeat for each colour channel
                            for( int chnl = 0; chnl < 3; ++chnl )
                            {
                                generateFlow( channels1[ chnl ], 
                                              channels2[ chnl ], 
                                              flow[ chnl ], 
                                              parm );
                                dbg( "flow generated %d", chnl );

                                cv::Mat dest;
                                remapImage( channels2[ chnl ], dest, flow[ chnl ] );
                                dbg( "remapped %d", chnl );

                                cv::imshow( "Remap", dest );
                                cv::waitKey( 0 );

                                // FIXME
                                //parm->ssimScores[ chnl ] = getMSSIM( channelsTest[ chnl ], dest );
                                //dbg( "Chnl %d: %f", chnl, parm->ssimScores[ chnl ] );
                            }
                        }
                      );

        // for( std::iterator< FarnebackParams* > armItr = params.begin(); parmItr != params.end(); ++parmItr )

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
    }
}


void getChannels( const cv::String& filename, cv::Mat* channels )
{
    cv::Mat colour = cv::imread( filename, cv::IMREAD_COLOR );
    if (colour.empty())
    {
        dbg( "Could not read image from %s", filename );
    }
    else
    {
        dbg( "read colour %s", filename.c_str() );
    }

    cv::split( colour, channels );
    dbg( "split into channels" );
}


void generateFlow( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& flow, FarnebackParams* params )
{
    dbg( "scale is %f", params->scale );

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
               cv::INTER_NEAREST,   // interpolation method
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