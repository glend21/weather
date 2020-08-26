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
#include <chrono>

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
/*
bool processTriple( const std::string& fname1, 
                    const std::string& fname2, 
                    const std::string& testFname,
                    std::vector< FarnebackParams* > params );
*/
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
void getChannels_t( const std::string& fname, IMGVEC& img );

void buildParams( const std::string& fname, std::vector< FarnebackParams* >& params );
void cleanupParams( std::vector< FarnebackParams* >& params );
std::vector< cv::Mat > getChannels( const cv::String& filename );
void generateFlow( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& flow, FarnebackParams* params );
void remapImage( const cv::Mat& src, cv::Mat& dest, const cv::Mat& flow );
void dbg( std::string fmt, ... );
void usage();
bool dbgImage( const cv::Mat& img );

void myMerge(const cv::Mat* mv, size_t n, cv::OutputArray _dst);

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
                // dbg( "adding %s to src image list", path.str().c_str() );
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
    std::chrono::steady_clock::time_point begin_all = std::chrono::steady_clock::now();

    try
    {
        outName << options.destDir << '/' << options.paramFile;
        ofh.open( outName.str() );
        if ( ofh.fail() )
        {
            throw RainException( "Could not open parameter file" );
        }

        // Iterate over all parameter combinations, until generate() throws an exception
        std::chrono::steady_clock::time_point begin_param;
        for ( long n = 0l; ; ++n )
        {
            dbg( "Param iteration: %d", n );
            begin_param = std::chrono::steady_clock::now();

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

                /*
                if ( ! dbgImage( newImg ) )
                {
                    exit( 0 );
                }
                */
            }

            delete &proc;

            // End timer for this parameter set
            std::chrono::steady_clock::time_point end_param = std::chrono::steady_clock::now();
            std::cout << "Time for 1 parameter set: " 
                      << std::chrono::duration_cast< std::chrono::milliseconds >(end_param- begin_param).count() << std::endl;
        }   // for each parameter combination

        ofh.close();
    }
    catch ( RainException& ex )
    {
        std::cout << "Exception: " << ex.getMsg() << std::endl;
        ofh.close();
    }

    // End timer for all data
    std::chrono::steady_clock::time_point end_all = std::chrono::steady_clock::now();
    std::cout << "Time for all processing: " 
              << std::chrono::duration_cast< std::chrono::milliseconds >(end_all - begin_all).count() << std::endl;

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
        // dbg( "processing channel %d", chnl );

        std::thread task( processChannel_t, 
                          std::ref( chnls1[ chnl ] ),
                          std::ref( chnls2[ chnl ] ),
                          std::ref( outChnls[ chnl ] ),
                          std::ref( proc ) );
        tasks.push_back( std::move( task ) );
    }

    std::for_each( tasks.begin(), tasks.end(), []( std::thread& p ) { p.join(); } ); 

    // I should now have the 3 channels of an output image. Let's see
    //dest[ 3 ] = cv::Mat::zeros( dest[ 1 ].size(), dest[ 1 ].type() );
    // dbg( "AAA dest: (%d, %d, %d)", dest[3].rows, dest[3].cols, dest[3].depth() );
    // dbg( "AAA outImg: (%d, %d, %d)", outImg.rows, outImg.cols, outImg.depth() );

    addTransparency( outChnls );
    cv::merge( outChnls, outImg );
    dbg( "Created image of shape (%d, %d, %d)", outImg.rows, outImg.cols, outImg.channels() );
    dbg( "Created image of type %d", outImg.type() );

    // And use the full-colour structural similarity algorithm to determine our "fit"
    proc.storeFit( getMSSIM( outImg, test ) );

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


void addTransparency( IMGVEC& channels )
{
    cv::Mat tmp( channels[ 0 ].size(), channels[ 0 ].depth() ),
            grey( channels[ 0 ].size(), channels[ 0 ].depth() ),
            alpha( channels[ 0 ].size(), channels[ 0 ].depth() );

    dbg( "C1   : (%d, %d) - %d", channels[0].size[0], channels[0].size[1], channels[0].depth() );
    dbg( "C2   : (%d, %d) - %d", channels[1].size[0], channels[1].size[1], channels[1].depth() );
    dbg( "C3   : (%d, %d) - %d", channels[2].size[0], channels[2].size[1], channels[2].depth() );
    dbg( "tmp  : (%d, %d) - %d", tmp.cols, tmp.rows, tmp.depth() );
    dbg( "grey : (%d, %d) - %d", grey.cols, grey.rows, grey.depth() );
    dbg( "alpha: (%d, %d) - %d", alpha.cols, alpha.rows, alpha.depth() );

    // Merge the input channels and convert to greyscale. 
    cv::merge( channels, tmp );
    //myMerge( &channels[0], channels.size(), tmp );
    cv::cvtColor( tmp, grey, cv::COLOR_BGRA2GRAY );

    // Threshold it so that any pixel with a colour value is converted to the max value
    cv::threshold( grey, alpha, 0, 255, cv::THRESH_BINARY );

    // Add that image as the alpha channel to outImg
    //alpha.copyTo( channels[ 3 ] );
    channels.push_back( alpha );
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


// Return true to continue
bool dbgImage( const cv::Mat& img )
{
    cv::imshow( "dbgImage", img );
    return cv::waitKey( 0 ) != 'q';
}


void usage()
{
    std::cout << "Usage: rain [-t|--train <param-out-file>] | [-r|--run <param-in-file>] <src-dir>" << std::endl;
}

// - - -
/*
using namespace cv;

void myMerge(const Mat* mv, size_t n, OutputArray _dst)
{
    //CV_INSTRUMENT_REGION();

    CV_Assert( mv && n > 0 );

    int depth = mv[0].depth();
    bool allch1 = true;
    int k, cn = 0;
    size_t i;

    for( i = 0; i < n; i++ )
    {
        CV_Assert(mv[i].size == mv[0].size && mv[i].depth() == depth);
        allch1 = allch1 && mv[i].channels() == 1;
        cn += mv[i].channels();
    }

    CV_Assert( 0 < cn && cn <= CV_CN_MAX );
    _dst.create(mv[0].dims, mv[0].size, CV_MAKETYPE(depth, cn));
    Mat dst = _dst.getMat();

    if( n == 1 )
    {
        mv[0].copyTo(dst);
        return;
    }

    CV_IPP_RUN(allch1, ipp_merge(mv, dst, (int)n));

    if( !allch1 )
    {
        AutoBuffer<int> pairs(cn*2);
        int j, ni=0;

        for( i = 0, j = 0; i < n; i++, j += ni )
        {
            ni = mv[i].channels();
            for( k = 0; k < ni; k++ )
            {
                pairs[(j+k)*2] = j + k;
                pairs[(j+k)*2+1] = j + k;
            }
        }
        mixChannels( mv, n, &dst, 1, &pairs[0], cn );
        return;
    }

    MergeFunc func = getMergeFunc(depth);
    CV_Assert( func != 0 );

    size_t esz = dst.elemSize(), esz1 = dst.elemSize1();
    size_t blocksize0 = (int)((BLOCK_SIZE + esz-1)/esz);
    AutoBuffer<uchar> _buf((cn+1)*(sizeof(Mat*) + sizeof(uchar*)) + 16);
    const Mat** arrays = (const Mat**)_buf.data();
    uchar** ptrs = (uchar**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &dst;
    for( k = 0; k < cn; k++ )
        arrays[k+1] = &mv[k];

    NAryMatIterator it(arrays, ptrs, cn+1);
    size_t total = (int)it.size;
    size_t blocksize = std::min((size_t)CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn), cn <= 4 ? total : std::min(total, blocksize0));

    for( i = 0; i < it.nplanes; i++, ++it )
    {
        for( size_t j = 0; j < total; j += blocksize )
        {
            size_t bsz = std::min(total - j, blocksize);
            func( (const uchar**)&ptrs[1], ptrs[0], (int)bsz, cn );

            if( j + blocksize < total )
            {
                ptrs[0] += bsz*esz;
                for( int t = 0; t < cn; t++ )
                    ptrs[t+1] += bsz*esz1;
            }
        }
    }
}
*/