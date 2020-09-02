
#include "farneback_flow.hpp"

#include <vector>
#include <sstream>

#include <opencv2/video/tracking.hpp>
#include "except.hpp"


// The generator
OpticalFlowABC& FarnebackFlow::FarnebackGenerator::operator()()
{
    // This is where all the magic happens
    // If, by "magic" I mean "brute force". This is a combinatorial explosion waiting to happen
    // TODO: research an algorithm to find the max (in this case) of an n-dim value set
    //       some sort of multivariate gradient (asc)ent ?
    static std::vector< float > vec_scale { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    static std::vector< float >::iterator itr_scale = vec_scale.begin();
    static std::vector< int > vec_levels { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    static std::vector< int >::iterator itr_levels = vec_levels.begin();
    static std::vector< int > vec_smoothing { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 };
    static std::vector< int >::iterator itr_smoothing = vec_smoothing.begin();
    static std::vector< int > vec_iterations { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    static std::vector< int >::iterator itr_iter = vec_iterations.begin();
    static std::vector< int > vec_area { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    static std::vector< int >::iterator itr_area = vec_levels.begin();

    // Returns an initialised flow object
    FarnebackFlow* pFlow = new FarnebackFlow(
        *itr_scale,
        *itr_levels,
        *itr_smoothing,
        *itr_iter,
        *itr_area,
        (*itr_area * 1.2) + 0.1 );

    ++itr_scale;
    if (itr_scale == vec_scale.end())
    {
        itr_scale = vec_scale.begin();
        ++itr_levels;
    }
    if (itr_levels == vec_levels.end())
    {
        itr_levels = vec_levels.begin();
        ++itr_smoothing;
    }
    if (itr_smoothing == vec_smoothing.end())
    {
        itr_smoothing = vec_smoothing.begin();
        ++itr_iter;
    }
    if (itr_iter == vec_iterations.end())
    {
        itr_iter = vec_iterations.begin();
        ++itr_area;
    }
    if (itr_area == vec_area.end())
    {
        // We have exhausted all combinations
        throw RainException( "FB no more combinations" );
    }

    return *pFlow;
}


// The optical flow itself
FarnebackFlow::FarnebackFlow(
        float scale,
        int levels,
        int smoothingSize,
        int iterations,
        int polyArea,
        float polyWidth )
{
    this->scale = scale;
    this->levels = levels;
    this->smoothingSize = smoothingSize;
    this->iterations = iterations;
    this->polyArea = polyArea;
    this->polyWidth = polyWidth;
}


bool FarnebackFlow::execute( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& imgOut )
{
    cv::Mat flow;

    // Calculate the (reverse) optical flow field
    // Note that we are calculating from img2 to img1. This is important
/*
    cv::calcOpticalFlowFarneback( img2,                // ** An input image
                                  img1,                // ** Image immediately previous to img2
                                  flow,                // Flow vectors will be recorded here
                                  this->scale,         // Scale between pyramid levels (< '1.0')
                                  this->levels,        // Number of pyramid levels
                                  this->smoothingSize, // Size of window for pre-smoothing pass
                                  this->iterations,    // Iterations for each pyramid level
                                  this->polyArea,      // Area over which polynomial will be fit
                                  this->polyWidth,     // Width of fit polygon, usually '1.2*polyN'
                                  0 );                  // Option flags, combine with OR operator
*/
///*
 // This is parameter set I have seen in the literature
 // Limited testing shows a better fit than the first few iterations of the brute force approach
    cv::calcOpticalFlowFarneback( img2,                // ** An input image
                                  img1,                // ** Image immediately previous to img2
                                  flow,                // Flow vectors will be recorded here
                                  0.5,         // Scale between pyramid levels (< '1.0')
                                  3,        // Number of pyramid levels
                                  15, // Size of window for pre-smoothing pass
                                  3,    // Iterations for each pyramid level
                                  5,      // Area over which polynomial will be fit
                                  1.2,     // Width of fit polygon, usually '1.2*polyN'
                                  0 );                  // Option flags, combine with OR operator
//*/

    // OK, here's where the (other) magic happens
    // Create a mapping array from the flow data
    cv::Mat map( flow.size(), CV_32FC2 );
    for ( int y = 0; y < map.rows; ++y )
    {
        for ( int x = 0; x < map.cols; ++x )
        {
            const cv::Point2f& f = flow.at< cv::Point2f >( y, x );
            map.at< cv::Point2f >( y, x ) = cv::Point2f( x + f.x, y + f.y );
        }
    }

    // Nooow map that to image 2
    cv::remap( img2,            // starting image for the extrapolation
               imgOut,          // output image
               map,             // mapping matrix
               cv::Mat(),       // y mapping matrix, not needed here as flow is (x,y) data
               cv::INTER_LINEAR,   // interpolation method
               cv::BORDER_TRANSPARENT,  // border mode for extrapolations
               0 );             // border value, not needed here

    return true;
}


const std::string FarnebackFlow::paramHeaders()
{
    std::stringstream osh;

    osh << "Algo" << ','
        << "scale" <<','
        << "levels" << ','
        << "smoothingSize" << ','
        << "iterations" << ','
        << "polyArea" << ','
        << "polyWidth" << ',' 
        << "ssimScore_B" << ','
        << "ssimScore_G" << ',' 
        << "ssimScore_R" << ','
        << "ssimScore_mean";

    return osh.str();       // return by value as it's only little
}


const std::string FarnebackFlow::params()
{
    std::stringstream osh;
    float avg = (this->ssimScore[0] + this->ssimScore[1] + this->ssimScore[2] ) / 3.0;

    osh << "fb,"
        << this->scale <<','
        << this->levels << ','
        << this->smoothingSize << ','
        << this->iterations << ','
        << this->polyArea << ','
        << this->polyWidth << ',' 
        << this->ssimScore[0] << ',' << this->ssimScore[1] << ',' << this->ssimScore[1] << ','
        << avg;

    return osh.str();       // OK to return by value
}