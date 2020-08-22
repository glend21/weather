
#include "farneback_flow.hpp"

#include <vector>
#include <sstream>

#include <opencv2/video/tracking.hpp>


// The generator
OpticalFlowABC& FarnebackFlow::FarnebackGenerator::operator()()
{
    // This is where all the magic happens
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

    cv::calcOpticalFlowFarneback( img1,                 // An input image
                                  img2,                 // Image immediately subsequent to 'prevImg'
                                  flow,                 // Flow vectors will be recorded here
                                  this->scale,         // Scale between pyramid levels (< '1.0')
                                  this->levels,        // Number of pyramid levels
                                  this->smoothingSize, // Size of window for pre-smoothing pass
                                  this->iterations,    // Iterations for each pyramid level
                                  this->polyArea,      // Area over which polynomial will be fit
                                  this->polyWidth,     // Width of fit polygon, usually '1.2*polyN'
                                  0 );                  // Option flags, combine with OR operator

    cv::remap( img2,            // starting image for the extrapolation
               imgOut,          // output image
               flow,            // mapping matrix
               cv::Mat(),       // y mapping matrix, not needed here as flow is (x,y) data
               cv::INTER_LINEAR,   // interpolation method
               cv::BORDER_TRANSPARENT,  // border mode for extrapolations
               0 );             // border value, not needed here

    return true;
}


void FarnebackFlow::paramHeaders( std::string& str )
{
    std::stringstream osh;

    osh << "Algo"
        << "scale" <<','
        << "levels" << ','
        << "smoothingSize" << ','
        << "iterations" << ','
        << "polyArea" << ','
        << "polyWidth" << ',' 
        << "ssimScore_B" << ','
        << ">ssimScore_G" << ',' 
        << "ssimScore_R" << ','
        << "ssimScore_mean";

    str = osh.str();
}


void FarnebackFlow::params( std::string& str )
{
    std::stringstream osh;

    osh << "fb,"
        << this->scale <<','
        << this->levels << ','
        << this->smoothingSize << ','
        << this->iterations << ','
        << this->polyArea << ','
        << this->polyWidth << ',' 
        << this->ssimScore[0] << ',' << this->ssimScore[1] << ',' << this->ssimScore[1] << ','
        << cv::mean( this->ssimScore );

    str = osh.str();
}