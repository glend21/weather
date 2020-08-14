/*
 * First pass at using dense optical flow to extrapolate rain radar images
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <iostream>

int main( int argc, char **argv )
{
    if ( argc != 3 )
    {
        std::cerr << "Usage: " << argv[0] << " <first_img> <second_img>" << std::endl;
        return 1;
    }

    cv::Mat img1 = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    cv::Mat img2 = cv::imread( argv[2], cv::IMREAD_GRAYSCALE );

    if ( img1.empty() || img2.empty() )
    {
        std::cerr << "Could not read input file(s)" << std:: endl;
        std::cerr << "Usage: " << argv[0] << " <first_img> <second_img>" << std:: endl;
        return 1;
    }

    std::cout << "Images are loaded" << std:: endl;

    cv::Mat flow;
    cv::calcOpticalFlowFarneback( img1,         // An input image
                                  img2,         // Image immediately subsequent to 'prevImg'
                                  flow,         // Flow vectors will be recorded here
                                  0.5,          // Scale between pyramid levels (< '1.0')
                                  5,            // Number of pyramid levels
                                  13,           // Size of window for pre-smoothing pass
                                  10,           // Iterations for each pyramid level
                                  5,            // Area over which polynomial will be fit
                                  1.1,          // Width of fit polygon, usually '1.2*polyN'
                                  0 );          // Option flags, combine with OR operator

    std::cout << "Flow computed" << std:: endl;

    cv::Mat dest;
    cv::remap( img2,            // starting image for the extrapolation
               dest,            // output image
               flow,            // mapping matrix
               cv::Mat(),       // y mapping matrix, not needed here as flow is (x,y) data
               cv::INTER_NEAREST,   // interpolation method
               cv::BORDER_TRANSPARENT,  // border mode for extrapolations
               0 );             // border value, not needed here

    std::cout << "Extrapolated image completed" << std:: endl;

    //cv::imshow( "Flow image", flow );
    cv::imshow( "Extrapolated Image", dest );
    cv::waitKey();
    
    return 0;
}



