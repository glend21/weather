// OpenCV hello world
// Reads, displays, and writes an image

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;

int main()
{
    // std::string image_path = "../data/1/IDR703.T.202007221300.png";
    std::string image_path = cv::samples::findFile( "starry_night.jpg" );
    Mat img = imread( image_path, IMREAD_COLOR );

    if ( img.empty() )
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    imshow( "Display window", img );
    int k = waitKey( 0 );
    if (k == 's')
    {
        imwrite( "radar_1_1.jpg", img );
        imwrite( "starry.png", img );
    }

    return 0;
}
