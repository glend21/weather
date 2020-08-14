#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

using std::cout;
using std::endl;


int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, "{@input | box.png | input image}" );
    Mat src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_COLOR );
    if ( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints;
    detector->detect( src, keypoints );

    //-- Draw keypoints
    Mat img_keypoints;
    Mat blank( src.size[1], src.size[2], src.type() );
    drawKeypoints( blank, keypoints, img_keypoints );

    //-- Show detected (drawn) keypoints
    imshow("SURF Keypoints", img_keypoints );
    waitKey();
    return 0;
}
