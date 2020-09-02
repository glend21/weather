
#include "tvl1_flow.hpp"

#include <opencv2/video/tracking.hpp>


// Lots of stuff goes in here

// The generator
OpticalFlowABC& TVL1Flow::TVL1Generator::operator()()
{
    TVL1Flow* flow = new TVL1Flow();
    return *flow;
}

// The flow class itself
TVL1Flow::TVL1Flow()
{

}


bool TVL1Flow::execute( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& imgOut )
{
    cv::Mat flow;
    cv::Ptr< cv::DualTVL1OpticalFlow > tvl1 = cv::DualTVL1OpticalFlow::create();

    tvl1->calc( img2, img1, flow );

    return true;
}


const std::string TVL1Flow::paramHeaders()
{
    return "";
}

const std::string TVL1Flow::params()
{
    return "";
}

