
#ifndef __TVL1FLOW__
#define __TVL1FLOW__

/*
 * The Dual-TVL1 optical flow process from openCV
 */

#include "opticalflow.hpp"

#include <opencv2/core.hpp>


class TVL1Flow : OpticalFlowABC {
private:
    // TODO how is this algo parameterised

public:
    class TVL1Generator : OpticalFlowABC::Generator
    {
    public:
        virtual OpticalFlowABC& operator()();
    };

public:
    TVL1Flow();

    virtual bool execute( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& imgOut );

    virtual const std::string paramHeaders();
    virtual const std::string params();

};

#endif