
#ifndef __OPTICALFLOW_BASE__
#define __OPTICALFLOW_BASE__

/*
 *  Abstract base for all flow classes
 */

#include <opencv2/core.hpp>

#include "except.hpp"


// The flow class itself
class OpticalFlowABC
{
protected:
    class Generator
    {
    public:
        virtual OpticalFlowABC& operator()() = 0;
    };

protected:
    cv::Scalar ssimScore;

public:
    static OpticalFlowABC& generate( const char* mnemonic, long limit = 0l );

    virtual bool execute( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& imgOut ) = 0;

    virtual const std::string paramHeaders() = 0;
    virtual const std::string params() = 0;

    void storeFit( const cv::Scalar& fit ) { this->ssimScore = fit; };
};

#endif