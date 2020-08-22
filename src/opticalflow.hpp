
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
    //virtual const char* mnemonic() = 0;

    static OpticalFlowABC& generate( const char* mnemonic, long limit = 100l );

    virtual bool execute( const cv::Mat& img1, const cv::Mat& img2 ) = 0;

    virtual bool save( const std::string& fname ) = 0;
};

#endif