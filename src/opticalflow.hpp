
#ifndef __OPTICALFLOW_BASE__
#define __OPTICALFLOW_BASE__

/*
 *  Abstract base for all flow classes
 */

#include <opencv2/core.hpp>


// The flow class itself
class OpticalFlowABC
{
private:
    //OpticalFlowParamsABC* params;

protected:
    class Generator
    {
    public:
        virtual OpticalFlowABC& operator()() = 0;
    };

public:
    //virtual const char* mnemonic() = 0;

    static OpticalFlowABC& generate( const char* mnemonic );

    virtual bool execute( const cv::Mat& img1, const cv::Mat& img2 ) = 0;

    virtual bool save( const std:string& fname ) = 0;
};

#endif