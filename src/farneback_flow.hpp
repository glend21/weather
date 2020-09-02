
#ifndef __FARNEBACKFLOW__
#define __FARNEBACKFLOW__

/*
 * The Farneback optical flow process
 */

#include "opticalflow.hpp"

#include <opencv2/core.hpp>


class FarnebackFlow : OpticalFlowABC
{
private:
    float scale;
    int levels,
        smoothingSize,
        iterations,
        polyArea;
    float polyWidth;

public:
    class FarnebackGenerator : OpticalFlowABC::Generator
    {
    public:
        virtual OpticalFlowABC& operator()();
    };

public:
    FarnebackFlow(
        float scale,
        int levels,
        int smoothingSize,
        int iterations,
        int polyArea,
        float polyWidth );

    virtual bool execute( const cv::Mat& img1, const cv::Mat& img2, cv::Mat& imgOut );

    virtual const std::string paramHeaders();
    virtual const std::string params();
};

#endif