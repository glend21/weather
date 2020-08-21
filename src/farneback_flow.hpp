
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

    cv::Mat flow;       // FIXME this may move up to the base class

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

    virtual bool execute( const cv::Mat& img1, const cv::Mat& img2 );

};

#endif