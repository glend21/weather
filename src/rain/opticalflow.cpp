
#include "opticalflow.hpp"

#include <string.h>

#include "farneback_flow.hpp"
#include "tvl1_flow.hpp"
#include "except.hpp"


OpticalFlowABC& OpticalFlowABC::generate( const char* mnemonic, long limit )
{
    static long count = 0l;

    if (count == limit)
    {
        throw RainException( "Limit reached." );
    }
    else
    {
        // I hate using std::string for trival strings and I hate using
        // C-style string functions
        if ( ! strcmp( mnemonic, "fb" ) )
        {
            FarnebackFlow::FarnebackGenerator gen;
            ++count;
            return gen();
        }
        else if ( ! strcmp( mnemonic, "tvl") )
        {
            TVL1Flow::TVL1Generator gen;
            ++count;
            return gen();
        }
    }

    throw RainException( "Unknown optical flow algorithm" );
}
