
#include "opticalflow.hpp"
#include "farneback_flow.hpp"

#include <string.h>


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
    }
}


