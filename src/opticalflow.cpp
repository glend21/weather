
#include "opticalflow.hpp"
#include "farneback_flow.hpp"

#include <string.h>


OpticalFlowABC& OpticalFlowABC::generate( const char* mnemonic )
{
    // I hate using std::string for trival strings and I hate using
    // C-style string functions
    if (! strcmp( mnemonic, "fb" ) )
    {
        FarnebackFlow::FarnebackGenerator gen;
        return gen();
    }
}


