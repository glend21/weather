''' Wrapper script to init system and then run the mainline '''

# Usage: rain -a|--algo <algo> [-t|--train <param-out-file>] | [-r|--run <param-in-file>] <src-dir> <dest-dir>

import os
import sys
import subprocess

import argparse
from datetime import datetime


__ALGOS = [ "fb" ]
__IMGTYPE = ".png"
__EXE = "../src/rain"       # FIXME big red flag


def main( argc, argv ):
    ''' It '''

    # Handle the cmd-line
    parser = argparse.ArgumentParser()
    parser.add_argument( "-a", "--algo", nargs=1,
                         help="The optical flow algorithm to use" )
    group = parser.add_mutually_exclusive_group()
    group.add_argument( "-t", "--train", nargs=1, 
                        help="Train the model and output all parameter sets to param file" )
    group.add_argument( "-r", "--run", nargs=1, 
                        help="Run the model using the param file as input" )
    parser.add_argument( "source" )
    parser.add_argument( "dest", nargs='?' )
    args = parser.parse_args()

    print( "Algo: %s" % args.algo )
    print( "Training: %s" % args.train )
    print( "Run: %s" % args.run )
    print( "Source: %s" % args.source )
    print( "Dest: %s" % args.dest )

    # Enforce mandatory algorithm specification
    if not args.algo[ 0 ] in __ALGOS:
        print( "Optical flow algorithm must be specified" )
        parser.print_usage()
        sys.exit( -1 )

    # Normalise paths
    real_source = os.path.realpath( args.source )
    if args.dest is not None:
        real_dest = os.path.realpath( args.dest )
    else:
        real_dest = "%s.train.%s" % (os.path.realpath( args.source ), datetime.now().strftime( "%Y%m%d%H%M" ))
    print( "Real dest: %s" % real_dest )

    # Make sure at least the source dir exists
    if not os.path.exists( real_source ):
        print( "Source dir %s does not exist" )
        sys.exit( -1 )

    if args.train is not None:
        # Set up dirs for a training run
        src_count = len( [ f for f in os.listdir( real_source ) if os.path.isfile( os.path.join( real_source, f ) ) and f[ -4 : ] == __IMGTYPE ] )
        print( "%d source files" % src_count )

        for i in range( 1, src_count - 1 ):
            ddir = os.path.join( real_dest, "%02d_%02d" % (i, i + 1) )
            print( "Making %s" % ddir )
            os.makedirs( ddir, exist_ok=True )

    else:
        pass

    # Now run the C++ exe
    # The cmd is now: exe <algo> <t|r> <param_file> <src> <dest>
    # Note that the C++ exe will assume this is correct, so bloddy get it right here
    args_out = [ __EXE, args.algo[0] ]
    if args.train is not None:
        args_out += [ 't', args.train[0] ]
    else:
        args_out += [ 'r', args.run[0] ]
    args_out += [ real_source, real_dest ]

    print( args_out )
    print( " ^ ^ ^ PYTHON ENDS HERE ^ ^ ^" )
    res = subprocess.run( args_out )


if __name__ == "__main__":
    main( len( sys.argv ), sys.argv )