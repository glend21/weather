''' Download data from BOM FTP site to new local dir for a particular radar sit code '''

import sys
import os

from ftplib import FTP


def main( argc, cargv ):
    ''' It '''

    if argc != 2:
        print( "Supply a radar site code on the cmd-line" )
        sys.exit(-1)

    site = sys.argv[1]
        
    # Create the output dir. Hardcoding will change later.
    basedir = "../data"
    subdirs = os.listdir( basedir )
    print( subdirs )
    newdir = os.path.join( basedir, "%02d" % (int(subdirs[ -1 ]) + 1) )
    print( newdir )
    os.mkdir( newdir )

    try:
        ftp = FTP( "ftp2.bom.gov.au" )
        ftp.login( user="anonymous", passwd="guest" )
        ftp.cwd( "anon/gen/radar/" )

        lines = []
        ftp.retrlines( "LIST IDR%s*.png" % site, lambda f : lines.append( f.split()[ -1 ] ) )

        for ln in lines:
            outpath = os.path.join( newdir, ln )
            print( "%s --> %s" % (ln, outpath) )
            ftp.retrbinary( "RETR %s" % ln, open( outpath, 'wb' ).write )

        ftp.quit()

    except Exception as ex:
        msg = "FTP failed - %s" % ex
        ofp = open( os.path.join( newdir, "FtpFail.txt" ), "wt" )
        ofp.write( msg )
        ofp.close()
        print( msg )


if __name__ == "__main__":
    main( len( sys.argv ), sys.argv )
