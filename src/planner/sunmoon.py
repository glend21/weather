'''
    Determines position of sun and moon in sky given lat, long, date and time

    GD Dec25
'''

import sundialy



def main() -> None:
    ''' It '''

    retval, _ = sundialy.tools.SAMPA( year=2025,
                                      month=12,
                                      day=10,
                                      hour=4,       # UTC time by the looks of it
                                      minute=28,
                                      second=0,
                                      microsecond=0,
                                      latitude=-31.83626207727829,
                                      longitude=115.77857421424893,
                                      elevation=50,
                                      pressure=1013.7,
                                      temperature=19
                                    )
    breakpoint()
    if len( retval ) >= 4:
        data: { str, float } = { "sun_zenith" : retval[ 0 ],    # num degrees down from directly overhead
                                 "sun_azimuth" : retval[ 1 ],
                                 "moon_zenith" : retval[ 2 ],
                                 "moon_azimuth" : retval[ 3 ]
                               }
                               
        for key, val in data.items():
            print( f"{key} : {val}" )  


if __name__ == "__main__":
    main()
