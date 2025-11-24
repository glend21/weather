'''
	Look for favourable weather conditions for photography (or anything really)

	GD Nov25
'''

import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry



def main() -> None:
	''' It '''

	# Read the config file, which will contain the definitions of what we are planning for

	# Init
	cache_session = requests_cache.CachedSession( ".cache", expire_after=3600 )
	retry_session = retry( cache_session, retries=5, backoff_factor=0.2 )
	openmeteo = openmeteo_requests.Client( session=retry_session )

	# Weather call
	meteo_params: {} = {
		"latitude" : 31.83,
		"longitude" : 115.77,
		"hourly" : [ "temperature_2m",
					 "precipitation_probability",
					 "cloud_cover",
					 "wind_direction_10m",
					 "wind_speed_10m"
				   ]
	}

	response = openmeteo.weather_api( "https://api.open-meteo.com/v1/forecast",
								      meteo_params )
	hourly_data = response[ 0 ].Hourly()
	h_temp = hourly_data.Variables( 0 ).ValuesAsNumpy()
	h_precip = hourly_data.Variables( 1 ).ValuesAsNumpy()
	h_cloud = hourly_data.Variables( 2 ).ValuesAsNumpy()
	h_wind_dir = hourly_data.Variables( 3 ).ValuesAsNumpy()
	h_wind = hourly_data.Variables( 4 ).ValuesAsNumpy()

	hourly = { "date" : pd.date_range( start=pd.to_datetime( hourly_data.Time(), unit='s', utc=False ),
									   end=pd.to_datetime( hourly_data.TimeEnd(), unit='s', utc=False ),
									   freq = pd.Timedelta( seconds=hourly_data.Interval() ),
									   inclusive="left"
									 )
			 }
	hourly[ "temperature_2m" ] = h_temp
	hourly[ "precipitation_prob" ] = h_precip
	hourly[ "cloud" ] = h_cloud
	hourly[ "h_wind_dir" ] = h_wind_dir
	hourly[ "wind" ] = h_wind
	hourly_df = pd.DataFrame( data=hourly )

	print( "\nHourly data\n", hourly_df )


if __name__ == "__main__":
	main()


'''
# Setup the Open-Meteo API client with cache and retry on error

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
params = {
	"latitude": 52.52,
	"longitude": 13.41,
	"hourly": "temperature_2m",
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)
'''
