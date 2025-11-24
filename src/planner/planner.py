'''
	Look for favourable weather conditions for photography (or anything really)

	GD Nov25
'''

from typing import Any, Callable

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

import pyparsing as pp
from pyparsing import pyparsing_common as ppc


class Stack():
	''' A very simple stack class; a semantic wrapper around Python's in-built list type '''

	def __init__( self, values: [ Any ] = [] ):
		''' Ctor '''

		self._data: [ Any ] = values

	def push( self, value: Any ) -> None:
		''' Stack push '''
		self._data.append( value )

	def pop( self ) -> Any:
		''' Stack pop '''
		return self._data.pop()

	def peek( self ) -> Any:
		''' Stack peek '''
		return self._data[ -1 ]

	def is_empty( self ) -> bool:
		''' Is the stack empty? '''
		return len( self._data ) == 0

	def __str__( self ) -> str:
		''' Output formatter '''
		return str( self._data )

	def __repr__( self ) -> str:
		''' Output formatter '''
		return str( self._data )


# ---
def act_location( lex_stack: Stack ) -> (str, Any):
	''' Acts on the location variable and returns a key, value pair '''
	print( "act_location" )

	if len( lex_stack ) < 2:
		return None

def act_wind( lex_stack: Stack ) -> int:			# FIXME
	''' Handles the wind variable '''
	print( "act_wind" )
	breakpoint()
	pass


jump_table: { str: Callable } = {
	"location" : act_location,
	"wind" : act_wind,	
}


def parse( operation: str ) -> Stack | None:
	''' Parse the operation string into a stack of lexical elements, first one
		last in the stack (so it will be popped first, etc)
	'''

	# Define the grammar here, as this method will only be called once
	loc_val: pp.ParseExpression  = ppc.number + pp.Suppress( ',' ) + ppc.number
	location: pp.ParseExpression = pp.CaselessLiteral( "location" ) + pp.Suppress( '=' ) + loc_val
	wind: pp.ParseExpression 	 = pp.CaselessLiteral( "wind" ) + pp.one_of( "> <" ) + ppc.number

	grammar: pp.ParseExpression = pp.Or( [ location, wind ] ) 
	lex = grammar.parse_string( operation )
	print( lex )

	retval = list( lex )
	retval.reverse()			
	return retval


def main() -> None:
	''' It '''

	# Read the config file, which will contain the definitions of what we are planning for

	# Init
	cache_session = requests_cache.CachedSession( ".cache", expire_after=3600 )
	retry_session = retry( cache_session, retries=5, backoff_factor=0.2 )
	openmeteo = openmeteo_requests.Client( session=retry_session )

	# Always make the weather call, we will filter the results based on the specified operation
	meteo_params: {} = {
		"latitude" : 31.83,
		"longitude" : 115.77,
		"hourly" : [ "temperature_2m",
					 "precipitation_probability",
					 "precipitation",
					 "cloud_cover",
					 "wind_direction_10m",
					 "wind_speed_10m"
				   ]
	}

	response = openmeteo.weather_api( "https://api.open-meteo.com/v1/forecast",
								      meteo_params )
	hourly_data = response[ 0 ].Hourly()
	h_temp = hourly_data.Variables( 0 ).ValuesAsNumpy()
	h_precip_prob = hourly_data.Variables( 1 ).ValuesAsNumpy()
	h_precip = hourly_data.Variables( 2 ).ValuesAsNumpy()
	h_cloud = hourly_data.Variables( 3 ).ValuesAsNumpy()
	h_wind_dir = hourly_data.Variables( 4 ).ValuesAsNumpy()
	h_wind = hourly_data.Variables( 5 ).ValuesAsNumpy()

	hourly = { "date" : pd.date_range( start=pd.to_datetime( hourly_data.Time(), unit='s', utc=False ),
									   end=pd.to_datetime( hourly_data.TimeEnd(), unit='s', utc=False ),
									   freq = pd.Timedelta( seconds=hourly_data.Interval() ),
									   inclusive="left"
									 )
			 }
	hourly[ "temperature_2m" ] = h_temp
	hourly[ "precipitation" ] = h_precip
	hourly[ "precipitation_prob" ] = h_precip_prob
	hourly[ "cloud" ] = h_cloud
	hourly[ "h_wind_dir" ] = h_wind_dir
	hourly[ "wind" ] = h_wind
	hourly_df = pd.DataFrame( data=hourly )

	print( "\nHourly data\n", hourly_df )


if __name__ == "__main__":
	#main()

	operation = "location = 31.83, 115.77"
	parse( operation )
	operation = "wind > 35"
	parse( operation )
	operation = "wind < 22.22"
	parse( operation )
	operation = "WIND < 22.22"
	ops: Stack = Stack( parse( operation ) )

	while not ops.is_empty():
		op = ops.pop()
		if op in jump_table:
			jump_table[ op ]( ops )

