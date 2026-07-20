'''
	Look for favourable weather conditions for photography (or anything really)

	GD Nov25
'''

from typing import Any, Callable, NamedTuple
import operator

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

import pyparsing as pp
from pyparsing import pyparsing_common as ppc


_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


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


class Expression( NamedTuple ):
	''' Represents a single expression to match against '''
	var: str
	op: str
	val: Any


# ---
class Planner():
	''' The top-level class '''

	def __init__( self ):
		''' Ctor '''
		self._location: (float, float) = None
		self._expressions: Stack = Stack()
		self._hourly_data: pd.DataFrame = None


	def parse( self, operation: str ) -> Stack | None:
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

		if retval[ 0 ] == "location":
			# There's always a special case
			self._location = (retval[ 1 ], retval[ 2 ] )
		else:
			self._expressions.push( retval )

		return retval


	def run( self ) -> None:
		''' It '''

		# Read the config file, which will contain the definitions of what we are planning for

		if self._location is None:
			print( "No location given." )
			return

		# Init
		cache_session = requests_cache.CachedSession( ".cache", expire_after=3600 )
		retry_session = retry( cache_session, retries=5, backoff_factor=0.2 )
		openmeteo = openmeteo_requests.Client( session=retry_session )

		# Map the display name of each variable to it's API name
		fields: { str, str } = { \
			"temperature" : "temperature_2m",
			"rain_prob" : "precipitation_probability",
			"rain_amount" : "precipitation",
			"cloud_total" : "cloud_cover",
			"cloud_low" : "cloud_cover_low",
			"cloud_Mid" : "cloud_cover_mid",
			"cloud_high" : "cloud_cover_high",
			"wind_dir" : "wind_direction_10m",
			"wind" : "wind_speed_10m"
		}

		# Always make the weather call, we will filter the results based on the specified operation
		meteo_params: {} = {
			"latitude" : self._location[ 0 ],
			"longitude" : self._location[ 1 ],
			"hourly" : [ val for val in fields.values() ]
		}

		response = openmeteo.weather_api( "https://api.open-meteo.com/v1/forecast",
									      meteo_params )
		hourly_data = response[ 0 ].Hourly()
		hourly = { "date" : pd.date_range( start=pd.to_datetime( hourly_data.Time(), unit='s', utc=False ),
										   end=pd.to_datetime( hourly_data.TimeEnd(), unit='s', utc=False ),
										   freq = pd.Timedelta( seconds=hourly_data.Interval() ),
										   inclusive="left"
										 )
				  }
		hourly.update( { item[ 0 ]: hourly_data.Variables( idx ).ValuesAsNumpy() \
						 for idx, item in enumerate( fields.items() ) } )

		self._hourly_data = pd.DataFrame( data=hourly )
		print( "\nHourly data\n", self._hourly_data )

		# Filters are always ANDed
		filtered: pd.DataFrame = self._hourly_data
		while not self._expressions.is_empty():
			expr: Expression = Expression._make( self._expressions.pop() )
			filtered = self._filter( filtered, expr )

		filtered.apply( lambda r: print( "%s: wind = %f" % (r[ "date" ], r[ "wind" ]) ),
						axis=1 )
		filtered.apply( lambda r: print( r ),
						axis=1 )


	def _filter( self, df: pd.DataFrame, expr: Expression ) -> pd.DataFrame:
		''' Filters the input dataframe according to the components of the expression '''
		return df[ _OPS[ expr.op ]( df[ expr.var ], expr.val ) ]



if __name__ == "__main__":

	planner: Planner = Planner()

	planner.parse( "location = -31.83, 115.77" )
	planner.parse( "wind < 28" )
	planner.parse( "wind > 22.22" )

	planner.run()

	# ops: Stack = Stack( planner.parse( "WIND < 22.22" ) )
	#
	# while not ops.is_empty():
	# op = ops.pop()
	# 	if op in planner.jump_table:
	#		planner.jump_table[ op ]( planner, ops )
