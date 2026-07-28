'''
	Look for favourable weather conditions for photography (or anything really)

	GD Nov25
'''

import sys
from typing import Any, Tuple, NamedTuple
import operator
import re
from datetime import datetime

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

import pyparsing as pp
from pyparsing import pyparsing_common as ppc
import astral
from astral import location
from timezonefinder import timezone_at


'''
	The meteo package can do 7, 10, 14 day forecasts. We want to look for a particular time period
	within the forecast window. eg. blue hour, sunrise, sunset, ... These are just a layer of filter
	on the forecasts we get back.
	ie. filter on weather conditions
	AND filter on named time period
'''

_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
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
class Config():
	''' Class to read "config" files. We can't use the standard ConfigParser because we want expressions
		like "wind > 20", whereas ConfigParser only allows key = value pairs
	'''
	def __init__( self ):
		''' Ctor '''
		self._data: { str: [ Expression ] } = {}


	def read( self, fname: str ) -> bool:
		''' Read the given filename and attempt to build a dict of sections and expressions '''
		sect_pattern = re.compile( r"\[(\S+)\]")

		try:
			with open( fname, "rt" ) as ifh:
				cur_sect: str = ""
				for ln in ifh:
					ln = ln.strip()
					if len( ln ) == 0:
						# blank line
						continue
					if ln[ 0 ] in "#;":
						# comment, only allowed at start of line
						continue

					mat = sect_pattern.match( ln )
					if mat is not None:
						# Start of section
						cur_sect = mat[ 1 ]
						self._data[ cur_sect ] = []

					else:
						# An expression within a section (will be parsed later)
						self._data[ cur_sect ].append( ln )

			return True

		except FileNotFoundError:
			print( "DANGER" )
			return False


	def sections( self ) -> str:
		''' Generator to iterate through all sections, returning the section name '''
		for sect in self._data:
			yield sect


	def items( self, section: str ) -> Expression:
		''' Generator to iterate through all conditions in a section '''
		for item in self._data[ section ]:
			yield item


# ---
class Planner():
	''' The top-level class '''

	_atmo_vars: [ str ] = [ \
		"wind",
		"cloud_total",
		"cloud_low",
		"cloud_mid",
		"cloud_high"
	]

	_astro_vars: [ str ] = [
		"golden_hour",
		"blue_hour",
		"moon_phase",
		"moon_percent"
	]

	def __init__( self ):
		''' Ctor '''
		self._location: location.Location | None = None
		self._cond_atmo: [ Expression ] = []		# atmospheric conditions
		self._cond_astro: [ Expression ] = []		# astronomical conditions
		self._hourly_data: pd.DataFrame | None = None

		# Define the expression grammar
		loc_val: pp.ParseExpression = ppc.number + pp.Suppress( ',' ) + ppc.number
		bool_val: pp.ParseExpression = pp.Keyword( "True", caseless=True ).setParseAction( lambda: True ) | \
									   pp.Keyword( "False", caseless=True ).setParseAction( lambda: False )

		vars: [ pp.ParseExpression ] = []
		vars.append( pp.CaselessLiteral( "location" ) + pp.Suppress( '=' ) + loc_val )

		# The atmospheric conditional expressions
		for var in Planner._atmo_vars:
			vars.append( pp.CaselessLiteral( var ) + pp.one_of( "< >" ) + ppc.number )
		# wind: pp.ParseExpression 	 = pp.CaselessLiteral( "wind" ) + pp.one_of( "> <" ) + ppc.number
		# cloud_t: pp.ParseExpression  = pp.CaselessLiteral( "cloud_total" ) + pp.one_of( "> <" ) + ppc.number
		# cloud_l: pp.ParseExpression  = pp.CaselessLiteral( "cloud_low" ) + pp.one_of( "> <" ) + ppc.number
		# cloud_m: pp.ParseExpression  = pp.CaselessLiteral( "cloud_mid" ) + pp.one_of( "> <" ) + ppc.number
		# cloud_h: pp.ParseExpression  = pp.CaselessLiteral( "cloud_high" ) + pp.one_of( "> <" ) + ppc.number

		# The astronomical conditional expressions
		vars.append( pp.CaselessLiteral( "golden_hour" ) + '=' + pp.one_of( "morning evening" ) )
		vars.append( pp.CaselessLiteral( "blue_hour" ) + '=' + pp.one_of( "morning evening" ) )
		vars.append( pp.CaselessLiteral( "moon_phase" ) + '=' + pp.Word( pp.alphas ) )
		vars.append( pp.CaselessLiteral( "moon_percent" ) + pp.one_of( "< >" ) + ppc.number )

		self._grammar: pp.ParseExpression = pp.Or( vars )


	def parse( self, section: str, operation: str ) -> Stack | None:
		''' Parse the operation string into a list of lexical elements and store internally
			in the correct cache
		'''
		lex = self._grammar.parse_string( operation )
		retval = list( lex )

		if retval[ 0 ] == "location":
			self._location = location.Location( \
				astral.LocationInfo( name=section,
									 latitude=retval[ 1 ],
									 longitude=retval[ 2 ],
									 timezone=timezone_at( lng=retval[ 2 ], lat=retval[ 1 ] )
									)
				) 
		elif retval[ 0 ] in Planner._atmo_vars:
			self._cond_atmo.append( retval )
		elif retval[ 0 ] in Planner._astro_vars:
			self._cond_astro.append( retval )
		else:
			print( "Unkown condition: %s" % retval [ 0 ] )

		return retval


	def run( self, cfg_name: str ) -> None:
		''' It '''

		# Read the config file, which will contain the definitions of what we are planning for
		cfg: Config = Config()
		cfg.read( cfg_name )

		for target in cfg.sections():
			print( " === %s ===" % target )
			self._do_init()
			for opt in cfg.items( target ):
				self.parse( target, opt )

			if self._location is None:
				print( "No location given for %s." % target )
				return

			hourly_data: pd.DataFrame = self._get_atomspheric_data()
			if len( hourly_data ) > 0 and len( self._cond_astro ) > 0:
				results: pd.Series = hourly_data.apply( self._filter_astronomical_data, axis=1 )
			else:
				results: pd.Series = pd.Series()

			breakpoint()
			print( "The following date/times match the criteria for %s:" % target )
			for idx in results.index:
				print( idx )


	def _do_init( self ) -> None:
		''' (Re-)initialise vars for each location '''
		self._location = None
		self._cond_atmo = []
		self._cond_astro = []
		self._hourly_data = None


	def _get_atomspheric_data( self ) -> pd.DataFrame:
		''' '''
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
			"cloud_mid" : "cloud_cover_mid",
			"cloud_high" : "cloud_cover_high",
			"wind_dir" : "wind_direction_10m",
			"wind" : "wind_speed_10m"
		}

		# Always make the weather call, we will filter the results based on the specified operation
		meteo_params: {} = {
			"latitude" : self._location.latitude,
			"longitude" : self._location.longitude,
			"hourly" : [ val for val in fields.values() ]
		}

		response = openmeteo.weather_api( "https://api.open-meteo.com/v1/forecast",
									      meteo_params )
		hourly_data = response[ 0 ].Hourly()
		hourly = { "date" : pd.date_range( start=pd.to_datetime( hourly_data.Time(), unit='s', utc=False ),
										   end=pd.to_datetime( hourly_data.TimeEnd(), unit='s', utc=False ),
										   freq = pd.Timedelta( seconds=hourly_data.Interval() ),
										   tz=timezone_at( lng=self._location.longitude, lat=self._location.latitude ),
										   inclusive="left"
										 )
				  }
		hourly.update( { item[ 0 ]: hourly_data.Variables( idx ).ValuesAsNumpy() \
						 for idx, item in enumerate( fields.items() ) } )

		hourly_data = pd.DataFrame( data=hourly )
		hourly_data = hourly_data.set_index( "date" )
		print( "\nHourly data\n", hourly_data )

		# Now filter on specified conditions
		# Filters are always ANDed
		filtered: pd.DataFrame = hourly_data
		for cond in self._cond_atmo:
			expr: Expression = Expression._make( cond )
			filtered = self._filter_col( filtered, expr )
			print( "Atmospheric filter %s-> %d records" % (str( cond ), len( filtered )) )

		return filtered


	def _filter_astronomical_data( self, row: pd.Series ) -> pd.Series | None:
		''' '''
		# This may end up being a callback to apply
		dt: datetime = row.name.to_pydatetime()

		result: bool = True
		for cond in self._cond_astro:
			expr = Expression._make( cond )
			match expr.var:
				case "golden_hour":
					sdir: astral.SunDirection = astral.SunDirection.RISING if expr.val == "morning" \
																		   else astral.SunDirection.SETTING
					rng: Tuple( datetime.datetime, datetime.datetime ) = \
						self._location.golden_hour( sdir, dt.date() )
					print( "%s: golden [%s -- %s]" % (str( dt.date() ), str( rng[ 0 ] ), str( rng[ 1 ] )) )
					result &= dt >= rng[ 0 ] and dt <= rng[ 1 ]

				case "blue_hour" : 
					sdir: astral.SunDirection = astral.SunDirection.RISING if expr.val == "morning" \
																		   else astral.SunDirection.SETTING
					rng: Tuple( datetime.datetime, datetime.datetime ) = \
						self._location.blue_hour( sdir, dt.date() )
					print( "%s: blue [%s -- %s]" % (str( dt.date() ), str( rng[ 0 ] ), str( rng[ 1 ] )) )
					result &= dt >= rng[ 0 ] and dt <= rng[ 1 ]

				case "moon_phase" :
					breakpoint()
					phase: float = self._location.moon_phase( dt )
					result &= (phase < 7.0 and expr.val == "new" ) or \
							  (phase >= 7.0 and phase < 14.0 and expr.val == "first" ) or \
							  (phase >= 14.0 and phase < 21.0 and expr.val == "full" ) or \
							  (phase >= 21 and expr.val == "last" )

				case "moon_percent" :
					phase: float = self._location.moon_phase( dt )
					percent: float = phase * 100.0 / 28.0
					result &= _OPS[ expr.op ]( percent, expr.val )

		return row if result else None


	def _filter_col( self, df: pd.DataFrame, expr: Expression ) -> pd.DataFrame:
		''' Filters on the columns of the input dataframe according to the
			components of the expression
		'''
		return df[ _OPS[ expr.op ]( df[ expr.var ], expr.val ) ]


	def _filter_idx( self, df: pd.DataFrame, expr: Expression ) -> pd.DataFrame:
		''' Filters on the index of the input dataframe according to the components
			of the expression
		'''
		return df[ _OPS[ expr.op ]( df.index, expr.val ) ]



if __name__ == "__main__":

	if len( sys.argv ) > 1:
		planner: Planner = Planner()
		planner.run( sys.argv[ 1 ] )
	else:
		print( "Specifiy config file on command line" )
		sys.exit( -1 )

	'''
	planner.parse( "location = -31.83, 115.77" )
	planner.parse( "wind < 28" )
	planner.parse( "wind > 22.22" )
	planner.parse( "cloud_total < 50" )

	planner.run()

	ops: Stack = Stack( planner.parse( "WIND < 22.22" ) )

	while not ops.is_empty():
	op = ops.pop()
		if op in planner.jump_table:
			planner.jump_table[ op ]( planner, ops )
	'''
