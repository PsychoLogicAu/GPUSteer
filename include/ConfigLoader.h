#ifndef OPENSTEER_CONFIGLOADER_H
#define OPENSTEER_CONFIGLOADER_H

#include <map>
#include <string>
#include <sstream>

namespace OpenSteer
{

class Value
{
private:
	std::stringstream	_ss;

public:
	Value( std::string _value )
	{
		// Set the stringstream value.
		_ss.str( _value );
	}

	~Value( void )
	{}

	template < typename T >
	void value( T & val )
	{
		_ss >> val;
		_ss.seekg( 0 );
	}
};

class ConfigLoader
{
	typedef std::map< std::string, Value > MapType;
private:
	 MapType	_map;

public:
	ConfigLoader( void );
	~ConfigLoader( void );

	bool LoadFromFile( char const* szFilename );

	template < typename T >
	bool GetValue( std::string key, T & value )
	{
		MapType::iterator it = _map.find( key );

		if( it != _map.end() )
		{
			it->second.value( value );
			return true;
		}

		return false;
	}


};	// class ConfigLoader
}	// namespace OpenSteer
#endif
