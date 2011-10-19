#ifndef OPENSTEER_CONFIGLOADER_H
#define OPENSTEER_CONFIGLOADER_H

#include <map>
#include <string>

namespace OpenSteer
{

class Value
{
private:
	std::string	m_value;

public:
	Value( void );
	Value( std::string value );
	~Value( void );

	float asFloat( void );
	int asInt( void );
};

class ConfigLoader
{
	typedef std::map< std::string, Value > MapType;
private:
	 MapType	m_keyValueMap;

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
