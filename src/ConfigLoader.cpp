#include "OpenSteer/ConfigLoader.h"

#include <fstream>

using namespace OpenSteer;

using namespace std;

Value::Value( void )
:	m_value()
{
}

Value::Value( std::string value )
:	m_value( value )
{
}

Value::~Value( void )
{}

float Value::asFloat( void )
{
	return (float)atof( m_value.c_str() );
}

int Value::asInt( void )
{
	return atoi( m_value.c_str() );
}

ConfigLoader::ConfigLoader( void )
{

}

ConfigLoader::~ConfigLoader( void )
{

}

bool ConfigLoader::LoadFromFile( char const* szFilename )
{
	std::ifstream inFile( szFilename );

	if( inFile.is_open() )
	{
		while( ! inFile.eof() )
		{
			std::string	key;
			std::string	val;

			// Read the key and value.
			std::getline( inFile, key, '=' );
			std::getline( inFile, val );

			// Make sure the two reads succeeded.
			if( ! inFile.good() )
				break;

			Value value( val );

			// Insert into the map.
			m_keyValueMap[ key ] = value;
		}

		inFile.close();
		return true;
	}

	return false;
}
