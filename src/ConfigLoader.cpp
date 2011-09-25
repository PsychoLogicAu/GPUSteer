#include "ConfigLoader.h"

#include <fstream>

using namespace OpenSteer;


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
			_map[ key ] = value;
		}

		inFile.close();
		return true;
	}

	return false;
}
