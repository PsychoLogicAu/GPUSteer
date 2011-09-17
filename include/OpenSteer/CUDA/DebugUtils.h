#ifndef OPENSTEER_DEBUG_UTILS_H
#define OPENSTEER_DEBUG_UTILS_H

#include <iostream>
#include <fstream>

void OutputDebugStringToFile( char const* szString, char const* szFilename = "debug.log" )
{
	std::ofstream of;

	// Open the file to append.
	of.open( szFilename, std::ios_base::app );

	if( of.is_open() )
	{
		of << szString << std::endl;
		of.close();
	}
}

template < typename T >
void OutputDeviceData( T const* pdData, size_t const nEnt, size_t const nElemsPerEnt, std::ostream & os )
{
	// Allocate host data.
	T * phData = (T*)malloc( nEnt * nElemsPerEnt * sizeof(T) );

	cudaMemcpy( phData, pdData, nEnt * nElemsPerEnt * sizeof(T), cudaMemcpyDeviceToHost );

	// For each entity...
	for( size_t i = 0; i < nEnt; i++ )
	{
		// For each element of the entity...
		for( size_t j = 0; j < nElemsPerEnt; j++ )
		{
			os << phData[i*nElemsPerEnt + j] << " ";
		}
		os << std::endl;
	}

	free( phData );
}

template < typename T >
void OutputDeviceDataToFile( char const* szFilename, T const* pdData, size_t const n, size_t const elemsPerLine, std::ios_base::openmode mode = std::ios_base::out )
{
	std::ofstream outFile;
	outFile.open( szFilename, mode );

	if( outFile.is_open() )
	{
		OutputDeviceData( pdData, n, elemsPerLine, outFile );
	}

	outFile.close();
}

#endif
