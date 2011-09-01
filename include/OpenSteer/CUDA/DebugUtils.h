#ifndef OPENSTEER_DEBUG_UTILS_H
#define OPENSTEER_DEBUG_UTILS_H

#include <iostream>
#include <fstream>

template < typename T >
void OutputDeviceData( T const* pdData, size_t const n, std::ostream & os )
{
	// Allocate host data.
	T * phData = (T*)malloc( n * sizeof(T) );

	cudaMemcpy( phData, pdData, n * sizeof(T), cudaMemcpyDeviceToHost );

	for( size_t i = 0; i < n; i++ )
	{
		os << phData[i] << " ";
	}
	os << std::endl;

	free( phData );
}

template < typename T >
void OutputDeviceDataToFile( char const* szFilename, T const* pdData, size_t const n, std::ios_base::openmode mode = std::ios_base::out )
{
	std::ofstream outFile;
	outFile.open( szFilename, mode );

	if( outFile.is_open() )
	{
		OutputDeviceData( pdData, n, outFile );
	}

	outFile.close();
}

#endif
