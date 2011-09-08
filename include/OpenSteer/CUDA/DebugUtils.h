#ifndef OPENSTEER_DEBUG_UTILS_H
#define OPENSTEER_DEBUG_UTILS_H

#include <iostream>
#include <fstream>

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

//template <>
//void OutputDeviceData<float3>( float3 const* pdData, size_t const n, size_t const elemsPerLine, std::ostream & os )
//{
//	// Allocate host data.
//	float3 * phData = (float3*)malloc( n * sizeof(float3) );
//
//	cudaMemcpy( phData, pdData, n * sizeof(float3), cudaMemcpyDeviceToHost );
//
//	for( size_t i = 0; i < (n / elemsPerLine); i++ )
//	{
//		for( size_t j = 0; j < elemsPerLine; j++ )
//		{
//			os << phData[i*elemsPerLine + j].x << "," << phData[i*elemsPerLine + j].y << "," << phData[i*elemsPerLine + j].z << std::endl;
//		}
//		os << std::endl;
//	}
//
//	free( phData );
//}

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
