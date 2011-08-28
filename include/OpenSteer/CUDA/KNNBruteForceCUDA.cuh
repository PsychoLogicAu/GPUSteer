#ifndef OPENSTEER_KNNBRUTEFORCECUDA_CUH
#define OPENSTEER_KNNBRUTEFORCECUDA_CUH

#include "AbstractCUDAKernel.cuh"

#include "NearestNeighborData.cuh"

namespace OpenSteer
{
class KNNBruteForceCUDA : public AbstractCUDAKernel
{
protected:
	// Temporary device storage.
	float *		m_pdDistanceMatrix;
	size_t *	m_pdIndexMatrix;

public:
	KNNBruteForceCUDA( VehicleGroup * pVehicleGroup );
	virtual ~KNNBruteForceCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class KNNBruteForceCUDA

class KNNBruteForceCUDAV2 : public AbstractCUDAKernel
{
protected:
	NearestNeighborData *	m_pNearestNeighborData;
public:
	KNNBruteForceCUDAV2( VehicleGroup * pVehicleGroup );
	virtual ~KNNBruteForceCUDAV2( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class KNNBruteForceCUDAV2

class KNNBruteForceCUDAV3 : public AbstractCUDAKernel
{
protected:
	NearestNeighborData *	m_pNearestNeighborData;
public:
	KNNBruteForceCUDAV3( VehicleGroup * pVehicleGroup );
	virtual ~KNNBruteForceCUDAV3( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class KNNBruteForceCUDAV3
}	// namespace OpenSteer

#endif
