#ifndef OPENSTEER_STEERFORSEPARATIONCUDA_H
#define OPENSTEER_STEERFORSEPARATIONCUDA_H

#include "AbstractCUDAKernel.cuh"



namespace OpenSteer
{
class SteerForSeparationCUDA : public AbstractCUDAKernel
{
protected:



public:
	SteerForSeparationCUDA(	VehicleGroup * pVehicleGroup, float const fWeight );
	virtual ~SteerForSeparationCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class SteerForSeparationCUDA
}	// namespace OpenSteer
#endif