#ifndef OPENSTEER_ABSTRACTCUDAKERNEL_H
#define OPENSTEER_ABSTRACTCUDAKERNEL_H

#include <cuda_runtime.h>

#include "..\AbstractKernel.h"
#include "CUDAGlobals.h"

namespace OpenSteer
{
	const int THREADSPERBLOCK = 128;

	class AbstractCUDAKernel : public AbstractKernel
	{
	protected:
		int m_threadsPerBlock;

		// Structures containing device pointers.
		VehicleGroupData *	m_pdVehicleGroupData;
		VehicleGroupConst *	m_pdVehicleGroupConst;

	public:
		AbstractCUDAKernel( VehicleGroup * pVehicleGroup )
		:	AbstractKernel( pVehicleGroup ),
			m_threadsPerBlock( THREADSPERBLOCK )
		{
			m_pdVehicleGroupData = &m_pVehicleGroup->GetVehicleGroupData();
			m_pdVehicleGroupConst = &m_pVehicleGroup->GetVehicleGroupConst();
		}

		virtual void init( void ) = 0;
		virtual void run( void ) = 0;
		virtual void close( void ) = 0;

		virtual dim3 gridDim( void )
		{
			return dim3( ( getNumAgents() + m_threadsPerBlock - 1 ) / m_threadsPerBlock );
		}

		virtual dim3 blockDim( void )
		{
			return dim3( m_threadsPerBlock );
		}
	};
}
#endif