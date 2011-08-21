#ifndef OPENSTEER_ABSTRACTCUDAKERNEL_H
#define OPENSTEER_ABSTRACTCUDAKERNEL_H

#include <cuda_runtime.h>

#include "..\AbstractKernel.h"
#include "CUDAGlobals.cuh"
#include "CUDAKernelGlobals.cuh"

#include <cutil_inline.h>

namespace OpenSteer
{
	class AbstractCUDAKernel : public AbstractKernel
	{
	protected:
		int m_threadsPerBlock;

		// Structures containing device pointers.
		VehicleGroupData *	m_pVehicleGroupData;
		VehicleGroupConst *	m_pVehicleGroupConst;

	public:
		AbstractCUDAKernel( VehicleGroup * pVehicleGroup )
		:	AbstractKernel( pVehicleGroup ),
			m_threadsPerBlock( THREADSPERBLOCK )
		{
			m_pVehicleGroupData = &m_pVehicleGroup->GetVehicleGroupData();
			m_pVehicleGroupConst = &m_pVehicleGroup->GetVehicleGroupConst();
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