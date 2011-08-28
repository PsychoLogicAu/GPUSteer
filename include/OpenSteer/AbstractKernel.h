#ifndef OPENSTEER_ABSTRACTKERNEL_H
#define OPENSTEER_ABSTRACTKERNEL_H

#include <cuda_runtime.h>
#include "VehicleGroupData.cuh"
#include "VehicleGroup.h"

namespace OpenSteer
{
	class AbstractKernel
	{
	protected:
		VehicleGroup *	m_pVehicleGroup;
		float			m_fWeight;

		inline size_t getNumAgents( void )
		{
			if( m_pVehicleGroup != NULL )
				return m_pVehicleGroup->Size();

			return 0;
		}

		inline float getWeight( void )
		{
			return m_fWeight;
		}

	public:
		AbstractKernel( VehicleGroup * pVehicleGroup, float const fWeight = 1.f )
		:	m_pVehicleGroup( pVehicleGroup ),
			m_fWeight( fWeight )
		{ }

		virtual void init(void) = 0;
		virtual void run(void) = 0;
		virtual void close(void) = 0;

		virtual void reset(void)
		{
			close();
			init();
		}

		VehicleGroup * GetVehicleGroup( void )
		{
			return m_pVehicleGroup;
		}
	};
} // namespace OpenSteer
#endif