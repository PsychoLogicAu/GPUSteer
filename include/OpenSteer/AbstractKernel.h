#ifndef OPENSTEER_ABSTRACTKERNEL_H
#define OPENSTEER_ABSTRACTKERNEL_H

#include <cuda_runtime.h>
#include "VehicleGroupData.cu"
#include "VehicleGroup.h"

namespace OpenSteer
{
	class AbstractKernel
	{
	protected:
		VehicleGroup *	m_pVehicleGroup;

		inline size_t getNumAgents( void )
		{
			if( m_pVehicleGroup != NULL )
				return m_pVehicleGroup->Size();

			return 0;
		}

	public:
		AbstractKernel( VehicleGroup * pVehicleGroup )
			: m_pVehicleGroup( pVehicleGroup )
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