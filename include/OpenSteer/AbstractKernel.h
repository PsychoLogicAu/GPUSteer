#ifndef OPENSTEER_ABSTRACTKERNEL_H
#define OPENSTEER_ABSTRACTKERNEL_H

#include <cuda_runtime.h>
#include "VehicleData.cu"
#include "VehicleGroup.h"

namespace OpenSteer
{
	class AbstractKernel
	{
	protected:
		VehicleGroup*	m_pVehicleGroup;

		inline size_t getDataSizeInBytes(void)
		{
			return sizeof(VehicleData) * getNumberOfAgents();
		}

		inline size_t getConstSizeInBytes(void)
		{
			return sizeof(VehicleConst) * getNumberOfAgents();
		}

		inline int getNumberOfAgents(void)
		{
			if(m_pVehicleGroup != NULL)
				return m_pVehicleGroup->Size();

			return 0;
		}

	public:
		AbstractKernel(VehicleGroup *vehicleGroup)
			: m_pVehicleGroup(vehicleGroup)
		{ }

		virtual void init(void) = 0;
		virtual void run(void) = 0;
		virtual void close(void) = 0;

		virtual void reset(void)
		{
			close();
			init();
		}

		VehicleData* getVehicleData(void)
		{
			if(m_pVehicleGroup != NULL)
				return m_pVehicleGroup->GetVehicleData();

			return NULL;
		}

		VehicleConst* getVehicleConst(void)
		{
			if(m_pVehicleGroup != NULL)
				return m_pVehicleGroup->GetVehicleConst();

			return NULL;
		}
	};
} // namespace OpenSteer
#endif