#ifndef OPENSTEER_ABSTRACTKERNEL_H
#define OPENSTEER_ABSTRACTKERNEL_H

#include <cuda_runtime.h>
#include "AgentGroupData.cuh"
#include "AgentGroup.h"

namespace OpenSteer
{
	class AbstractKernel
	{
	protected:
		AgentGroup *	m_pAgentGroup;
		float			m_fWeight;

		inline size_t getNumAgents( void )
		{
			if( m_pAgentGroup != NULL )
				return m_pAgentGroup->Size();

			return 0;
		}

	public:
		AbstractKernel( AgentGroup * pAgentGroup, float const fWeight = 1.f )
		:	m_pAgentGroup( pAgentGroup ),
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

		AgentGroup * GetAgentGroup( void )
		{
			return m_pAgentGroup;
		}
	};
} // namespace OpenSteer
#endif