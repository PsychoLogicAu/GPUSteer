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
		int		m_threadsPerBlock;
		uint	m_doNotApplyWith;

		// Structures containing device pointers.
		AgentGroupData *	m_pAgentGroupData;
		AgentGroupConst *	m_pAgentGroupConst;

	public:
		AbstractCUDAKernel( AgentGroup * pAgentGroup, float const fWeight, uint const doNotApplyWith )
		:	AbstractKernel( pAgentGroup, fWeight ),
			m_threadsPerBlock( THREADSPERBLOCK ),
			m_pAgentGroupData( NULL ),
			m_pAgentGroupConst( NULL ),
			m_doNotApplyWith( doNotApplyWith )
		{
			if( pAgentGroup )
			{
				m_pAgentGroupData = &m_pAgentGroup->GetAgentGroupData();
				m_pAgentGroupConst = &m_pAgentGroup->GetAgentGroupConst();
			}
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