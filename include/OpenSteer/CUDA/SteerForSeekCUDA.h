#ifndef STEERFORSEEKCUDA_H
#define STEERFORSEEKCUDA_H

#include "AbstractCUDAKernel.cuh"

namespace OpenSteer
{
	class SteerForSeekCUDA : public AbstractCUDAKernel
	{
	protected:
		float3		m_target;

	public:
		SteerForSeekCUDA( AgentGroup *pAgentGroup, const float3 &target, float const fWeight );
		~SteerForSeekCUDA( void ) {}

		virtual void init( void );
		virtual void run( void );
		virtual void close(void );
	};
} // namespace OpenSteer
#endif