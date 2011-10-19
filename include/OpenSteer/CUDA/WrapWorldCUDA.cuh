#ifndef OPENSTEER_WRAPWORLDCUDA_CUH
#define OPENSTEER_WRAPWORLDCUDA_CUH

#include "AbstractCUDAKernel.cuh"

namespace OpenSteer
{
class WrapWorldCUDA : public AbstractCUDAKernel
{
protected:
	float3	m_worldSize;

public:
	WrapWorldCUDA( AgentGroup * pAgentGroup, float3 const& worldSize );
	virtual ~WrapWorldCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class WrapWorldCUDA

}	// namespace OpenSteer
#endif
