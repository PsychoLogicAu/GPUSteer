#ifndef OPENSTEER_STEERTOFOLLOWPATHCUDA_CUH
#define OPENSTEER_STEERTOFOLLOWPATHCUDA_CUH

#include "AbstractCUDAKernel.cuh"

#include "PolylinePathwayCUDA.cuh"

namespace OpenSteer
{
	class SteerToFollowPathCUDA : public AbstractCUDAKernel
	{
	protected:
		PolylinePathwayCUDA *		m_pPath;

		float						m_fPredictionTime;

	public:
		SteerToFollowPathCUDA( AgentGroup *pAgentGroup, PolylinePathwayCUDA * pPath, float const predictionTime, float const fWeight, uint const doNotApplyWith );
		virtual ~SteerToFollowPathCUDA( void ) {}

		virtual void init( void );
		virtual void run( void );
		virtual void close(void );
	};	// class SteerToFollowPathCUDA
}	// namespace OpenSteer




#endif
