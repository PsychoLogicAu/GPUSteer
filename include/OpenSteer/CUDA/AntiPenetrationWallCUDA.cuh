#ifndef OPENSTEER_ANTIPENETRATIONWALLCUDA_CUH
#define OPENSTEER_ANTIPENETRATIONWALLCUDA_CUH

#include "AbstractCUDAKernel.cuh"

#include "KNNData.cuh"
#include "../WallGroup.h"

namespace OpenSteer
{
class AntiPenetrationWALLCUDA : public AbstractCUDAKernel
{
protected:
	WallGroup *		m_pWallGroup;
	KNNData *		m_pKNNData;

	float			m_fElapsedTime;

public:
	AntiPenetrationWALLCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, WallGroup * pWallGroup, float const elapsedTime, uint const doNotApplyWith );
	virtual ~AntiPenetrationWALLCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class AntiPenetrationWALLCUDA

}	// namespace OpenSteer
#endif
