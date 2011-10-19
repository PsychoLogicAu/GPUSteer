#ifndef UPDATECUDA_H
#define UPDATECUDA_H

#include "AbstractCUDAKernel.cuh"

#include "KNNData.cuh"
#include "../WallGroup.h"

namespace OpenSteer
{
	class UpdateCUDA : public AbstractCUDAKernel
	{
	protected:
		float m_fElapsedTime;

		//KNNData *	m_pKNNData;
		//WallGroup *	m_pWallGroup;

	public:
		UpdateCUDA( AgentGroup * pAgentGroup, /*KNNData * pKNNData, WallGroup * pWallGroup,*/ const float fElapsedTime );
		~UpdateCUDA( void ) {}

		virtual void init( void );
		virtual void run( void );
		virtual void close( void );
	};
}

#endif