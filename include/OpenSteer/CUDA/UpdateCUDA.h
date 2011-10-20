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

	public:
		UpdateCUDA( AgentGroup * pAgentGroup, const float fElapsedTime );
		virtual ~UpdateCUDA( void ) {}

		virtual void init( void );
		virtual void run( void );
		virtual void close( void );
	};	// class UpdateCUDA

	class UpdateWithAntiPenetrationCUDA : public AbstractCUDAKernel
	{
	protected:
		float m_fElapsedTime;

		KNNData *	m_pKNNData;
		WallGroup *	m_pWallGroup;

	public:
		UpdateWithAntiPenetrationCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, WallGroup * pWallGroup, const float fElapsedTime );
		virtual ~UpdateWithAntiPenetrationCUDA( void ) {}

		virtual void init( void );
		virtual void run( void );
		virtual void close( void );
	};	// class UpdateWithAntiPenetrationCUDA
}

#endif