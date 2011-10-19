#ifndef OPENSTEER_KNNBINNINGV1_CUH
#define OPENSTEER_KNNBINNINGV1_CUH

#include "AbstractCUDAKernel.cuh"

#include "KNNBinData.cuh"
#include "KNNData.cuh"

namespace OpenSteer
{
class KNNBinningV1UpdateDBCUDA : public AbstractCUDAKernel
{
protected:
	BaseGroup *		m_pGroup;
	KNNBinData *	m_pKNNBinData;

public:
	KNNBinningV1UpdateDBCUDA( BaseGroup * pGroup, KNNBinData * pKNNBinData );
	virtual ~KNNBinningV1UpdateDBCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class KNNBinningV1UpdateDBCUDA

class KNNBinningV1CUDA : public AbstractCUDAKernel
{
protected:
	KNNData *		m_pKNNData;
	KNNBinData *	m_pKNNBinData;
	BaseGroup *		m_pOtherGroup;

	uint			m_searchRadius;

public:
	KNNBinningV1CUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinData * pKNNBinData, BaseGroup * pOtherGroup, uint const searchRadius );
	virtual ~KNNBinningV1CUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class KNNBinningV1CUDA
}	// namespace OpenSteer

#endif
