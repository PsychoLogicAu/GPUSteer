#ifndef OPENSTEER_KNNBINNINGV2_CUH
#define OPENSTEER_KNNBINNINGV2_CUH

#include "AbstractCUDAKernel.cuh"

#include "KNNBinData.cuh"
#include "KNNData.cuh"

namespace OpenSteer
{
class KNNBinningV2UpdateDBCUDA : public AbstractCUDAKernel
{
protected:
	BaseGroup *		m_pGroup;
	KNNBinData *	m_pKNNBinData;

public:
	KNNBinningV2UpdateDBCUDA( BaseGroup * pGroup, KNNBinData * pKNNBinData );
	virtual ~KNNBinningV2UpdateDBCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};

class KNNBinningV2CUDA : public AbstractCUDAKernel
{
protected:
	KNNData *		m_pKNNData;
	KNNBinData *	m_pKNNBinData;
	BaseGroup *		m_pOtherGroup;

	uint			m_searchRadius;

public:
	KNNBinningV2CUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinData * pKNNBinData, BaseGroup * pOtherGroup, uint const searchRadius );
	virtual ~KNNBinningV2CUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );




};	// class KNNBinningCUDA
}	// namespace OpenSteer


#endif