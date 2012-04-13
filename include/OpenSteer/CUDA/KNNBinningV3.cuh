#ifndef OPENSTEER_KNNBINNINGV3_CUH
#define OPENSTEER_KNNBINNINGV3_CUH

#include "AbstractCUDAKernel.cuh"

#include "KNNBinDataV3.cuh"
#include "KNNData.cuh"

namespace OpenSteer
{
class KNNBinningV3UpdateDBCUDA : public AbstractCUDAKernel
{
protected:
	BaseGroup *		m_pGroup;
	KNNBinDataV3 *	m_pKNNBinData;

public:
	KNNBinningV3UpdateDBCUDA( BaseGroup * pGroup, KNNBinDataV3 * pKNNBinData );
	virtual ~KNNBinningV3UpdateDBCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};

class KNNBinningV3CUDA : public AbstractCUDAKernel
{
protected:
	KNNData *		m_pKNNData;
	KNNBinDataV3 *	m_pKNNBinData;
	BaseGroup *		m_pOtherGroup;

	uint			m_searchRadius;

public:
	KNNBinningV3CUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinDataV3 * pKNNBinData, BaseGroup * pOtherGroup, uint const searchRadius );
	virtual ~KNNBinningV3CUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );




};	// class KNNBinningCUDA
}	// namespace OpenSteer


#endif